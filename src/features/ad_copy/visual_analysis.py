"""이미지의 고신뢰 시각 사실을 분리하는 자동 구성 배너 문구 v1.10."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from src.features.ad_copy.generator import MODEL_CONFIGS, AdCopy, CopyGenerationError, _generate_with_gemini
from src.features.ad_copy.internal.base_rules import _normalize_compact, _terms
from src.features.ad_copy.models import BannerCopyGenerationResult, CopyInputError, CopyValidationIssue, analyze_copy_input
import src.features.ad_copy.internal.base_workflow as base


COPY_PROMPT_VERSION = "gemini_copy_v1.10.0-visual-fact-auto-compose"
VISUAL_PROMPT_VERSION = "gemini_visual_facts_v1.0.0-high-confidence"
VisualStatus = Literal["extracted", "empty", "omitted"]

_ALLOWED_VISUAL_CATEGORIES = {
    "product_type",
    "color",
    "shape",
    "count",
    "container",
    "packaging",
    "visible_component",
    "arrangement",
}
_NON_VISUAL_INFERENCE_PATTERN = re.compile(
    r"(?:맛|향|풍미|식감|바삭|촉촉|쫄깃|부드러운\s*식감|달콤|고소|상큼|"
    r"시원한\s*맛|따뜻한\s*맛|신선|갓\s*(?:만든|구운)|효능|인기|최고)"
)
_VISUAL_REFERENCE_TERMS = {
    "얼음",
    "레몬",
    "빨대",
    "유리잔",
    "컵",
    "접시",
    "상자",
    "포장",
    "토핑",
    "위에",
    "아래에",
    "옆에",
    "얹은",
    "올린",
    "담긴",
}


@dataclass(frozen=True)
class VisualFactExtractionResult:
    verified_facts: tuple[dict[str, str], ...]
    rejected_observations: tuple[dict[str, str], ...]
    excluded_inferences: tuple[str, ...]
    status: VisualStatus
    raw_output: str = ""
    notice: str | None = None
    prompt_version: str = VISUAL_PROMPT_VERSION


VISUAL_FACT_SYSTEM_INSTRUCTIONS = """당신은 광고 문구 생성 전에 상품 사진에서 직접 확인 가능한 시각 정보만 구조화하는 관찰자입니다.

[관찰 원칙]

1. 픽셀에서 명확하게 보이는 상품 종류·색상·형태·개수·용기·포장·구성 요소·공간적 배치만 기록하세요.
2. 메뉴명은 이미지 대상을 찾는 참고 정보일 뿐이며, 메뉴명에 적혔다는 이유로 보이지 않는 내용을 관찰했다고 기록하지 마세요.
3. 맛·향·풍미·식감·온도 체감·신선도·재료의 품질·제조 방식·원산지·효능·인기는 절대 시각 사실로 기록하지 마세요.
4. 갈색으로 보인다고 바삭하다고 판단하거나, 얼음이 보인다고 시원한 맛이라고 판단하지 마세요.
5. 확신이 높지 않으면 confidence를 medium 또는 low로 표시하세요. 문구 생성에는 high만 사용됩니다.
6. 하나의 observation에는 한 가지 짧은 사실만 넣으세요.

[허용 category]

- product_type: 사진에서 분명한 상품 종류
- color: 명확한 색상
- shape: 명확한 형태
- count: 명확히 셀 수 있는 개수
- container: 잔·컵·접시 등 용기
- packaging: 상자·봉투 등 포장
- visible_component: 얼음·장식·토핑처럼 실제로 보이는 구성 요소
- arrangement: 위·아래·옆처럼 명확한 공간적 배치

JSON 객체만 반환하세요."""


VISUAL_FACT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "minItems": 0,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "value": {"type": "string", "minLength": 1, "maxLength": 30},
                    "confidence": {"type": "string"},
                },
                "required": ["category", "value", "confidence"],
                "additionalProperties": False,
            },
        },
        "excluded_inferences": {
            "type": "array",
            "minItems": 0,
            "maxItems": 10,
            "items": {"type": "string", "maxLength": 40},
        },
    },
    "required": ["observations", "excluded_inferences"],
    "additionalProperties": False,
}


def build_visual_fact_prompt(menu_name: str) -> str:
    return (
        f"시각 정보 프롬프트 버전: {VISUAL_PROMPT_VERSION}\n"
        "첨부된 상품 사진에서 직접 확인 가능한 사실만 추출하세요.\n"
        + json.dumps(
            {
                "menu_name_for_reference": menu_name,
                "allowed_categories": sorted(_ALLOWED_VISUAL_CATEGORIES),
                "high_confidence_policy": "명확히 보이는 경우에만 high",
                "forbidden_inferences": [
                    "맛",
                    "향",
                    "풍미",
                    "식감",
                    "온도 체감",
                    "신선도",
                    "제조 방식",
                    "효능",
                    "인기",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_visual_fact_response(output_text: str) -> VisualFactExtractionResult:
    try:
        payload = json.loads(output_text)
        observations = payload["observations"]
        excluded = payload["excluded_inferences"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CopyGenerationError("이미지 관찰 응답을 구조화된 형식으로 읽을 수 없습니다.") from exc
    if not isinstance(observations, list) or not isinstance(excluded, list):
        raise CopyGenerationError("이미지 관찰 응답의 배열 형식이 올바르지 않습니다.")

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in observations:
        if not isinstance(raw, dict):
            continue
        category = str(raw.get("category") or "").strip()
        value = " ".join(str(raw.get("value") or "").split())
        confidence = str(raw.get("confidence") or "").strip().casefold()
        reason = ""
        if category not in _ALLOWED_VISUAL_CATEGORIES:
            reason = "허용되지 않은 시각 정보 범주"
        elif not value or len(value) > 30:
            reason = "비어 있거나 너무 긴 관찰"
        elif confidence != "high":
            reason = "고신뢰 관찰이 아님"
        elif _NON_VISUAL_INFERENCE_PATTERN.search(value):
            reason = "사진만으로 확인할 수 없는 감각·품질 추론"
        signature = _normalize_compact(f"{category}:{value}")
        record = {
            "category": category,
            "value": value,
            "confidence": confidence,
        }
        if reason:
            rejected.append({**record, "reason": reason})
        elif signature and signature not in seen:
            seen.add(signature)
            accepted.append({
                "id": f"v{len(accepted) + 1}",
                **record,
            })

    excluded_values = tuple(
        dict.fromkeys(
            " ".join(str(value).split())
            for value in excluded
            if str(value).strip()
        )
    )
    return VisualFactExtractionResult(
        verified_facts=tuple(accepted),
        rejected_observations=tuple(rejected),
        excluded_inferences=excluded_values,
        status="extracted" if accepted else "empty",
        raw_output=output_text,
        notice=(
            None
            if accepted
            else "문구에 사용할 고신뢰 시각 정보를 찾지 못했습니다."
        ),
    )


def extract_verified_visual_facts(
    *,
    menu_name: str,
    model_id: str,
    api_key: str,
    image_bytes: bytes,
    image_mime_type: str,
    client: Any | None = None,
) -> VisualFactExtractionResult:
    try:
        output_text = _generate_with_gemini(
            prompt=build_visual_fact_prompt(menu_name),
            model_id=model_id,
            api_key=api_key,
            client=client,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            system_instructions=VISUAL_FACT_SYSTEM_INSTRUCTIONS,
            response_schema=VISUAL_FACT_RESPONSE_SCHEMA,
        )
        return parse_visual_fact_response(output_text)
    except Exception as exc:
        return VisualFactExtractionResult(
            verified_facts=(),
            rejected_observations=(),
            excluded_inferences=(),
            status="omitted",
            notice=f"시각 정보 추출을 생략했습니다: {type(exc).__name__}: {exc}",
        )


def attach_visual_context(
    ad_input: dict[str, Any],
    visual_result: VisualFactExtractionResult,
) -> dict[str, Any]:
    enriched = deepcopy(ad_input)
    campaign = enriched.setdefault("campaign", {})
    campaign["verified_visual_facts"] = [
        dict(fact) for fact in visual_result.verified_facts
    ]
    campaign["rejected_visual_observations"] = [
        dict(fact) for fact in visual_result.rejected_observations
    ]
    campaign["excluded_visual_inferences"] = list(
        visual_result.excluded_inferences
    )
    campaign["visual_extraction_status"] = visual_result.status
    return enriched


BANNER_COPY_SYSTEM_INSTRUCTIONS = base.BANNER_COPY_SYSTEM_INSTRUCTIONS + """

[검증된 시각 정보 사용 규칙]

1. 원본 이미지 자체를 새로운 문구 사실의 근거로 다시 해석하지 마세요. 문구에서 사용할 수 있는 시각 정보는 verified_visual_facts에 전달된 고신뢰 관찰뿐입니다.
2. verified_visual_facts는 상품 종류·색상·형태·개수·용기·포장·보이는 구성·배치를 묘사할 때만 선택적으로 사용할 수 있습니다.
3. 시각 정보는 모든 후보에 억지로 넣지 마세요. 제목의 메시지와 자연스럽게 연결될 때만 사용하세요.
4. 얼음이 보인다는 사실을 ‘시원한 맛·청량한 풍미’로, 갈색 표면을 ‘바삭한 식감’으로 바꾸지 마세요.
5. 맛·향·풍미·식감은 verified_product_facts만 근거로 사용하세요.
6. headline_and_subcopy에서는 제목이 추상적인 관심 메시지만 전달해도 됩니다. 보조 문구가 확인된 상품 특징을 모두 반영하면 제목에 상품 사실을 다시 강제하지 마세요.
7. 친근한 보조 문구는 ‘특징이에요’ 같은 설명형 해요체를 사용할 수 있지만 ‘즐겨보세요·느껴보세요’ 같은 명령형 CTA는 사용하지 마세요.
8. 보조 문구에는 verified_product_facts의 표현을 우선 사용하고, 입력에 없는 수식어를 덧붙이지 마세요.
"""


def _visual_facts(ad_input: dict[str, Any]) -> list[dict[str, str]]:
    values = ad_input.get("campaign", {}).get("verified_visual_facts", [])
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, dict)]


def _prompt_payload(profile: Any, ad_input: dict[str, Any]) -> dict[str, Any]:
    payload = base._prompt_payload(profile)
    payload["product_image"] = {
        "provided": profile.image_provided,
        "role": "원본 재해석 금지. 아래 verified_visual_facts를 추출하는 데만 사용됨",
    }
    payload["verified_visual_facts"] = {
        "facts": _visual_facts(ad_input),
        "role": "사진에서 직접 확인한 고신뢰 시각 사실. 선택적으로만 사용",
        "may_support": ["종류", "색상", "형태", "개수", "용기", "포장", "보이는 구성", "배치"],
        "must_not_support": ["맛", "향", "풍미", "식감", "온도 체감", "신선도", "효능"],
    }
    campaign = ad_input.get("campaign", {})
    payload["excluded_visual_inferences"] = campaign.get(
        "excluded_visual_inferences", []
    )
    if profile.generation_mode == "headline_and_subcopy":
        payload["auto_composition"]["headline"] = (
            "관심 메시지. 제품 사실이나 시각 사실을 반드시 포함할 필요 없음"
        )
        payload["auto_composition"]["subcopy"] = (
            "verified_product_facts를 빠짐없이 설명. 시각 사실로 맛·향·식감을 보충하지 않음"
        )
    return payload


def build_banner_copy_prompt(
    ad_input: dict[str, Any],
    *,
    image_provided: bool = False,
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=image_provided)
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "검증된 상품 사실과 별도로 추출된 고신뢰 시각 사실을 구분해 광고 문구 선택지 2개를 작성하세요. "
        "사용자가 제목과 보조 문구 역할을 지정하지 않아도 자동 구성하세요.\n"
        + json.dumps(_prompt_payload(profile, ad_input), ensure_ascii=False, indent=2)
    )


def build_replacement_prompt(
    ad_input: dict[str, Any],
    copies: list[AdCopy],
    option_indices: list[int],
    issues: list[CopyValidationIssue],
    *,
    image_provided: bool = False,
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=image_provided)
    replacing = set(option_indices)
    context = {
        **_prompt_payload(profile, ad_input),
        "repair": True,
        "locked_options": [
            {
                "option_index": index,
                "headline": copy["headline"],
                "subcopy": copy["body"] or None,
            }
            for index, copy in enumerate(copies, start=1)
            if index not in replacing
        ],
        "replace_option_indices": option_indices,
        "violations": [
            {
                "option_index": issue.option_index,
                "field": issue.field,
                "code": issue.code,
                "message": issue.message,
            }
            for issue in issues
            if issue.option_index in replacing
        ],
        "repair_priority": [
            "보조 문구에는 verified_product_facts의 핵심어를 모두 유지",
            "입력에 없는 감각·강도 수식어 제거",
            "명령형 CTA 대신 명사구 또는 설명형 문장 사용",
            "locked_options와 다른 제목 작성",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "실패한 선택지만 교체하세요. locked_options는 다시 출력하거나 수정하지 말고, "
        "replace_option_indices 순서에 맞춘 교체 선택지만 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _visual_fact_reflected(text: str, ad_input: dict[str, Any]) -> bool:
    text_terms = set(_terms(text))
    for fact in _visual_facts(ad_input):
        fact_terms = [
            term for term in _terms(str(fact.get("value") or ""))
            if len(term) >= 2
        ]
        if fact_terms and any(term in text_terms for term in fact_terms):
            return True
    return False


def _unverified_visual_terms(text: str, ad_input: dict[str, Any]) -> list[str]:
    allowed_source = " ".join(
        [
            str(ad_input.get("product", {}).get("name") or ""),
            str(ad_input.get("product", {}).get("features") or ""),
        ]
        + [str(fact.get("value") or "") for fact in _visual_facts(ad_input)]
    )
    normalized_allowed = _normalize_compact(allowed_source)
    return [
        term for term in _VISUAL_REFERENCE_TERMS
        if term in text and _normalize_compact(term) not in normalized_allowed
    ]


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    profile = analyze_copy_input(ad_input)
    base_issues = base.collect_banner_copy_issues(copies, ad_input)
    issues: list[CopyValidationIssue] = []
    for issue in base_issues:
        if (
            issue.code == "HEADLINE_NOT_GROUNDED"
            and profile.generation_mode == "headline_and_subcopy"
        ):
            continue
        if (
            issue.code == "HEADLINE_NOT_GROUNDED"
            and issue.option_index is not None
            and _visual_fact_reflected(
                copies[issue.option_index - 1]["headline"], ad_input
            )
        ):
            continue
        issues.append(issue)

    for index, copy in enumerate(copies, start=1):
        for field, text in (("headline", copy["headline"]), ("subcopy", copy["body"])):
            unknown = _unverified_visual_terms(text, ad_input)
            if unknown:
                issues.append(CopyValidationIssue(
                    "UNVERIFIED_VISUAL_REFERENCE",
                    f"선택지 {index} {field}에 검증되지 않은 시각 표현이 있습니다: {', '.join(unknown)}",
                    index,
                    field,
                ))

    deduplicated: list[CopyValidationIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for issue in issues:
        key = (issue.code, issue.message, issue.option_index, issue.field)
        if key not in seen:
            seen.add(key)
            deduplicated.append(issue)
    return deduplicated


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> BannerCopyGenerationResult:
    profile = analyze_copy_input(ad_input, image_provided=image_bytes is not None)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider != "gemini":
        raise CopyGenerationError("배너 문구 생성은 Gemini 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    copies: list[AdCopy] | None = None
    issues: list[CopyValidationIssue] = []
    first_issues: tuple[str, ...] = ()
    history: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        replacement_indices = sorted({
            issue.option_index
            for issue in issues
            if issue.option_index is not None
        })
        targeted = bool(copies is not None and replacement_indices)
        option_count = len(replacement_indices) if targeted else 2
        prompt = (
            build_replacement_prompt(
                ad_input,
                copies,
                replacement_indices,
                issues,
                image_provided=image_bytes is not None,
            )
            if targeted and copies is not None
            else build_banner_copy_prompt(
                ad_input,
                image_provided=image_bytes is not None,
            )
        )
        output_text = ""
        try:
            output_text = _generate_with_gemini(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                system_instructions=BANNER_COPY_SYSTEM_INSTRUCTIONS,
                response_schema=base._response_schema(profile, option_count),
            )
            returned = base.parse_banner_copy_options(
                output_text,
                ad_input,
                expected_count=option_count,
            )
            if targeted and copies is not None:
                merged = [dict(copy) for copy in copies]
                for option_index, replacement in zip(replacement_indices, returned):
                    merged[option_index - 1] = replacement
                copies = merged
            else:
                copies = returned
            issues = collect_banner_copy_issues(copies, ad_input)
            history.append({
                "attempt": attempt,
                "repair": targeted,
                "replace_option_indices": replacement_indices,
                "raw_output": output_text,
                "options": [
                    {
                        "headline": copy["headline"],
                        "headline_length": len(copy["headline"]),
                        "subcopy": copy["body"],
                        "subcopy_length": len(copy["body"]),
                    }
                    for copy in copies
                ],
                "issues": [issue.message for issue in issues],
            })
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            history.append({
                "attempt": attempt,
                "repair": targeted,
                "replace_option_indices": replacement_indices,
                "raw_output": output_text,
                "options": [],
                "issues": [error_message],
            })
            if not first_issues:
                first_issues = (error_message,)
            copies = None
            issues = []
            continue

        if not issues:
            return BannerCopyGenerationResult(
                copies=copies,
                generation_mode=profile.generation_mode,
                status="generated" if attempt == 1 else "retried",
                attempts=attempt,
                notice=None if attempt == 1 else "실패한 선택지만 다시 생성했습니다.",
                validation_issues=first_issues,
                attempt_history=tuple(history),
                prompt_version=COPY_PROMPT_VERSION,
            )
        if not first_issues:
            first_issues = tuple(issue.message for issue in issues)

    fallback = base._safe_fallback_options(ad_input)
    return BannerCopyGenerationResult(
        copies=fallback,
        generation_mode=profile.generation_mode,
        status="fallback",
        attempts=max_attempts,
        notice="모델 문구가 검증을 통과하지 못해 확인된 입력 기반 문구를 사용했습니다.",
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=COPY_PROMPT_VERSION,
    )
