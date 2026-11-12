"""이미지 분석 캡션과 광고 카피를 분리하는 문구 파이프라인 v1.10.16.

Gemini의 시각 정보는 원본 판독 기록과 표현 검증에 보존한다. 문구 생성기에는 위치·배치·
용기·단순 색상 설명을 제외하고, 고객이 알 가치가 있는 시각 요소만 정규화한
``copy_eligible_visual_facts``로 전달한다. 상품명은 배너의 별도 슬롯에 표시되므로
제목에서는 반복하지 않고, 제품 특징은 보조 문구에 원문 기반으로 보존한다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import re
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    _generate_with_gemini_generate_content as _generate_with_gemini,
    _generate_with_openai,
    build_deterministic_hashtags,
)
from src.features.ad_copy.promotion import (
    BADGE_ONLY,
    copy_input_for_promotion_policy,
    headline_is_exact_promotion,
    headline_mentions_promotion,
    normalize_promotion_copy_mode,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.natural_request as previous


COPY_PROMPT_VERSION = "gpt-copy-v1.10.16-caption-separated-natural-qc"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = 4
FINAL_OPTION_COUNT = 2

_CAPTION_LEAK_PATTERN = re.compile(
    r"(?:보이는|확인된|확인되는|돋보이는\s*구성|"
    r"특징인\s*구성|상단의|하단의|표면에|배치된|"
    r"과자로\s*확인|종류로\s*확인|이미지에서|사진에서)"
)
_AWKWARD_PATTERN = re.compile(
    r"(?:여백을\s*두는|여백을\s*담은|여백에\s*머무는\s*말차|"
    r"과자\s*종류의|음료이\s*있는|메뉴예요\.?$)"
)
_INCOMPLETE_ENDINGS = (
    "하는", "되는", "담은", "채운", "마시는", "즐기는", "위한", "감도는", "이어지는", "전하는",
)
_SENSORY_ROOTS = (
    "달콤", "새콤", "쌍싸름", "쓸쓸", "바삭", "촉촉", "부드러", "쫄깃",
    "담백", "진한", "깔끔", "향", "풍미", "식감", "크림", "캐러멜",
)
_SAFE_VISUAL_COMPONENT_ROOTS = (
    "과일", "딸기", "망고", "레몬", "오렌지", "블루베리", "바나나",
    "초콜릿", "견과", "크림", "아이스크림", "쿠키", "캐러멜", "토핑", "장식",
)
_LOW_VALUE_VISUAL_ROOTS = (
    "가루", "시럽", "컵", "잔", "접시", "받침", "용기", "손잡이", "얼음",
)


def _provider_copy_prompt_version(provider: str) -> str:
    return f"{provider}_copy_v1.10.16-caption-separated-natural-qc"


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _tokens(value: str) -> list[str]:
    return [
        _normalize(token)
        for token in re.findall(r"[0-9A-Za-z가-힣]+", value)
        if _normalize(token)
    ]


def _normalize_visual_value(value: str) -> str:
    cleaned = " ".join(value.split())
    cleaned = re.sub(r"^(?:상단|ud558단|ud45c면|uc704|uc544래|uc88c측|uc6b0측)(?:에|의|에서)\s*", "", cleaned)
    cleaned = re.sub(r"\s*(?:이|가|uc740|는)?\s*(?:보임|보이는|확인됨|확인되는|놓여\s*있음)$", "", cleaned)
    cleaned = re.sub(r"\s*(?:이|가)?\s*특징인\s*구성$", "", cleaned)
    return cleaned.strip(" ,.")


def advertising_safe_visual_facts(ad_input: dict[str, Any]) -> list[dict[str, str]]:
    """사진 판독문이 아닌 광고 참고용 상품 요소만 선별한다."""

    raw_facts = ad_input.get("campaign", {}).get("verified_visual_facts") or []
    safe: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "")
        value = _normalize_visual_value(str(item.get("value") or ""))
        compact = _normalize(value)
        if category not in {"visible_component", "packaging", "product_type"}:
            continue
        if category == "product_type" and compact in {"음료", "커피", "디저트", "빵", "과자"}:
            continue
        if not value or has_caption_leak(value):
            continue
        if any(root in compact for root in _LOW_VALUE_VISUAL_ROOTS):
            continue
        if category == "visible_component" and not any(
            root in compact for root in _SAFE_VISUAL_COMPONENT_ROOTS
        ):
            continue
        if compact in seen:
            continue
        seen.add(compact)
        safe.append({"category": category, "value": value})
    return safe


def _copy_input_without_visual_caption(ad_input: dict[str, Any]) -> dict[str, Any]:
    """원본 캡션을 구조적으로 분리하고 정제된 시각 사실만 전달한다."""

    prepared = deepcopy(copy_input_for_promotion_policy(ad_input))
    campaign = prepared.setdefault("campaign", {})
    campaign["visual_analysis_facts"] = deepcopy(
        ad_input.get("campaign", {}).get("verified_visual_facts") or []
    )
    campaign["copy_eligible_visual_facts"] = advertising_safe_visual_facts(ad_input)
    campaign["verified_visual_facts"] = []
    campaign["visual_facts_for_copy"] = []
    campaign["visual_caption_usage"] = "filtered_ad_safe_facts_only"
    return prepared


def headline_repeats_menu_name(headline: str, menu_name: str) -> bool:
    compact_headline = _normalize(headline)
    compact_menu = _normalize(menu_name)
    if compact_menu and compact_menu in compact_headline:
        return True
    menu_tokens = list(dict.fromkeys(_tokens(menu_name)))
    return bool(menu_tokens) and all(token in compact_headline for token in menu_tokens)


def has_caption_leak(text: str) -> bool:
    return bool(_CAPTION_LEAK_PATTERN.search(" ".join(text.split())))


def has_awkward_expression(text: str) -> bool:
    normalized = " ".join(text.split()).rstrip(" .!?\n")
    if not normalized:
        return False
    return bool(_AWKWARD_PATTERN.search(normalized)) or normalized.endswith(
        _INCOMPLETE_ENDINGS
    )


def _unsupported_sensory_claim(headline: str, feature: str) -> bool:
    normalized_headline = _normalize(headline)
    normalized_feature = _normalize(feature)
    return any(
        root in normalized_headline and root not in normalized_feature
        for root in _SENSORY_ROOTS
    )


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    """레이아웃 인지형 QC: 상품명은 제목에서 제외하고 특징은 본문에 보존한다."""

    prepared = _copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    product = prepared.get("product", {})
    menu_name = str(product.get("name") or "").strip()
    feature = " ".join(str(product.get("features") or "").split())
    campaign = prepared.get("campaign", {})
    promotion = str(campaign.get("promotion") or "").strip()
    mode = normalize_promotion_copy_mode(campaign.get("promotion_copy_mode"))
    issues: list[CopyValidationIssue] = []

    if len(copies) != FINAL_OPTION_COUNT:
        issues.append(CopyValidationIssue(
            "OPTION_COUNT",
            f"최종 문구는 {FINAL_OPTION_COUNT}개여야 합니다.",
        ))

    seen: set[tuple[str, str]] = set()
    for index, copy in enumerate(copies, start=1):
        headline = " ".join(str(copy.get("headline") or "").split())
        body = " ".join(str(copy.get("body") or "").split())
        if not headline:
            issues.append(CopyValidationIssue(
                "MISSING_HEADLINE", f"선택지 {index} 제목이 비어 있습니다.", index, "headline"
            ))
        elif len(headline) > profile.limits.headline:
            issues.append(CopyValidationIssue(
                "HEADLINE_LENGTH", f"선택지 {index} 제목이 길이 제한을 넘습니다.", index, "headline"
            ))
        if feature and not body:
            issues.append(CopyValidationIssue(
                "MISSING_SUBCOPY", f"선택지 {index} 보조 문구가 비어 있습니다.", index, "subcopy"
            ))
        elif len(body) > profile.limits.subcopy:
            issues.append(CopyValidationIssue(
                "SUBCOPY_LENGTH", f"선택지 {index} 보조 문구가 길이 제한을 넘습니다.", index, "subcopy"
            ))

        if headline_repeats_menu_name(headline, menu_name):
            issues.append(CopyValidationIssue(
                "MENU_NAME_REPEAT",
                f"선택지 {index} 제목이 별도로 표시될 메뉴명을 반복합니다.",
                index,
                "headline",
            ))
        for field, value in (("headline", headline), ("subcopy", body)):
            if has_caption_leak(value):
                issues.append(CopyValidationIssue(
                    "IMAGE_CAPTION_LEAK",
                    f"선택지 {index} {field}에 사진 판독형 표현이 있습니다.",
                    index,
                    field,
                ))
            if has_awkward_expression(value):
                issues.append(CopyValidationIssue(
                    "AWKWARD_EXPRESSION",
                    f"선택지 {index} {field}이 광고 문구로 자연스럽지 않습니다.",
                    index,
                    field,
                ))

        if feature and _normalize(feature) not in _normalize(body):
            issues.append(CopyValidationIssue(
                "PRODUCT_FACT_NOT_PRESERVED",
                f"선택지 {index} 보조 문구에 확인된 상품 특징이 보존되지 않았습니다.",
                index,
                "subcopy",
            ))
        if _unsupported_sensory_claim(headline, feature):
            issues.append(CopyValidationIssue(
                "UNSUPPORTED_SENSORY_HEADLINE",
                f"선택지 {index} 제목에 입력 특징에 없는 감각 표현이 있습니다.",
                index,
                "headline",
            ))
        if mode == BADGE_ONLY and (
            headline_mentions_promotion(headline)
            or headline_mentions_promotion(body)
        ):
            issues.append(CopyValidationIssue(
                "PROMOTION_IN_COPY",
                f"선택지 {index}에 배지 전용 프로모션이 반복됩니다.",
                index,
            ))
        if promotion and mode != BADGE_ONLY and headline_mentions_promotion(headline):
            if not headline_is_exact_promotion(headline, promotion):
                issues.append(CopyValidationIssue(
                    "PROMOTION_REWRITTEN",
                    f"선택지 {index} 프로모션 조건이 원문과 다릅니다.",
                    index,
                    "headline",
                ))

        key = (_normalize(headline), _normalize(body))
        if key in seen:
            issues.append(CopyValidationIssue(
                "DUPLICATE_COPY", f"선택지 {index}가 다른 선택지와 중복됩니다.", index
            ))
        seen.add(key)
        if headline and body and (
            _normalize(headline) in _normalize(body)
            or _normalize(body) in _normalize(headline)
        ):
            issues.append(CopyValidationIssue(
                "HEADLINE_BODY_REPEAT",
                f"선택지 {index} 제목과 보조 문구가 같은 말을 반복합니다.",
                index,
            ))
    return issues


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    prepared = _copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    product = prepared.get("product", {})
    campaign = prepared.get("campaign", {})
    payload = {
        "prompt_version": COPY_PROMPT_VERSION,
        "product_name_rendered_separately": str(product.get("name") or ""),
        "verified_product_feature_for_subcopy": str(product.get("features") or ""),
        "requested_tone": str(campaign.get("tone") or ""),
        "copy_request": str(campaign.get("copy_request") or ""),
        "copy_eligible_visual_facts": list(campaign.get("copy_eligible_visual_facts") or []),
        "headline_limit": profile.limits.headline,
        "candidate_count": INTERNAL_CANDIDATE_COUNT,
        "rules": [
            "상품명은 배너에 별도로 크게 표시되므로 제목에 상품명 전체나 구성 단어를 반복하지 않는다.",
            "제목은 맛을 임의로 추가하지 않고, 입력된 톤·상황·즐기는 순간을 말하는 짧은 훅으로 작성한다.",
            "copy_eligible_visual_facts는 Gemini 결과에서 광고 가치가 있는 요소만 선별한 참고 정보다. 선택적으로 사용하되 위치·보임·구성을 설명하는 문장으로 복사하지 않는다.",
            "보이는, 확인된, 구성, 상단, 하단, 표면, 배치 같은 사진 판독형 단어를 사용하지 않는다.",
            "의미가 불분명한 시적 조합이나 뒤에 명사가 필요한 관형형으로 끝내지 않는다.",
            "프로모션·가격·사진의 시각 캡션은 제목에 넣지 않는다.",
        ],
        "output": "서로 다른 자연스러운 한국어 headline 4개",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "minItems": INTERNAL_CANDIDATE_COUNT,
            "maxItems": INTERNAL_CANDIDATE_COUNT,
            "items": {
                "type": "object",
                "properties": {"headline": {"type": "string"}},
                "required": ["headline"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["options"],
    "additionalProperties": False,
}

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "minItems": INTERNAL_CANDIDATE_COUNT,
            "maxItems": INTERNAL_CANDIDATE_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 1, "maximum": 4},
                    "meaningful": {"type": "boolean"},
                    "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "ad_value": {"type": "integer", "minimum": 1, "maximum": 5},
                    "grounded": {"type": "boolean"},
                    "menu_name_repeated": {"type": "boolean"},
                    "descriptive_caption": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "index", "meaningful", "naturalness", "ad_value", "grounded",
                    "menu_name_repeated", "descriptive_caption", "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}

_SYSTEM_INSTRUCTIONS = """
너는 카페 광고의 한국어 훅 문구 편집자다. 상품명은 별도 영역에 표시되므로
제목에서 반복하지 마라. 사진을 설명하는 판독 문장을 쓰지 마라. 입력에 없는
맛·향·식감·재료·효과를 추가하지 마라. 의미가 완결되고 일상적으로 자연스러운
문장만 작성하라. JSON 스키마와 정확히 일치하는 결과만 반환하라.
""".strip()

_JUDGE_INSTRUCTIONS = """
너는 한국어 광고 카피 QC 감수자다. 각 제목이 의미가 통하는지, 자연스러운지,
짧은 광고 훅으로 쓸 수 있는지, 입력 톤과 상품 특징의 범위 안에 있는지를 엄격하게
판정한다. 상품명 반복과 사진 판독형 문장은 탈락으로 판정한다. JSON만 반환하라.
""".strip()


def _call_model(
    *,
    prompt: str,
    model_id: str,
    api_key: str,
    client: Any | None,
    schema: dict[str, Any],
    instructions: str,
) -> tuple[str, dict[str, int]]:
    config = MODEL_CONFIGS[model_id]
    if config.provider == "openai":
        return _generate_with_openai(
            prompt=prompt,
            model_id=model_id,
            api_key=api_key,
            client=client,
            system_instructions=instructions,
            response_schema=schema,
        )
    if config.provider == "gemini":
        output = _generate_with_gemini(
            prompt=prompt,
            model_id=model_id,
            api_key=api_key,
            client=client,
            system_instructions=instructions,
            response_schema=schema,
        )
        return output, {}
    raise CopyGenerationError("배너 문구는 OpenAI 또는 Gemini 모델만 지원합니다.")


def _fallback_hooks(ad_input: dict[str, Any]) -> list[str]:
    campaign = ad_input.get("campaign", {})
    tone = str(campaign.get("tone") or "")
    request = str(campaign.get("copy_request") or "")
    context = f"{tone} {request}"
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.split())
        menu_name = str(ad_input.get("product", {}).get("name") or "")
        if (
            4 <= len(cleaned) <= analyze_copy_input(ad_input, image_provided=False).limits.headline
            and not headline_repeats_menu_name(cleaned, menu_name)
            and not has_caption_leak(cleaned)
            and not has_awkward_expression(cleaned)
            and cleaned not in values
        ):
            values.append(cleaned)

    if "대비" in context or "조합" in context:
        add("서로 다른 매력의 조화")
        add("두 매력이 만나는 순간")
    if "여백" in context or "차분" in context:
        add("여유가 머무는 순간")
        add("천천히 즐기는 시간")
    if "향" in context:
        add("향긋함이 머무는 순간")
    if "식감" in context:
        add("한입마다 살아나는 식감")
    if any(term in context for term in ("활기", "경쾌", "밝고")):
        add("기분 좋게 즐기는 순간")
        add("즐거움을 더하는 시간")
    if any(term in context for term in ("따뜻", "감성", "포근")):
        add("포근하게 즐기는 순간")
    if any(term in context for term in ("고급", "세련")):
        add("천천히 음미하는 순간")
    add("오늘을 위한 작은 즐거움")
    add("천천히 즐기는 기분 좋은 시간")
    return values


def _candidate_copy(headline: str, feature: str, hashtags: list[str]) -> AdCopy:
    return {
        "headline": " ".join(headline.strip(" ,.!?").split()),
        "body": " ".join(feature.strip(" .").split()),
        "hashtags": hashtags.copy(),
    }


def _local_candidate_issues(copy: AdCopy, ad_input: dict[str, Any]) -> list[CopyValidationIssue]:
    return [
        issue
        for issue in collect_banner_copy_issues([copy, {
            "headline": "임시 선택지 구분용",
            "body": copy["body"],
            "hashtags": copy["hashtags"],
        }], ad_input)
        if issue.option_index == 1
    ]


def _judge_candidates(
    candidates: list[AdCopy],
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, int]]:
    prompt = json.dumps({
        "product_name_rendered_separately": ad_input.get("product", {}).get("name"),
        "verified_product_feature": ad_input.get("product", {}).get("features"),
        "requested_tone": ad_input.get("campaign", {}).get("tone"),
        "copy_request": ad_input.get("campaign", {}).get("copy_request"),
        "copy_eligible_visual_facts": list(
            ad_input.get("campaign", {}).get("copy_eligible_visual_facts") or []
        ),
        "candidates": [
            {"index": index, "headline": copy["headline"], "subcopy": copy["body"]}
            for index, copy in enumerate(candidates, start=1)
        ],
    }, ensure_ascii=False, indent=2)
    raw, usage = _call_model(
        prompt=prompt,
        model_id=model_id,
        api_key=api_key,
        client=client,
        schema=_JUDGE_SCHEMA,
        instructions=_JUDGE_INSTRUCTIONS,
    )
    payload = json.loads(raw)
    evaluations = list(payload.get("evaluations") or [])
    passed = {
        int(item["index"])
        for item in evaluations
        if item.get("meaningful") is True
        and int(item.get("naturalness", 0)) >= 4
        and int(item.get("ad_value", 0)) >= 3
        and item.get("grounded") is True
        and item.get("menu_name_repeated") is False
        and item.get("descriptive_caption") is False
    }
    return passed, evaluations, usage


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    prepared = _copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider not in {"openai", "gemini"}:
        raise CopyGenerationError("배너 문구 생성은 OpenAI 또는 Gemini 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    feature = " ".join(str(prepared.get("product", {}).get("features") or "").split())
    hashtags = build_deterministic_hashtags(prepared)
    history: list[dict[str, Any]] = []
    valid: list[AdCopy] = []
    first_issues: tuple[str, ...] = ()

    for attempt in range(1, max_attempts + 1):
        raw = ""
        usage: dict[str, int] = {}
        try:
            prompt = build_banner_copy_prompt(prepared)
            if attempt > 1:
                prompt += "\n재작성: 이전 후보와 다른 표현을 쓰고, 메뉴명·사진 판독형 단어를 다시 점검한다."
            raw, usage = _call_model(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                schema=_GENERATION_SCHEMA,
                instructions=_SYSTEM_INSTRUCTIONS,
            )
            payload = json.loads(raw)
            generated = [
                _candidate_copy(str(item.get("headline") or ""), feature, hashtags)
                for item in payload.get("options", [])
            ]
            local_diagnostics: list[dict[str, Any]] = []
            locally_valid: list[AdCopy] = []
            for index, copy in enumerate(generated, start=1):
                issues = _local_candidate_issues(copy, prepared)
                local_diagnostics.append({
                    "index": index,
                    "headline": copy["headline"],
                    "issues": [issue.code for issue in issues],
                })
                if not issues:
                    locally_valid.append(copy)
            judge_passed: set[int] = set()
            judge_evaluations: list[dict[str, Any]] = []
            judge_usage: dict[str, int] = {}
            if len(generated) == INTERNAL_CANDIDATE_COUNT:
                judge_passed, judge_evaluations, judge_usage = _judge_candidates(
                    generated,
                    prepared,
                    model_id=model_id,
                    api_key=api_key,
                    client=client,
                )
            valid = [
                copy for index, copy in enumerate(generated, start=1)
                if index in judge_passed and not _local_candidate_issues(copy, prepared)
            ]
            history.append({
                "attempt": attempt,
                "raw_image_sent": False,
                "visual_caption_sent_to_copy_model": False,
                "filtered_visual_facts_sent": list(
                    prepared.get("campaign", {}).get("copy_eligible_visual_facts") or []
                ),
                "raw_output": raw,
                "usage": usage,
                "local_qc": local_diagnostics,
                "naturalness_judge": judge_evaluations,
                "judge_usage": judge_usage,
                "qualified_count": len(valid),
            })
            if len(valid) >= FINAL_OPTION_COUNT:
                selected = valid[:FINAL_OPTION_COUNT]
                issues = collect_banner_copy_issues(selected, prepared)
                if not issues:
                    return BannerCopyGenerationResult(
                        copies=selected,
                        generation_mode="headline_and_subcopy" if feature else "headline_only",
                        status="generated" if attempt == 1 else "retried",
                        attempts=attempt,
                        notice=(
                            "시각 캡션을 문구 소스에서 제외하고 메뉴명 중복·자연스러움 "
                            "QC를 통과한 후보만 선택했습니다."
                        ),
                        validation_issues=first_issues,
                        attempt_history=tuple(history),
                        prompt_version=_provider_copy_prompt_version(config.provider),
                    )
            if not first_issues:
                first_issues = tuple(
                    code
                    for item in local_diagnostics
                    for code in item["issues"]
                )
        except Exception as exc:
            history.append({
                "attempt": attempt,
                "raw_image_sent": False,
                "visual_caption_sent_to_copy_model": False,
                "filtered_visual_facts_sent": list(
                    prepared.get("campaign", {}).get("copy_eligible_visual_facts") or []
                ),
                "raw_output": raw,
                "usage": usage,
                "issues": [f"{type(exc).__name__}: {exc}"],
                "qualified_count": 0,
            })
            if not first_issues:
                first_issues = (f"{type(exc).__name__}: {exc}",)

    fallback: list[AdCopy] = []
    for headline in _fallback_hooks(prepared):
        copy = _candidate_copy(headline, feature, hashtags)
        if not _local_candidate_issues(copy, prepared):
            fallback.append(copy)
        if len(fallback) == FINAL_OPTION_COUNT:
            break
    if len(fallback) < FINAL_OPTION_COUNT:
        raise CopyGenerationError("자연스러움 QC를 통과하는 안전한 문구 두 개를 만들지 못했습니다.")
    final_issues = collect_banner_copy_issues(fallback, prepared)
    if final_issues:
        raise CopyGenerationError("fallback 문구가 최종 QC를 통과하지 못했습니다.")
    return BannerCopyGenerationResult(
        copies=fallback,
        generation_mode="headline_and_subcopy" if feature else "headline_only",
        status="fallback",
        attempts=max_attempts,
        notice=(
            "모델 후보가 메뉴명 중복·묘사형 표현·자연스러움 QC를 통과하지 "
            "못해 입력 톤과 제품 특징만 사용한 결정론적 fallback을 사용했습니다."
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=_provider_copy_prompt_version(config.provider),
    )


attach_visual_context = previous.attach_visual_context
extract_verified_visual_facts = previous.extract_verified_visual_facts

