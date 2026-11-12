"""단일 자유 입력 요청으로 제목 또는 제목+보조 문구를 생성하는 v1.8 흐름."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    _generate_with_gemini,
    build_deterministic_hashtags,
    split_distinct_feature_facts,
)
from src.features.ad_copy.validation import RISKY_CLAIM_TERMS, _fact_is_reflected
from src.features.ad_copy.internal.base_rules import (
    _CAMPAIGN_CONTEXT_NOUNS,
    _EMOJI_PATTERN,
    _PRODUCT_CLAIM_NOUNS,
    _PROMOTION_OR_PRICE_PATTERN,
    _UNREQUESTED_EMBELLISHMENTS,
    _UNVERIFIED_FACT_EXPRESSIONS,
    _UNVERIFIED_SENSORY_EXPRESSIONS,
    _context_aware_unsupported_labels,
    _fallback_title_copies,
    _normalize_compact,
    parse_request_constraints,
    parse_request_dimensions,
    title_lexical_similarity,
)


COPY_PROMPT_VERSION = "gemini_copy_v1.8.0-single-request-banner-copy"
GenerationMode = Literal["headline_only", "headline_and_subcopy"]
GenerationStatus = Literal["generated", "retried", "mixed", "fallback"]

DEFAULT_HEADLINE_LIMIT = 18
DEFAULT_SUBCOPY_LIMIT = 35
MIN_HEADLINE_LIMIT = 6
MAX_HEADLINE_LIMIT = 20
MIN_SUBCOPY_LIMIT = 10
MAX_SUBCOPY_LIMIT = 40


class CopyInputError(CopyGenerationError):
    """모델 호출 전에 필수 입력 또는 길이 조건을 보완해야 할 때 발생한다."""


@dataclass(frozen=True)
class CopyRequestLimits:
    headline: int = DEFAULT_HEADLINE_LIMIT
    subcopy: int = DEFAULT_SUBCOPY_LIMIT


@dataclass(frozen=True)
class CopyInputProfile:
    menu_name: str
    verified_facts_text: str
    verified_facts: tuple[str, ...]
    generation_mode: GenerationMode
    copy_request: str
    limits: CopyRequestLimits
    image_provided: bool
    input_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class CopyValidationIssue:
    code: str
    message: str
    option_index: int | None = None
    field: str | None = None


@dataclass(frozen=True)
class BannerCopyGenerationResult:
    copies: list[AdCopy]
    generation_mode: GenerationMode
    status: GenerationStatus
    attempts: int
    notice: str | None = None
    validation_issues: tuple[str, ...] = ()
    attempt_history: tuple[dict[str, Any], ...] = ()
    prompt_version: str = COPY_PROMPT_VERSION


def _normalized_mode(value: Any) -> GenerationMode:
    normalized = str(value or "headline_only").strip().casefold()
    aliases = {
        "headline_only": "headline_only",
        "title_only": "headline_only",
        "제목만": "headline_only",
        "headline_and_subcopy": "headline_and_subcopy",
        "headline_and_body": "headline_and_subcopy",
        "title_and_body": "headline_and_subcopy",
        "제목+본문": "headline_and_subcopy",
        "제목+보조 문구": "headline_and_subcopy",
        "제목+보조문구": "headline_and_subcopy",
    }
    if normalized not in aliases:
        raise CopyInputError("생성 방식은 '제목만' 또는 '제목+보조 문구'여야 합니다.")
    return aliases[normalized]  # type: ignore[return-value]


def parse_copy_request_limits(copy_request: str) -> CopyRequestLimits:
    """한 개의 자유 입력에서 제목·보조 문구 길이만 보수적으로 추출한다."""

    headline = DEFAULT_HEADLINE_LIMIT
    subcopy = DEFAULT_SUBCOPY_LIMIT
    headline_match = re.search(
        r"제목(?:은|을|:|：)?\s*(?:공백\s*포함\s*)?(\d{1,2})\s*자(?:\s*이내)?",
        copy_request,
    )
    subcopy_match = re.search(
        r"(?:본문|보조\s*문구)(?:은|을|:|：)?\s*"
        r"(?:공백\s*포함\s*)?(\d{1,2})\s*자(?:\s*이내)?",
        copy_request,
    )
    if headline_match:
        headline = int(headline_match.group(1))
    else:
        generic_match = re.search(r"(\d{1,2})\s*자\s*이내", copy_request)
        if generic_match:
            headline = int(generic_match.group(1))
    if subcopy_match:
        subcopy = int(subcopy_match.group(1))
    return CopyRequestLimits(headline=headline, subcopy=subcopy)


def analyze_copy_input(
    ad_input: dict[str, Any],
    *,
    image_provided: bool = False,
) -> CopyInputProfile:
    product = ad_input.get("product", {})
    campaign = ad_input.get("campaign", {})
    menu_name = str(product.get("name") or "").strip()
    facts_text = str(
        campaign.get("focus_feature") or product.get("features") or ""
    ).strip()
    copy_request = str(campaign.get("copy_request") or "").strip()
    mode = _normalized_mode(campaign.get("generation_mode") or "headline_only")
    limits = parse_copy_request_limits(copy_request)
    facts = tuple(split_distinct_feature_facts(facts_text))

    issues: list[str] = []
    if not menu_name:
        issues.append("메뉴명을 입력해 주세요.")
    if not facts_text or not facts:
        issues.append("확인된 상품 특징을 한 가지 이상 입력해 주세요.")
    if not MIN_HEADLINE_LIMIT <= limits.headline <= MAX_HEADLINE_LIMIT:
        issues.append(
            f"제목 길이는 {MIN_HEADLINE_LIMIT}~{MAX_HEADLINE_LIMIT}자로 요청해 주세요."
        )
    if (
        mode == "headline_and_subcopy"
        and not MIN_SUBCOPY_LIMIT <= limits.subcopy <= MAX_SUBCOPY_LIMIT
    ):
        issues.append(
            f"보조 문구 길이는 {MIN_SUBCOPY_LIMIT}~{MAX_SUBCOPY_LIMIT}자로 요청해 주세요."
        )

    return CopyInputProfile(
        menu_name=menu_name,
        verified_facts_text=facts_text,
        verified_facts=facts,
        generation_mode=mode,
        copy_request=copy_request,
        limits=limits,
        image_provided=image_provided,
        input_issues=tuple(issues),
    )


BANNER_COPY_SYSTEM_INSTRUCTIONS = """당신은 카페·음료·디저트·베이커리 소상공인을 위한 한국어 광고 배너 카피라이터입니다.

상품 이미지, 메뉴명, 확인된 상품 특징, 생성 방식과 한 칸의 문구 요청 사항을 바탕으로 배너 문구 선택지 2개를 작성하세요.

[입력 역할]

- product_image: 이미지에서 명확히 보이는 상품 종류·색상·형태·개수·구성·포장만 확인하는 보조 정보
- menu: 어떤 상품인지 식별하는 정보이며 배너에 별도로 표시될 수 있음
- verified_product_facts: 맛·향·식감·재료·구성·제조 방식 등 상품 사실을 표현할 수 있는 유일한 근거
- generation_mode: headline_only이면 제목만, headline_and_subcopy이면 제목과 보조 문구를 함께 생성
- copy_request.raw: 스타일·톤·강조 메시지·길이·메뉴명 사용 여부 등을 자유롭게 적은 하나의 요청 사항

copy_request.raw에 고객이나 이용 상황이 적혀 있으면 광고 콘셉트로 사용할 수 있지만, 입력되어 있지 않다고 새로 만들 필요는 없습니다. 스타일과 톤은 표현 방식이지 상품 사실의 근거가 아닙니다.

[제목과 보조 문구의 역할]

1. 제목은 관심을 끄는 한 가지 메시지만 짧고 분명하게 전달합니다.
2. 보조 문구는 제목을 뒷받침하도록 verified_product_facts의 확인된 특징 1~2개를 구체적으로 설명합니다.
3. 요청 사항이 제목과 보조 문구의 역할을 따로 지정하면 그대로 따릅니다.
4. 감성·휴식·일상·분위기 같은 광고 콘셉트는 주로 제목에, 맛·향·식감·재료 같은 상품 사실은 주로 보조 문구에 배치합니다.
5. 요청 사항에 광고 콘셉트가 없으면 제목에서도 가장 중요한 확인 사실을 간결하게 강조할 수 있습니다.
6. 제목과 보조 문구가 같은 문장을 반복하거나 단어 순서만 바꾼 관계가 되지 않게 합니다.

[사실 보호]

1. 입력에 없는 맛·향·식감·온도·재료·제조 방식·원산지·신선도·효능·인기·고객 반응을 만들지 마세요.
2. ‘깊은 풍미’, ‘풍부한 향’, ‘입안에 퍼지는 맛’, ‘본연의 맛’, ‘씹을수록’, ‘갓 만든’처럼 새로운 상품 상태나 경험을 암시하는 표현도 확인된 근거가 있을 때만 사용하세요.
3. 상품 이미지만 보고 맛·향·온도·식감을 추측하지 마세요.
4. 가격·할인·기간·행사·혜택·후기·해시태그를 넣지 마세요.
5. 의미 없는 영어·별점·인증 표시·이모지·최상급·명령형 CTA를 넣지 마세요.

[작성 규칙]

- 선택지는 정확히 2개 작성합니다.
- 두 제목은 서로 다른 시작 어절과 표현 관점을 사용합니다.
- 제목은 문장부호 없이 독립적으로 이해되는 완결된 구로 작성합니다.
- 전체 메뉴명은 사용자가 요청한 경우가 아니면 불필요하게 반복하지 않습니다.
- 사용자가 길이를 지정하지 않으면 제목은 공백 포함 18자 이내, 보조 문구는 공백 포함 35자 이내로 작성합니다.
- headline_only에서는 body·본문·보조 문구를 출력하지 않습니다.
- headline_and_subcopy에서는 각 제목에 대응하는 비어 있지 않은 subcopy를 작성합니다.

[출력 형식]

headline_only:
{"options":[{"headline":"제목 1"},{"headline":"제목 2"}]}

headline_and_subcopy:
{"options":[{"headline":"제목 1","subcopy":"보조 문구 1"},{"headline":"제목 2","subcopy":"보조 문구 2"}]}

JSON 객체 이외의 설명, 유형명, 검토 과정과 마크다운 코드 블록은 출력하지 마세요."""


def build_copy_response_schema(profile: CopyInputProfile) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "headline": {
            "type": "string",
            "minLength": 1,
            "maxLength": profile.limits.headline,
        }
    }
    required = ["headline"]
    if profile.generation_mode == "headline_and_subcopy":
        properties["subcopy"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": profile.limits.subcopy,
        }
        required.append("subcopy")
    return {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["options"],
        "additionalProperties": False,
    }


def _prompt_payload(profile: CopyInputProfile) -> dict[str, Any]:
    return {
        "product_image": {
            "provided": profile.image_provided,
            "role": "보이는 종류·색상·형태·개수·구성·포장 확인용 보조 정보",
        },
        "menu": {"name": profile.menu_name},
        "verified_product_facts": {
            "source_text": profile.verified_facts_text,
            "facts": list(profile.verified_facts),
        },
        "generation_mode": profile.generation_mode,
        "copy_request": {
            "raw": profile.copy_request or None,
            "role": "스타일·톤·강조 메시지·길이·메뉴명 사용 여부 등을 적는 하나의 자유 입력",
        },
        "output_limits": {
            "headline_max_length": profile.limits.headline,
            "subcopy_max_length": (
                profile.limits.subcopy
                if profile.generation_mode == "headline_and_subcopy"
                else None
            ),
            "option_count": 2,
        },
        "division_policy": {
            "headline": "관심을 끄는 한 가지 메시지",
            "subcopy": (
                "제목을 뒷받침하는 확인된 상품 특징"
                if profile.generation_mode == "headline_and_subcopy"
                else "생성하지 않음"
            ),
        },
    }


def build_banner_copy_prompt(
    ad_input: dict[str, Any],
    *,
    image_provided: bool = False,
    retry_context: dict[str, Any] | None = None,
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=image_provided)
    prompt = (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "다음 입력으로 광고 배너 문구 선택지 2개를 작성하세요.\n"
        + json.dumps(_prompt_payload(profile), ensure_ascii=False, indent=2)
    )
    if retry_context:
        prompt += (
            "\n\n이전 응답은 검증을 통과하지 못했습니다. 길이와 사실 근거를 "
            "다시 확인하여 선택지 2개 전체를 수정하세요.\n"
            + json.dumps(retry_context, ensure_ascii=False, indent=2)
        )
    return prompt


def parse_banner_copy_options(
    output_text: str,
    ad_input: dict[str, Any],
) -> list[AdCopy]:
    profile = analyze_copy_input(ad_input)
    try:
        payload = json.loads(output_text)
        raw_options = payload["options"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CopyGenerationError("AI 응답을 배너 문구 형식으로 읽을 수 없습니다.") from exc
    if not isinstance(raw_options, list) or len(raw_options) != 2:
        raise CopyGenerationError("광고 문구 선택지는 정확히 2개여야 합니다.")

    copies: list[AdCopy] = []
    hashtags = build_deterministic_hashtags(ad_input)
    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            raise CopyGenerationError("광고 문구 선택지 형식이 올바르지 않습니다.")
        raw_headline = raw_option.get("headline")
        if not isinstance(raw_headline, str) or not raw_headline.strip():
            raise CopyGenerationError("비어 있는 광고 제목이 있습니다.")
        headline = " ".join(raw_headline.split())
        if len(headline) > profile.limits.headline:
            raise CopyGenerationError(
                f"광고 제목은 공백 포함 {profile.limits.headline}자 이하여야 합니다."
            )

        raw_subcopy = raw_option.get("subcopy", "")
        if profile.generation_mode == "headline_and_subcopy":
            if not isinstance(raw_subcopy, str) or not raw_subcopy.strip():
                raise CopyGenerationError("제목+보조 문구 모드에는 보조 문구가 필요합니다.")
            body = " ".join(raw_subcopy.split())
            if len(body) > profile.limits.subcopy:
                raise CopyGenerationError(
                    f"보조 문구는 공백 포함 {profile.limits.subcopy}자 이하여야 합니다."
                )
        else:
            if "subcopy" in raw_option or "body" in raw_option:
                raise CopyGenerationError("제목만 모드에서는 보조 문구를 생성하지 않습니다.")
            body = ""
        copies.append({"headline": headline, "body": body, "hashtags": hashtags.copy()})
    return copies


def _fact_is_covered(text: str, fact: str) -> bool:
    return _fact_is_reflected(text, fact)


def _unsupported_product_expressions(
    text: str,
    profile: CopyInputProfile,
    *,
    creative_scope: str,
    allow_contextual_use: bool,
) -> list[str]:
    request_dimensions = parse_request_dimensions(profile.copy_request)
    product_terms = tuple(dict.fromkeys(_PRODUCT_CLAIM_NOUNS))
    context_terms = tuple(dict.fromkeys(
        _CAMPAIGN_CONTEXT_NOUNS
        + tuple(request_dimensions["target_audience"])
        + tuple(request_dimensions["usage_context"])
        + tuple(request_dimensions["emotional_message"])
    ))
    labels: list[str] = []
    for expressions, contextual in (
        (_UNVERIFIED_SENSORY_EXPRESSIONS, allow_contextual_use),
        (_UNVERIFIED_FACT_EXPRESSIONS, False),
        (_UNREQUESTED_EMBELLISHMENTS, allow_contextual_use),
    ):
        labels.extend(
            _context_aware_unsupported_labels(
                text,
                profile.verified_facts_text,
                expressions,
                creative_scope=creative_scope,
                product_terms=product_terms,
                context_terms=context_terms,
                allow_contextual_use=contextual,
            )
        )
    normalized_facts = _normalize_compact(profile.verified_facts_text)
    for term in RISKY_CLAIM_TERMS:
        if term in text and _normalize_compact(term) not in normalized_facts:
            labels.append(term)
    return list(dict.fromkeys(labels))


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    profile = analyze_copy_input(ad_input)
    request_dimensions = parse_request_dimensions(profile.copy_request)
    request_constraints = parse_request_constraints(profile.copy_request)
    creative_scope = str(request_dimensions["creative_scope"])
    normalized_menu = _normalize_compact(profile.menu_name)
    issues: list[CopyValidationIssue] = []

    if len(copies) != 2:
        issues.append(
            CopyValidationIssue("OPTION_COUNT", "광고 문구 선택지는 정확히 2개여야 합니다.")
        )
        return issues

    headline_signatures: list[str] = []
    for index, copy in enumerate(copies, start=1):
        headline = " ".join(copy["headline"].split())
        body = " ".join(copy["body"].split())
        headline_signatures.append(_normalize_compact(headline))

        if len(headline) > profile.limits.headline:
            issues.append(CopyValidationIssue(
                "HEADLINE_TOO_LONG",
                f"선택지 {index} 제목이 {len(headline)}자로, {profile.limits.headline}자를 초과했습니다.",
                index,
                "headline",
            ))
        if len(headline) < 4:
            issues.append(CopyValidationIssue(
                "HEADLINE_TOO_SHORT",
                f"선택지 {index} 제목이 너무 짧아 독립적으로 이해하기 어렵습니다.",
                index,
                "headline",
            ))
        if (
            not request_constraints.allow_menu_name
            and normalized_menu
            and normalized_menu in _normalize_compact(headline)
        ):
            issues.append(CopyValidationIssue(
                "FULL_MENU_NAME_REPEATED",
                f"선택지 {index} 제목에 별도 표시되는 전체 메뉴명이 반복되었습니다.",
                index,
                "headline",
            ))
        if _PROMOTION_OR_PRICE_PATTERN.search(headline) or _EMOJI_PATTERN.search(headline):
            issues.append(CopyValidationIssue(
                "DISALLOWED_HEADLINE_CONTENT",
                f"선택지 {index} 제목에 가격·행사 또는 이모지 표현이 있습니다.",
                index,
                "headline",
            ))

        reflected_headline_facts = [
            fact for fact in profile.verified_facts
            if _fact_is_covered(headline, fact)
        ]
        if creative_scope == "fact_focused" and not reflected_headline_facts:
            issues.append(CopyValidationIssue(
                "HEADLINE_NOT_GROUNDED",
                f"선택지 {index} 제목에 확인된 상품 특징이 반영되지 않았습니다.",
                index,
                "headline",
            ))
        unsupported_headline = _unsupported_product_expressions(
            headline,
            profile,
            creative_scope=creative_scope,
            allow_contextual_use=True,
        )
        if unsupported_headline:
            issues.append(CopyValidationIssue(
                "UNVERIFIED_HEADLINE_EXPRESSION",
                f"선택지 {index} 제목에 확인되지 않은 상품 표현이 있습니다: {', '.join(unsupported_headline)}",
                index,
                "headline",
            ))

        if profile.generation_mode == "headline_only":
            if body:
                issues.append(CopyValidationIssue(
                    "UNEXPECTED_SUBCOPY",
                    f"선택지 {index}는 제목만 모드인데 보조 문구가 있습니다.",
                    index,
                    "subcopy",
                ))
        else:
            if not body:
                issues.append(CopyValidationIssue(
                    "SUBCOPY_MISSING",
                    f"선택지 {index}에 보조 문구가 없습니다.",
                    index,
                    "subcopy",
                ))
            elif len(body) > profile.limits.subcopy:
                issues.append(CopyValidationIssue(
                    "SUBCOPY_TOO_LONG",
                    f"선택지 {index} 보조 문구가 {len(body)}자로, {profile.limits.subcopy}자를 초과했습니다.",
                    index,
                    "subcopy",
                ))
            missing_facts = [
                fact for fact in profile.verified_facts
                if not _fact_is_covered(body, fact)
            ]
            if missing_facts:
                issues.append(CopyValidationIssue(
                    "SUBCOPY_MISSING_FACTS",
                    f"선택지 {index} 보조 문구에 확인된 특징이 빠졌습니다: {', '.join(missing_facts)}",
                    index,
                    "subcopy",
                ))
            unsupported_body = _unsupported_product_expressions(
                body,
                profile,
                creative_scope="fact_focused",
                allow_contextual_use=False,
            )
            if unsupported_body:
                issues.append(CopyValidationIssue(
                    "UNVERIFIED_SUBCOPY_EXPRESSION",
                    f"선택지 {index} 보조 문구에 확인되지 않은 상품 표현이 있습니다: {', '.join(unsupported_body)}",
                    index,
                    "subcopy",
                ))
            if _PROMOTION_OR_PRICE_PATTERN.search(body) or _EMOJI_PATTERN.search(body):
                issues.append(CopyValidationIssue(
                    "DISALLOWED_SUBCOPY_CONTENT",
                    f"선택지 {index} 보조 문구에 가격·행사 또는 이모지 표현이 있습니다.",
                    index,
                    "subcopy",
                ))
            if re.search(r"(?:만나|즐겨|느껴|확인|선택|놓치지\s*말아).*세요", body):
                issues.append(CopyValidationIssue(
                    "SUBCOPY_CTA",
                    f"선택지 {index} 보조 문구가 명령형 행동 유도 문구입니다.",
                    index,
                    "subcopy",
                ))
            if (
                _normalize_compact(headline) == _normalize_compact(body)
                or title_lexical_similarity(headline, body) >= 0.85
            ):
                issues.append(CopyValidationIssue(
                    "HEADLINE_SUBCOPY_REPETITION",
                    f"선택지 {index}의 제목과 보조 문구가 같은 내용을 반복합니다.",
                    index,
                ))

        for phrase in request_constraints.required_phrases:
            pair_text = headline if profile.generation_mode == "headline_only" else f"{headline} {body}"
            if _normalize_compact(phrase) not in _normalize_compact(pair_text):
                issues.append(CopyValidationIssue(
                    "MISSING_REQUIRED_PHRASE",
                    f"선택지 {index}에 필수 표현 '{phrase}'이 없습니다.",
                    index,
                ))
        for phrase in request_constraints.forbidden_phrases:
            if _normalize_compact(phrase) in _normalize_compact(f"{headline} {body}"):
                issues.append(CopyValidationIssue(
                    "FORBIDDEN_PHRASE",
                    f"선택지 {index}에 제외 요청한 표현 '{phrase}'이 있습니다.",
                    index,
                ))

    if len(set(headline_signatures)) != len(headline_signatures):
        issues.append(CopyValidationIssue(
            "DUPLICATE_HEADLINE",
            "두 선택지의 제목이 동일합니다.",
            2,
            "headline",
        ))

    deduplicated: list[CopyValidationIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for issue in issues:
        key = (issue.code, issue.message, issue.option_index, issue.field)
        if key not in seen:
            seen.add(key)
            deduplicated.append(issue)
    return deduplicated


def _safe_fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    profile = analyze_copy_input(ad_input)
    title_copies = _fallback_title_copies(ad_input)
    titles = [copy["headline"] for copy in title_copies]
    while len(titles) < 2:
        titles.append(profile.verified_facts[0])
    body = " · ".join(profile.verified_facts)
    if len(body) > profile.limits.subcopy:
        body = profile.verified_facts[0][: profile.limits.subcopy].rstrip()
    hashtags = build_deterministic_hashtags(ad_input)
    return [
        {
            "headline": title[: profile.limits.headline].rstrip(),
            "body": body if profile.generation_mode == "headline_and_subcopy" else "",
            "hashtags": hashtags.copy(),
        }
        for title in titles[:2]
    ]


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

    first_issues: tuple[str, ...] = ()
    retry_context: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        prompt = build_banner_copy_prompt(
            ad_input,
            image_provided=image_bytes is not None,
            retry_context=retry_context,
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
                response_schema=build_copy_response_schema(profile),
            )
            copies = parse_banner_copy_options(output_text, ad_input)
            issues = collect_banner_copy_issues(copies, ad_input)
            history.append({
                "attempt": attempt,
                "raw_output": output_text,
                "options": [
                    {"headline": copy["headline"], "subcopy": copy["body"]}
                    for copy in copies
                ],
                "issues": [issue.message for issue in issues],
            })
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            history.append({
                "attempt": attempt,
                "raw_output": output_text,
                "options": [],
                "issues": [error_message],
            })
            if not first_issues:
                first_issues = (error_message,)
            retry_context = {
                "previous_error": error_message,
                "headline_max_length": profile.limits.headline,
                "subcopy_max_length": (
                    profile.limits.subcopy
                    if profile.generation_mode == "headline_and_subcopy"
                    else None
                ),
            }
            continue

        if not issues:
            return BannerCopyGenerationResult(
                copies=copies,
                generation_mode=profile.generation_mode,
                status="generated" if attempt == 1 else "retried",
                attempts=attempt,
                notice=None if attempt == 1 else "검증 문제를 반영해 문구 전체를 다시 생성했습니다.",
                validation_issues=first_issues,
                attempt_history=tuple(history),
            )
        issue_messages = tuple(issue.message for issue in issues)
        if not first_issues:
            first_issues = issue_messages
        retry_context = {
            "previous_options": history[-1]["options"],
            "violations": [
                {
                    "option_index": issue.option_index,
                    "field": issue.field,
                    "message": issue.message,
                }
                for issue in issues
            ],
            "headline_max_length": profile.limits.headline,
            "subcopy_max_length": (
                profile.limits.subcopy
                if profile.generation_mode == "headline_and_subcopy"
                else None
            ),
        }

    return BannerCopyGenerationResult(
        copies=_safe_fallback_options(ad_input),
        generation_mode=profile.generation_mode,
        status="fallback",
        attempts=max_attempts,
        notice="모델 문구가 검증을 통과하지 못해 확인된 입력 기반 문구를 사용했습니다.",
        validation_issues=first_issues,
        attempt_history=tuple(history),
    )

