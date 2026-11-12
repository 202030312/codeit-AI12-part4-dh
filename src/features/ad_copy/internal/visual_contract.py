"""v1.10의 생성 흐름을 유지하며 사실·문장 검증을 강화한 v1.10.1."""

from __future__ import annotations

import json
import re
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    _generate_with_gemini,
)
from src.features.ad_copy.internal.base_rules import (
    _CAMPAIGN_CONTEXT_NOUNS,
    _PRODUCT_CLAIM_NOUNS,
    _context_aware_unsupported_labels,
    _normalize_compact,
    parse_request_dimensions,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.base_workflow as base
from src.features.ad_copy.visual_analysis import (
    VISUAL_FACT_RESPONSE_SCHEMA,
    VISUAL_FACT_SYSTEM_INSTRUCTIONS,
    VISUAL_PROMPT_VERSION,
    VisualFactExtractionResult,
    attach_visual_context,
    extract_verified_visual_facts,
    parse_visual_fact_response,
)
import src.features.ad_copy.visual_analysis as visual_base


COPY_PROMPT_VERSION = "gemini_copy_v1.10.1-grounded-claim-guard"


# 활용형까지 잡되, 아래 검증 함수에서 상품 사실과 광고 콘셉트 문맥을 구분한다.
_CONTEXTUAL_MODIFIER_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "깊은": ("깊",),
    "풍부한": ("풍부",),
    "가득": ("가득",),
    "극도로": ("극도",),
    "완벽한": ("완벽",),
    "강렬한": ("강렬",),
}


# 문장 자체가 아니라 고객 반응·반복 행동·인기도를 주장하는 의미 유형을 찾는다.
# 정규식은 공백과 문장부호를 제거한 문자열에 적용한다.
_CUSTOMER_CLAIM_PATTERNS: tuple[
    tuple[str, re.Pattern[str], tuple[str, ...], bool], ...
] = (
    (
        "질리지 않는다는 고객 반응",
        re.compile(r"(?:질리.{0,5}(?:않|없)|안질리)"),
        ("질리",),
        False,
    ),
    (
        "자꾸 손이 간다는 고객 반응",
        re.compile(r"(?:자꾸|계속|절로|또)?손(?:이)?가(?:는|게|요|다|고|서|는다는)?"),
        ("손이가", "손이간"),
        False,
    ),
    (
        "계속 찾게 된다는 반복 행동",
        re.compile(r"(?:(?:매일|계속|자꾸|다시|또)?찾게되|(?:매일|계속|자꾸|다시|또)찾)"),
        ("찾게되", "매일찾", "계속찾", "자꾸찾"),
        False,
    ),
    (
        "반복 구매·주문 행동",
        re.compile(r"(?:재구매|(?:매일|계속|자꾸|다시|또)주문|주문하게되)"),
        ("재구매", "반복주문", "계속주문"),
        False,
    ),
    (
        "먹거나 마시고 싶다는 욕구",
        re.compile(r"(?:(?:매일|계속|자꾸|또)?(?:먹|마시)고싶)"),
        ("먹고싶", "마시고싶"),
        False,
    ),
    (
        "계속 생각난다는 고객 반응",
        re.compile(r"(?:(?:매일|계속|자꾸|또)생각나|잊을수없)"),
        ("생각나", "잊을수없"),
        False,
    ),
    (
        "사랑받는다는 평판",
        re.compile(r"사랑받"),
        ("사랑받",),
        False,
    ),
    (
        "인기·베스트셀러 주장",
        re.compile(r"(?:인기|베스트셀러|베스트메뉴|인기메뉴|입소문|품절대란|줄서서먹)"),
        ("인기", "베스트셀러", "베스트메뉴", "입소문", "품절", "줄서서먹"),
        False,
    ),
    (
        "누구나 좋아한다는 절대적 선호",
        re.compile(r"(?:(?:누구나|모두가|모두|남녀노소).{0,8}좋아|호불호없)"),
        (),
        True,
    ),
    (
        "중독성·멈출 수 없다는 과장",
        re.compile(r"(?:중독|멈출수없|끊을수없)"),
        (),
        True,
    ),
    (
        "후회·실패가 없다는 만족도 단정",
        re.compile(r"(?:(?:절대)?후회없|실패없(?:는선택)?|만족보장)"),
        (),
        True,
    ),
)


_DANGLING_HEADLINE_PATTERN = re.compile(
    r"(?:같은|담은|채우는|전하는|느껴지는|어울리는|위한|"
    r"로운|스러운|적인|특징인|부드러운|따뜻한|달콤한|바삭한|"
    r"촉촉한|쫄깃한|쌉싸름한|담백한|향긋한|고소한|진한|깊은)$"
)


BANNER_COPY_SYSTEM_INSTRUCTIONS = base.BANNER_COPY_SYSTEM_INSTRUCTIONS + """

[v1.10.1 검증된 시각 정보와 제목 근거 규칙]

1. 원본 이미지 자체를 새로운 문구 사실의 근거로 다시 해석하지 마세요. 문구에서 사용할 수 있는 시각 정보는 verified_visual_facts에 전달된 고신뢰 관찰뿐입니다.
2. verified_visual_facts는 상품 종류·색상·형태·개수·용기·포장·보이는 구성·배치를 묘사할 때만 선택적으로 사용할 수 있습니다.
3. 얼음이 보인다는 사실을 ‘시원한 맛·청량한 풍미’로, 갈색 표면을 ‘바삭한 식감’으로 바꾸지 마세요. 맛·향·풍미·식감은 verified_product_facts만 근거로 사용하세요.
4. copy_request에 타깃·이용 상황·강조할 감성 메시지가 있으면 제목에서 추상적인 광고 콘셉트로 재구성할 수 있습니다.
5. copy_request가 비어 있거나 스타일·톤만 있으면 제목도 verified_product_facts 또는 verified_visual_facts에 근거해야 합니다. 스타일·톤만 보고 여유·휴식·행복·시간·순간 같은 새 메시지를 만들지 마세요.
6. ‘깊고·풍부하게·가득한·극도로·완벽한·강렬한’과 같은 표현은 일괄 금지어가 아닙니다. 맛·향·풍미·식감 등 상품 특징을 꾸밀 때는 verified_product_facts에 같은 의미가 있어야 하고, 하루·시간·여유·휴식·기분 같은 광고 상황을 꾸밀 때는 copy_request에 해당 콘셉트가 있어야 합니다.
7. 입력에 근거 없는 고객 반응·재구매 행동·대중적 선호·인기도·중독성·만족도를 주장하지 마세요. 예: ‘질리지 않는’, ‘자꾸 손이 가는’, ‘계속 찾게 되는’, ‘매일 먹고 싶은’, ‘사랑받는’, ‘인기 메뉴’, ‘누구나 좋아하는’, ‘중독되는’, ‘후회 없는’. 이 목록의 문장만 피하는 것이 아니라 같은 의미의 변형도 만들지 마세요.
8. copy_request는 고객 반응이나 인기도의 사실 근거가 아닙니다. verified_product_facts에 인기도·고객 반응이 명시된 경우에만 비절대적 표현을 사용할 수 있고, ‘누구나·호불호 없는·중독·절대 후회 없는’ 같은 절대적·과장 표현은 사용하지 마세요.
9. 제목은 독립적으로 이해되는 완결된 구여야 합니다. ‘같은·담은·채우는·전하는·느껴지는·어울리는·위한·로운·스러운·적인’처럼 뒤에 꾸밀 말이 필요한 형태로 끝내지 마세요.
10. 보조 문구에는 verified_product_facts의 표현을 우선 사용하고 입력에 없는 수식어를 덧붙이지 마세요. 친근한 보조 문구는 설명형 해요체를 사용할 수 있지만 명령형 CTA는 사용하지 마세요.
"""


def _visual_facts(ad_input: dict[str, Any]) -> list[dict[str, str]]:
    return visual_base._visual_facts(ad_input)


def _visual_fact_reflected(text: str, ad_input: dict[str, Any]) -> bool:
    return visual_base._visual_fact_reflected(text, ad_input)


def _unverified_visual_terms(text: str, ad_input: dict[str, Any]) -> list[str]:
    return visual_base._unverified_visual_terms(text, ad_input)


def _prompt_payload(profile: Any, ad_input: dict[str, Any]) -> dict[str, Any]:
    payload = visual_base._prompt_payload(profile, ad_input)
    dimensions = parse_request_dimensions(profile.copy_request)
    has_campaign_concept = bool(dimensions["has_campaign_concept"])
    payload["copy_request"]["parsed_scope"] = dimensions["creative_scope"]
    payload["copy_request"]["campaign_concepts"] = {
        "target_audience": dimensions["target_audience"],
        "usage_context": dimensions["usage_context"],
        "emotional_message": dimensions["emotional_message"],
    }
    payload["auto_composition"]["headline"] = (
        "요청에 명시된 광고 콘셉트로 관심 메시지 구성. 상품 사실을 반드시 반복할 필요 없음"
        if has_campaign_concept
        else "verified_product_facts 또는 verified_visual_facts에 근거한 관심 메시지"
    )
    payload["claim_policy"] = {
        "style_tone_are_not_message_sources": True,
        "customer_reaction_popularity_requires_verified_fact": True,
        "absolute_preference_and_addiction_claims": "사용 금지",
        "examples_are_not_exhaustive": True,
    }
    return payload


def build_banner_copy_prompt(
    ad_input: dict[str, Any],
    *,
    image_provided: bool = False,
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=image_provided)
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "검증된 상품 사실과 고신뢰 시각 사실을 분리해 광고 문구 선택지 2개를 작성하세요. "
        "사용자가 역할을 지정하지 않아도 자동 구성하되, 제목의 메시지 출처와 고객 반응 주장 규칙을 지키세요.\n"
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
            "입력에 없는 상품 강도 수식어와 고객 반응·인기도 주장 제거",
            "요청에 광고 콘셉트가 없으면 제목을 상품 사실 또는 시각 사실에 연결",
            "제목을 독립적으로 이해되는 완결된 명사구나 문장으로 마무리",
            "보조 문구에는 verified_product_facts의 핵심어를 모두 유지",
            "locked_options와 다른 제목 작성",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "실패한 선택지만 교체하세요. locked_options는 다시 출력하거나 수정하지 말고, "
        "replace_option_indices 순서에 맞춘 교체 선택지만 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _modifier_labels(text: str, profile: Any) -> list[str]:
    """상품 속성 수식은 사실에서, 광고 상황 수식은 요청 콘셉트에서 허용한다."""

    dimensions = parse_request_dimensions(profile.copy_request)
    context_terms = tuple(dict.fromkeys(
        _CAMPAIGN_CONTEXT_NOUNS
        + tuple(dimensions["target_audience"])
        + tuple(dimensions["usage_context"])
        + tuple(dimensions["emotional_message"])
    ))
    return _context_aware_unsupported_labels(
        text,
        profile.verified_facts_text,
        _CONTEXTUAL_MODIFIER_EXPRESSIONS,
        creative_scope=str(dimensions["creative_scope"]),
        product_terms=tuple(_PRODUCT_CLAIM_NOUNS),
        context_terms=context_terms,
        allow_contextual_use=True,
    )


def _unsupported_headline_expressions(text: str, profile: Any) -> list[str]:
    dimensions = parse_request_dimensions(profile.copy_request)
    labels = base._unsupported_product_expressions(
        text,
        profile,
        creative_scope=str(dimensions["creative_scope"]),
        allow_contextual_use=True,
    )
    contextual_labels = _modifier_labels(text, profile)

    # v1.9에서는 '가득'을 문맥과 무관하게 차단했다. v1.10.1에서는 광고 상황을
    # 꾸미거나 상품 사실에 명시된 경우에만 기존 차단을 해제한다.
    if "가득" in labels and "가득" not in contextual_labels:
        labels = [label for label in labels if label != "가득"]
    labels.extend(contextual_labels)
    return list(dict.fromkeys(labels))


def _customer_claim_labels(text: str, verified_facts_text: str) -> list[str]:
    normalized_text = _normalize_compact(text)
    normalized_facts = _normalize_compact(verified_facts_text)
    labels: list[str] = []
    for label, pattern, support_roots, always_block in _CUSTOMER_CLAIM_PATTERNS:
        if not pattern.search(normalized_text):
            continue
        supported = any(
            _normalize_compact(root) in normalized_facts
            for root in support_roots
        )
        if always_block or not supported:
            labels.append(label)
    return labels


def _headline_is_dangling(headline: str) -> bool:
    cleaned = headline.strip().rstrip(".!?。！？,，·:;\"'”’)").strip()
    return bool(_DANGLING_HEADLINE_PATTERN.search(cleaned))


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
            and issue.option_index is not None
            and _visual_fact_reflected(
                copies[issue.option_index - 1]["headline"], ad_input
            )
        ):
            continue
        if issue.code == "UNVERIFIED_HEADLINE_EXPRESSION" and issue.option_index:
            headline = copies[issue.option_index - 1]["headline"]
            unsupported = _unsupported_headline_expressions(headline, profile)
            if unsupported:
                issues.append(CopyValidationIssue(
                    "UNVERIFIED_HEADLINE_EXPRESSION",
                    f"선택지 {issue.option_index} 제목에 확인되지 않은 상품 표현이 있습니다: {', '.join(unsupported)}",
                    issue.option_index,
                    "headline",
                ))
            continue
        issues.append(issue)

    for index, copy in enumerate(copies, start=1):
        headline = copy["headline"]
        body = copy["body"]

        # 기존 검증이 활용형을 놓치는 경우를 보완한다.
        modifier_labels = _modifier_labels(headline, profile)
        if modifier_labels and not any(
            issue.code == "UNVERIFIED_HEADLINE_EXPRESSION"
            and issue.option_index == index
            for issue in issues
        ):
            issues.append(CopyValidationIssue(
                "UNVERIFIED_HEADLINE_EXPRESSION",
                f"선택지 {index} 제목에 확인되지 않은 상품 표현이 있습니다: {', '.join(modifier_labels)}",
                index,
                "headline",
            ))

        if _headline_is_dangling(headline):
            issues.append(CopyValidationIssue(
                "DANGLING_HEADLINE",
                f"선택지 {index} 제목이 뒤에 꾸밀 말이 필요한 형태로 끝납니다.",
                index,
                "headline",
            ))

        for field, text in (("headline", headline), ("subcopy", body)):
            if not text:
                continue
            claims = _customer_claim_labels(text, profile.verified_facts_text)
            if claims:
                issues.append(CopyValidationIssue(
                    "UNVERIFIED_CUSTOMER_CLAIM",
                    f"선택지 {index} {field}에 근거 없는 고객 반응·행동·인기도 주장이 있습니다: {', '.join(claims)}",
                    index,
                    field,
                ))
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
