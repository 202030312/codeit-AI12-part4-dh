"""간단한 단일 요청을 자동 배치하는 제목/보조 문구 v1.9 흐름."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    _generate_with_gemini,
    build_deterministic_hashtags,
)
from src.features.ad_copy.validation import RISKY_CLAIM_TERMS
from src.features.ad_copy.internal.base_rules import (
    _CAMPAIGN_CONTEXT_NOUNS,
    _EMOJI_PATTERN,
    _PRODUCT_CLAIM_NOUNS,
    _PROMOTION_OR_PRICE_PATTERN,
    _UNREQUESTED_EMBELLISHMENTS,
    _UNVERIFIED_FACT_EXPRESSIONS,
    _UNVERIFIED_SENSORY_EXPRESSIONS,
    _context_aware_unsupported_labels,
    _normalize_compact,
    _term_is_reflected,
    _terms,
    parse_request_constraints,
    parse_request_dimensions,
    title_lexical_similarity,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyInputProfile,
    CopyValidationIssue,
    analyze_copy_input,
)


COPY_PROMPT_VERSION = "gemini_copy_v1.9.0-auto-compose-banner-copy"

_GENERIC_FACT_TERMS = {
    "맛",
    "향",
    "풍미",
    "식감",
    "결",
    "겉",
    "속",
    "윗면",
    "특징",
    "특징인",
    "메뉴",
    "상품",
    "음료",
    "디저트",
}
_RELATION_ROOTS = {"관계", "조합", "조화", "만남", "어우러"}
_PRODUCT_ATTRIBUTE_NOUNS = {
    "맛",
    "향",
    "풍미",
    "식감",
    "결",
    "조합",
    "관계",
    "원두",
    "크림",
    "치즈",
    "캐러멜",
}
_MODIFIER_ENDING_PATTERN = re.compile(
    r"(?:한|하는|되는|있는|없는|로운|스러운|적인|은|운|진)$"
)
_UNVERIFIED_ARRANGEMENT_ROOTS = {
    "얹": "얹은",
    "올린": "올린",
    "쌓": "쌓은",
    "섞": "섞은",
    "채운": "채운",
    "바른": "바른",
    "뿌린": "뿌린",
}
_DANGLING_SUBCOPY_PATTERN = re.compile(
    r"[0-9A-Za-z가-힣]+(?:한|하는|되는|로운|스러운|적인|부드러운|따뜻한|달콤한|바삭한|촉촉한|진한)$"
)


@dataclass(frozen=True)
class ParsedModelOptions:
    copies: list[AdCopy]
    expected_count: int


BANNER_COPY_SYSTEM_INSTRUCTIONS = """당신은 카페·음료·디저트·베이커리 소상공인을 위한 한국어 광고 배너 카피라이터입니다.

사용자는 상품 이미지, 메뉴명, 확인된 상품 특징, 생성 방식과 선택적인 문구 요청 사항 한 칸만 제공합니다. 사용자가 제목과 보조 문구의 역할을 일일이 지정하지 않아도 아래 기본 정책에 따라 자동으로 구성하세요.

[입력 역할]

- product_image: 명확히 보이는 종류·색상·형태·개수·구성·포장 확인용 보조 정보
- menu: 상품 식별 정보. 기본적으로 배너에 따로 표시되므로 전체 메뉴명을 문구에서 반복하지 않음
- verified_product_facts: 맛·향·식감·재료·구성·제조 방식 등 상품 사실의 유일한 근거
- generation_mode: headline_only 또는 headline_and_subcopy
- copy_request.raw: 스타일·톤·강조 메시지·길이·메뉴명 사용 여부 등을 자유롭게 적는 선택 입력. 비어 있어도 됨

copy_request는 표현 방법을 정하는 요청이지 새로운 상품 사실의 근거가 아닙니다. 요청이 없으면 확인된 상품 특징에 맞는 표현을 자동 선택하세요. 고객·이용 상황은 사용자가 적었을 때만 광고 콘셉트로 사용할 수 있습니다.

[headline_only 자동 구성]

1. 후보 1은 사용자가 강조한 확인 사실 또는 가장 중요한 확인 사실을 직접적이고 간결하게 전달합니다.
2. 후보 2는 요청에 감성 메시지가 있으면 그 메시지를 사용하고, 없으면 같은 사실을 다른 시작 어절과 관점으로 표현합니다.
3. 여러 사실을 모두 강조하라고 요청하면 부가적인 감성 표현을 빼고 사실 핵심어를 우선해 길이 안에 압축합니다.
4. 확인 사실과 요청 메시지가 짧을 때 표현을 풍성하게 보이게 하려고 새로운 감각·강도·상황을 만들지 마세요.

[headline_and_subcopy 자동 구성]

1. 제목은 관심을 끄는 한 가지 메시지만 전달합니다.
2. 요청에 강조 메시지가 있으면 이를 제목에 우선 반영합니다.
3. 강조 메시지가 없으면 가장 중요한 확인 사실을 제목의 중심으로 삼습니다.
4. 보조 문구는 verified_product_facts의 확인 사실을 빠짐없이 사용해 제목을 뒷받침합니다.
5. 제목에 모든 사실을 억지로 넣지 말고, 구체적인 맛·향·식감·재료 설명은 보조 문구가 담당하게 합니다.
6. 사용자가 제목과 보조 문구의 역할을 직접 지정한 경우에만 그 지정 내용을 우선합니다.

[보조 문구 표현 형식]

- 고급스럽고 미니멀한 요청: 짧은 명사구를 우선합니다. 예: ‘진한 치즈 풍미와 부드러운 식감’
- 트렌디하고 강렬한 요청: 리듬감 있는 짧은 구를 사용할 수 있습니다. 예: ‘바삭한 겉, 촉촉한 속’
- 친근한 요청: 짧은 해요체 문장을 사용할 수 있습니다.
- 신뢰감 있고 설명적인 요청: 짧은 설명 문장을 사용할 수 있습니다.
- 요청이 없으면 배너에 잘 맞는 명사구 또는 완결된 짧은 구를 우선합니다.
- 모든 보조 문구를 ‘~다.’나 ‘~요.’로 끝낼 필요는 없습니다.
- ‘진하고 부드러운’처럼 뒤에 꾸밀 말이 필요한 관형형으로 끝내지 마세요.
- 완전한 문장일 때만 마침표를 사용하세요.

[사실 보호]

1. 입력에 없는 맛·향·식감·온도·재료·배치·제조 방식·원산지·신선도·효능·인기·고객 반응을 만들지 마세요.
2. 맛·향·풍미·식감·크림·치즈·원두·빵 결처럼 상품 특징을 나타내는 말에 수식어를 붙이려면 그 의미가 verified_product_facts에 직접 포함돼야 합니다.
3. 스타일이 고급스럽다고 ‘고급스러운 풍미’로, 톤이 따뜻하다고 ‘따뜻한 음료’로, 새로운 분위기라고 ‘이국적인 향’으로 바꾸지 마세요.
4. ‘A와 B의 조합’을 ‘A를 B에 얹은·섞은·채운’처럼 구체적인 배치나 제조 방식으로 바꾸지 마세요. 입력에서 확인된 관계만 유지하세요.
5. 상황·감정을 꾸미는 표현은 요청된 광고 콘셉트와 연결될 때만 허용합니다. 예: 고요한 아침, 따뜻한 순간
6. 가격·할인·행사·혜택·후기·해시태그·최상급·명령형 CTA·이모지를 넣지 마세요.

[길이와 최종 점검]

1. 제목과 보조 문구의 공백을 포함한 실제 글자 수를 출력 직전에 직접 세세요.
2. hard_max_length는 절대 넘지 마세요. 가능하면 recommended_target_length 안으로 작성하세요.
3. 길이를 줄일 때는 부가 수식어와 상황 표현을 먼저 제거하고 확인 사실의 핵심어는 유지하세요.
4. 글자 수를 맞추기 위해 단어 중간을 자르거나 불완전한 구로 끝내지 마세요.
5. 제목과 보조 문구가 같은 문장을 반복하지 않게 하세요.

[선택지 수]

- 최초 생성에서는 선택지 2개를 반환합니다.
- repair 요청에서는 replace_option_indices의 개수만큼 교체 선택지를 반환합니다.
- locked_options는 수정하거나 다시 출력하지 않습니다.

[출력 형식]

headline_only:
{"options":[{"headline":"제목 1"},{"headline":"제목 2"}]}

headline_and_subcopy:
{"options":[{"headline":"제목 1","subcopy":"보조 문구 1"},{"headline":"제목 2","subcopy":"보조 문구 2"}]}

JSON 객체 이외의 설명, 유형명, 검토 과정과 마크다운 코드 블록은 출력하지 마세요."""


def _response_schema(profile: CopyInputProfile, option_count: int) -> dict[str, Any]:
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
                "minItems": option_count,
                "maxItems": option_count,
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


def _ending_policy(profile: CopyInputProfile) -> str:
    dimensions = parse_request_dimensions(profile.copy_request)
    styles = set(dimensions["style"])
    tones = set(dimensions["tone"])
    if {"고급스러운", "간결한"} & styles:
        return "명사구 우선"
    if "친근한" in tones:
        return "명사구 또는 짧은 해요체"
    if {"신뢰감 있는", "전문적인"} & tones:
        return "명사구 또는 짧은 설명 문장"
    if {"트렌디한", "강렬한"} & styles:
        return "리듬감 있는 짧은 구"
    return "배너용 명사구 또는 완결된 짧은 구"


def _prompt_payload(profile: CopyInputProfile) -> dict[str, Any]:
    constraints = parse_request_constraints(profile.copy_request)
    return {
        "product_image": {
            "provided": profile.image_provided,
            "role": "보이는 종류·색상·형태·개수·구성·포장 확인용 보조 정보",
        },
        "menu": {
            "name": profile.menu_name,
            "full_name_policy": (
                "요청되어 후보 중 하나에 사용"
                if constraints.allow_menu_name
                else "전체 메뉴명 사용 금지"
            ),
        },
        "verified_product_facts": {
            "source_text": profile.verified_facts_text,
            "facts": list(profile.verified_facts),
            "role": "상품 사실의 유일한 근거",
        },
        "generation_mode": profile.generation_mode,
        "copy_request": {
            "raw": profile.copy_request or None,
            "optional": True,
            "role": "스타일·톤·강조 메시지·길이 등을 적는 하나의 자유 입력",
        },
        "auto_composition": {
            "headline": "요청의 강조 메시지, 없으면 대표 확인 사실",
            "subcopy": (
                "확인된 상품 특징을 빠짐없이 설명"
                if profile.generation_mode == "headline_and_subcopy"
                else "생성하지 않음"
            ),
            "subcopy_ending": _ending_policy(profile),
        },
        "output_limits": {
            "headline": {
                "hard_max_length": profile.limits.headline,
                "recommended_target_length": max(6, profile.limits.headline - 2),
            },
            "subcopy": (
                {
                    "hard_max_length": profile.limits.subcopy,
                    "recommended_target_length": max(10, profile.limits.subcopy - 2),
                }
                if profile.generation_mode == "headline_and_subcopy"
                else None
            ),
            "initial_option_count": 2,
        },
    }


def build_banner_copy_prompt(
    ad_input: dict[str, Any],
    *,
    image_provided: bool = False,
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=image_provided)
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "사용자가 제목과 보조 문구의 역할을 따로 지정하지 않아도 자동 구성 정책에 따라 "
        "광고 배너 문구 선택지 2개를 작성하세요.\n"
        + json.dumps(_prompt_payload(profile), ensure_ascii=False, indent=2)
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
        **_prompt_payload(profile),
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
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "검증에 실패한 선택지만 교체하세요. locked_options는 다시 출력하거나 수정하지 마세요. "
        "replace_option_indices 순서에 맞춰 교체 선택지만 options 배열에 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def parse_banner_copy_options(
    output_text: str,
    ad_input: dict[str, Any],
    *,
    expected_count: int = 2,
) -> list[AdCopy]:
    """길이 초과 문구도 보존해 후보별 검증과 부분 재시도가 가능하게 한다."""

    profile = analyze_copy_input(ad_input)
    try:
        payload = json.loads(output_text)
        raw_options = payload["options"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CopyGenerationError("AI 응답을 배너 문구 형식으로 읽을 수 없습니다.") from exc
    if not isinstance(raw_options, list) or len(raw_options) != expected_count:
        raise CopyGenerationError(f"광고 문구 선택지는 정확히 {expected_count}개여야 합니다.")

    hashtags = build_deterministic_hashtags(ad_input)
    copies: list[AdCopy] = []
    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            raise CopyGenerationError("광고 문구 선택지 형식이 올바르지 않습니다.")
        raw_headline = raw_option.get("headline")
        if not isinstance(raw_headline, str) or not raw_headline.strip():
            raise CopyGenerationError("비어 있는 광고 제목이 있습니다.")
        headline = " ".join(raw_headline.split())
        if profile.generation_mode == "headline_and_subcopy":
            raw_subcopy = raw_option.get("subcopy")
            if not isinstance(raw_subcopy, str) or not raw_subcopy.strip():
                raise CopyGenerationError("제목+보조 문구 모드에는 보조 문구가 필요합니다.")
            body = " ".join(raw_subcopy.split())
        else:
            if "subcopy" in raw_option or "body" in raw_option:
                raise CopyGenerationError("제목만 모드에서는 보조 문구를 생성하지 않습니다.")
            body = ""
        copies.append({"headline": headline, "body": body, "hashtags": hashtags.copy()})
    return copies


def _fact_required_roots(fact: str) -> list[str]:
    roots = [
        root for root in _terms(fact)
        if root not in _GENERIC_FACT_TERMS
    ]
    if any(root in _RELATION_ROOTS for root in roots):
        return list(dict.fromkeys(roots))
    specific = [root for root in roots if root not in {"관계"}]
    return list(dict.fromkeys(specific or roots))


def _fact_is_covered_strict(text: str, fact: str) -> bool:
    roots = _fact_required_roots(fact)
    if not roots:
        return bool(_normalize_compact(fact) in _normalize_compact(text))
    text_terms = set(_terms(text))
    return all(_term_is_reflected(root, text_terms, text) for root in roots)


def _fact_is_touched(text: str, fact: str) -> bool:
    """제목에서 한 사실의 대표 핵심어가 드러나는지 확인한다."""

    roots = _fact_required_roots(fact)
    text_terms = set(_terms(text))
    return any(_term_is_reflected(root, text_terms, text) for root in roots)


def _unknown_direct_modifiers(
    text: str,
    profile: CopyInputProfile,
) -> list[str]:
    """상품 사실 명사를 가까이 꾸미는 입력 외 관형 수식어를 찾는다."""

    fact_roots = set(_terms(profile.verified_facts_text))
    words = re.findall(r"[0-9A-Za-z가-힣]+", text)
    unknown: list[str] = []
    for index, word in enumerate(words):
        roots = _terms(word)
        root = roots[0] if roots else ""
        if root not in _PRODUCT_ATTRIBUTE_NOUNS:
            continue
        for previous in words[max(0, index - 3) : index]:
            previous_roots = _terms(previous)
            previous_root = previous_roots[0] if previous_roots else ""
            if not previous_root or previous_root in fact_roots:
                continue
            if _MODIFIER_ENDING_PATTERN.search(previous):
                unknown.append(previous)
    return list(dict.fromkeys(unknown))


def _unsupported_product_expressions(
    text: str,
    profile: CopyInputProfile,
    *,
    creative_scope: str,
    allow_contextual_use: bool,
) -> list[str]:
    dimensions = parse_request_dimensions(profile.copy_request)
    context_terms = tuple(dict.fromkeys(
        _CAMPAIGN_CONTEXT_NOUNS
        + tuple(dimensions["target_audience"])
        + tuple(dimensions["usage_context"])
        + tuple(dimensions["emotional_message"])
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
                product_terms=tuple(_PRODUCT_CLAIM_NOUNS),
                context_terms=context_terms,
                allow_contextual_use=contextual,
            )
        )
    labels.extend(_unknown_direct_modifiers(text, profile))
    normalized_facts = _normalize_compact(profile.verified_facts_text)
    for term in RISKY_CLAIM_TERMS:
        if term in text and _normalize_compact(term) not in normalized_facts:
            labels.append(term)
    if not allow_contextual_use:
        normalized_text = _normalize_compact(text)
        for root, label in _UNVERIFIED_ARRANGEMENT_ROOTS.items():
            if root in normalized_text and root not in normalized_facts:
                labels.append(label)
    return list(dict.fromkeys(labels))


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    profile = analyze_copy_input(ad_input)
    dimensions = parse_request_dimensions(profile.copy_request)
    constraints = parse_request_constraints(profile.copy_request)
    creative_scope = str(dimensions["creative_scope"])
    normalized_menu = _normalize_compact(profile.menu_name)
    issues: list[CopyValidationIssue] = []

    if len(copies) != 2:
        return [CopyValidationIssue(
            "OPTION_COUNT", "최종 광고 문구 선택지는 정확히 2개여야 합니다."
        )]

    signatures: list[str] = []
    for index, copy in enumerate(copies, start=1):
        headline = " ".join(copy["headline"].split())
        body = " ".join(copy["body"].split())
        signatures.append(_normalize_compact(headline))

        if len(headline) > profile.limits.headline:
            over = len(headline) - profile.limits.headline
            issues.append(CopyValidationIssue(
                "HEADLINE_TOO_LONG",
                f"선택지 {index} 제목은 {len(headline)}자입니다. 제한 {profile.limits.headline}자에 맞게 최소 {over}자를 줄이세요.",
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
            not constraints.allow_menu_name
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

        covered_headline_facts = [
            fact for fact in profile.verified_facts
            if (
                _fact_is_covered_strict(headline, fact)
                if profile.generation_mode == "headline_only"
                else _fact_is_touched(headline, fact)
            )
        ]
        required_headline_facts = (
            list(profile.verified_facts)
            if constraints.requires_multi_fact_coverage
            else []
        )
        if creative_scope == "fact_focused" and not covered_headline_facts:
            issues.append(CopyValidationIssue(
                "HEADLINE_NOT_GROUNDED",
                f"선택지 {index} 제목에 확인된 상품 특징이 구체적으로 반영되지 않았습니다.",
                index,
                "headline",
            ))
        missing_required = [
            fact for fact in required_headline_facts
            if fact not in covered_headline_facts
        ]
        if profile.generation_mode == "headline_only" and missing_required:
            issues.append(CopyValidationIssue(
                "HEADLINE_MISSING_REQUESTED_FACTS",
                f"선택지 {index} 제목에 함께 강조하도록 요청한 특징이 빠졌습니다: {', '.join(missing_required)}",
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
            if len(body) > profile.limits.subcopy:
                over = len(body) - profile.limits.subcopy
                issues.append(CopyValidationIssue(
                    "SUBCOPY_TOO_LONG",
                    f"선택지 {index} 보조 문구는 {len(body)}자입니다. 제한 {profile.limits.subcopy}자에 맞게 최소 {over}자를 줄이세요.",
                    index,
                    "subcopy",
                ))
            missing_facts = [
                fact for fact in profile.verified_facts
                if not _fact_is_covered_strict(body, fact)
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
                    f"선택지 {index} 보조 문구에 가격·행사 또는 이모지가 있습니다.",
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
            if _DANGLING_SUBCOPY_PATTERN.fullmatch(body):
                issues.append(CopyValidationIssue(
                    "DANGLING_SUBCOPY",
                    f"선택지 {index} 보조 문구가 뒤에 꾸밀 말이 필요한 관형형으로 끝납니다.",
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

        for phrase in constraints.required_phrases:
            pair_text = headline if profile.generation_mode == "headline_only" else f"{headline} {body}"
            if _normalize_compact(phrase) not in _normalize_compact(pair_text):
                issues.append(CopyValidationIssue(
                    "MISSING_REQUIRED_PHRASE",
                    f"선택지 {index}에 필수 표현 '{phrase}'이 없습니다.",
                    index,
                ))
        for phrase in constraints.forbidden_phrases:
            if _normalize_compact(phrase) in _normalize_compact(f"{headline} {body}"):
                issues.append(CopyValidationIssue(
                    "FORBIDDEN_PHRASE",
                    f"선택지 {index}에 제외 요청한 표현 '{phrase}'이 있습니다.",
                    index,
                ))

    if len(set(signatures)) != len(signatures):
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


def _fit_words(value: str, limit: int) -> str:
    normalized = " ".join(value.split()).strip(" .!?。！？,，")
    if len(normalized) <= limit:
        return normalized
    shortened = normalized[:limit].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" ·,，")


def _fallback_body(profile: CopyInputProfile) -> str:
    if len(profile.verified_facts_text) <= profile.limits.subcopy:
        return profile.verified_facts_text
    selected: list[str] = []
    for fact in profile.verified_facts:
        candidate = " · ".join(selected + [fact])
        if len(candidate) <= profile.limits.subcopy:
            selected.append(fact)
    return " · ".join(selected) or _fit_words(
        profile.verified_facts[0], profile.limits.subcopy
    )


def _fallback_titles(profile: CopyInputProfile) -> list[str]:
    dimensions = parse_request_dimensions(profile.copy_request)
    contexts = list(dimensions["usage_context"])
    emotions = list(dimensions["emotional_message"])
    values: list[str] = []

    def add(value: str) -> None:
        fitted = _fit_words(value, profile.limits.headline)
        if 0 < len(fitted) < 4:
            suffix = "에 집중"
            fitted = _fit_words(
                f"{fitted}{suffix}",
                profile.limits.headline,
            )
        if fitted and _normalize_compact(fitted) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(fitted)

    if contexts and emotions:
        add(f"{contexts[0]}의 {emotions[0]}")
    elif emotions:
        add(f"{emotions[0]}의 시간")
    primary = profile.verified_facts[0]
    if profile.generation_mode == "headline_and_subcopy":
        for suffix in ("에 집중", " 그대로"):
            core = _fit_words(primary, max(4, profile.limits.headline - len(suffix)))
            add(f"{core}{suffix}")
        for fact in profile.verified_facts[1:]:
            core = _fit_words(fact, max(4, profile.limits.headline - 4))
            add(f"{core}에 집중")
    else:
        for fact in profile.verified_facts:
            add(fact)
            if "조합" in fact:
                add(fact.replace("조합", "만남"))
        for suffix in ("에 집중", " 그대로"):
            if len(primary) + len(suffix) <= profile.limits.headline:
                add(f"{primary}{suffix}")
    return values[:2]


def _safe_fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    profile = analyze_copy_input(ad_input)
    titles = _fallback_titles(profile)
    while len(titles) < 2:
        titles.append(_fit_words(profile.verified_facts[0], profile.limits.headline))
    body = _fallback_body(profile)
    hashtags = build_deterministic_hashtags(ad_input)
    copies = [
        {
            "headline": title,
            "body": body if profile.generation_mode == "headline_and_subcopy" else "",
            "hashtags": hashtags.copy(),
        }
        for title in titles[:2]
    ]
    return copies


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
        if targeted and copies is not None:
            prompt = build_replacement_prompt(
                ad_input,
                copies,
                replacement_indices,
                issues,
                image_provided=image_bytes is not None,
            )
        else:
            prompt = build_banner_copy_prompt(
                ad_input,
                image_provided=image_bytes is not None,
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
                response_schema=_response_schema(profile, option_count),
            )
            returned = parse_banner_copy_options(
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
                notice=(
                    None
                    if attempt == 1
                    else "검증에 실패한 선택지만 다시 생성했습니다."
                ),
                validation_issues=first_issues,
                attempt_history=tuple(history),
                prompt_version=COPY_PROMPT_VERSION,
            )
        current_messages = tuple(issue.message for issue in issues)
        if not first_issues:
            first_issues = current_messages

    fallback = _safe_fallback_options(ad_input)
    fallback_issues = collect_banner_copy_issues(fallback, ad_input)
    return BannerCopyGenerationResult(
        copies=fallback,
        generation_mode=profile.generation_mode,
        status="fallback",
        attempts=max_attempts,
        notice=(
            "모델 문구가 검증을 통과하지 못해 확인된 입력 기반 문구를 사용했습니다."
            + (
                " fallback 진단: "
                + " / ".join(issue.message for issue in fallback_issues)
                if fallback_issues
                else ""
            )
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=COPY_PROMPT_VERSION,
    )

