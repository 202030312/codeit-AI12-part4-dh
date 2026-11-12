"""구조화 사실 전용 문구 생성과 부분 fallback을 적용한 v1.10.2."""

from __future__ import annotations

from itertools import combinations
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
from src.features.ad_copy.internal.base_rules import (
    _normalize_compact,
    _terms,
    parse_request_constraints,
    parse_request_dimensions,
    title_lexical_similarity,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.base_workflow as base
import src.features.ad_copy.visual_analysis as visual_base
import src.features.ad_copy.internal.visual_contract as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.2-structured-facts-partial-fallback"
VISUAL_PROMPT_VERSION = visual_base.VISUAL_PROMPT_VERSION


BANNER_COPY_SYSTEM_INSTRUCTIONS = """당신은 카페·음료·디저트·베이커리 소상공인을 위한 한국어 광고 배너 카피라이터입니다.

구조화된 입력만 근거로 제목 또는 제목+보조 문구 선택지 2개를 작성하세요. 원본 상품 이미지는 문구 생성 모델에 제공되지 않습니다.

[근거 우선순위]

1. verified_product_facts는 맛·향·풍미·식감·재료·구성·제조 방식 등 상품 사실의 유일한 근거입니다.
2. verified_visual_facts는 앞 단계에서 사진으로 확인한 종류·색상·형태·개수·용기·포장·보이는 구성·배치의 근거입니다.
3. copy_request는 스타일·톤·타깃·이용 상황·감성 메시지·길이를 조절하는 요청입니다. 상품 사실, 고객 반응, 인기도, 효능의 근거가 아닙니다.
4. menu.name은 상품 식별용이며 배너에 별도로 표시되므로 요청이 없으면 전체 메뉴명을 문구에 반복하지 마세요.

[사실 보존]

1. 입력에 없는 맛·향·풍미·식감·온도·재료·배치·제조 방식·원산지·신선도·효능·인기·고객 반응을 만들지 마세요.
2. 상품 특징에 새로운 강도나 평가 수식어를 붙이지 마세요. 예: 진한, 깊은, 풍부한, 은은한, 향긋한, 가득, 극도로, 완벽한.
3. verified_visual_facts에 없는 담긴·올린·얹은·위에·아래에 같은 배치 표현을 만들지 마세요. 보이는 레몬·얼음·토핑에 상큼한·시원한·달콤한 같은 감각 수식어를 붙이지 마세요.
4. 감각 범주를 바꾸지 마세요. 식감은 맛이나 향으로, 향은 맛이나 식감으로, 풍미는 향이나 식감으로 바꾸지 마세요.
5. 보이는 얼음은 시원한 맛의 근거가 아니고, 갈색 표면은 바삭한 식감의 근거가 아닙니다.
6. A와 B의 조합을 A를 B에 얹은·섞은·채운 것으로 구체화하지 마세요.
7. 든든한·건강한·부담 없는·속이 편한·활력을 주는·에너지를 채우는 등 포만감·건강·효용 표현은 verified_product_facts에 같은 근거가 있을 때만 사용하세요.
8. 질리지 않는·손이 가는·계속 찾는·재구매·인기 메뉴·누구나 좋아하는·중독되는·후회 없는 등 고객 반응·반복 행동·인기도·만족도 주장을 만들지 마세요.

[자동 구성]

- headline_only: 후보 1은 가장 중요한 확인 사실을 간결하게 표현합니다. 후보 2는 요청된 광고 콘셉트 또는 다른 확인 사실·시각 사실을 사용합니다.
- headline_and_subcopy: 제목은 요청된 광고 콘셉트가 있으면 그 메시지를, 없으면 상품 또는 시각 사실을 사용합니다. 보조 문구는 verified_product_facts를 표현 변화 없이 우선 사용합니다.
- 스타일·톤만 입력되고 광고 콘셉트가 없으면 여유·행복·휴식·순간 같은 새 메시지를 만들지 마세요.
- 제목 목표 길이는 12~15자이며 hard_max_length를 절대 넘지 마세요.
- 제목과 보조 문구는 독립적으로 완결되어야 합니다. 뒤에 꾸밀 말이 필요한 형태로 끝내지 마세요.
- 가격·행사·혜택·후기·해시태그·최상급·이모지를 만들지 마세요.
- 제목과 보조 문구 모두 즐겨보세요·느껴보세요·만나보세요·선택하세요 같은 명령형 CTA를 사용하지 마세요.
- 같은 단어를 반복하지 말고, 두 후보는 핵심 관점이 달라야 합니다. 어순만 바꾸거나 ‘집중·그대로’만 바꾼 후보를 만들지 마세요.

[출력 형식]

headline_only:
{"options":[{"headline":"제목 1"},{"headline":"제목 2"}]}

headline_and_subcopy:
{"options":[{"headline":"제목 1","subcopy":"보조 문구 1"},{"headline":"제목 2","subcopy":"보조 문구 2"}]}

최초 생성은 선택지 2개를 반환합니다. repair 요청은 replace_option_indices에 해당하는 교체 선택지만 같은 순서로 반환합니다. locked_options는 다시 출력하거나 수정하지 않습니다. JSON 객체 이외의 설명이나 마크다운을 출력하지 마세요."""


_SENSORY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "맛": ("맛",),
    "향": ("향", "향긋"),
    "풍미": ("풍미",),
    "식감": ("식감", "질감", "촉감"),
}

_CTA_PATTERN = re.compile(
    r"(?:만나|즐겨|느껴|확인|선택|경험|맛보|놓치지\s*말아).*세요"
)
_UNSUPPORTED_BENEFIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("든든한 포만감", re.compile(r"든든")),
    ("건강 효용", re.compile(r"건강")),
    ("부담이 없다는 효용", re.compile(r"부담\s*(?:없|없이)")),
    ("속이 편하다는 효용", re.compile(r"속(?:이)?\s*편")),
    ("활력을 준다는 효용", re.compile(r"활력(?:을)?\s*(?:주|채우|더하)")),
    ("에너지를 준다는 효용", re.compile(r"에너지(?:를)?\s*(?:주|채우|더하)")),
    (
        "감정·휴식을 준다는 효용",
        re.compile(
            r"(?:휴식|여유|행복|기분\s*전환)(?:을|를)?\s*"
            r"(?:주|선사|채우|더하).{0,8}(?:맛|향|풍미|식감|음료|빵|디저트|커피|한\s*잔)"
        ),
    ),
)
_REPEATED_WORD_PATTERN = re.compile(
    r"(?P<word>[가-힣]{2,})(?:의|을|를|이|가|은|는|과|와|로|으로|에|에서)?\s+(?P=word)"
)
_GENERIC_TITLE_SUFFIX_PATTERN = re.compile(
    r"(?:에집중|그대로|의매력|포인트|한눈에)$"
)
_INCOMPLETE_HEADLINE_PATTERN = re.compile(r"(?:있는|없는)$")
_INCOMPLETE_COUNT_PATTERN = re.compile(
    r"(?:\s\d+|(?:형태|모양)의\s*\d+개)$"
)
_COLOR_ECHO_PATTERN = re.compile(
    r"(?P<color>붉은|노란|초록|파란|하얀|검은|보라|갈)색\s+(?P=color)"
)
_LATIN_TEXT_PATTERN = re.compile(r"[A-Za-z]{2,}")
_RISKY_SENSORY_MODIFIERS: dict[str, str] = {
    "상큼": "상큼한",
    "달콤": "달콤한",
    "고소": "고소한",
    "시원": "시원한",
    "향긋": "향긋한",
    "쌉싸름": "쌉싸름한",
    "바삭": "바삭한",
    "촉촉": "촉촉한",
    "쫄깃": "쫄깃한",
    "부드러": "부드러운",
}


def _visual_facts(ad_input: dict[str, Any]) -> list[dict[str, str]]:
    return [
        fact
        for fact in visual_base._visual_facts(ad_input)
        if not _LATIN_TEXT_PATTERN.search(str(fact.get("value") or ""))
    ]


def _prompt_payload(profile: Any, ad_input: dict[str, Any]) -> dict[str, Any]:
    constraints = parse_request_constraints(profile.copy_request)
    dimensions = parse_request_dimensions(profile.copy_request)
    has_campaign_concept = bool(dimensions["has_campaign_concept"])
    return {
        "raw_product_image": {
            "sent_to_copy_model": False,
            "policy": "verified_visual_facts만 시각 근거로 사용",
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
            "must_preserve_sensory_categories": True,
            "subcopy_policy": "가능하면 입력 표현을 그대로 사용",
        },
        "verified_visual_facts": {
            "facts": _visual_facts(ad_input),
            "may_support": [
                "종류", "색상", "형태", "개수", "용기", "포장", "보이는 구성", "배치"
            ],
            "must_not_support": [
                "맛", "향", "풍미", "식감", "온도 체감", "신선도", "효능"
            ],
        },
        "excluded_visual_inferences": ad_input.get("campaign", {}).get(
            "excluded_visual_inferences", []
        ),
        "generation_mode": profile.generation_mode,
        "copy_request": {
            "raw": profile.copy_request or None,
            "parsed_scope": dimensions["creative_scope"],
            "campaign_concepts": {
                "target_audience": dimensions["target_audience"],
                "usage_context": dimensions["usage_context"],
                "emotional_message": dimensions["emotional_message"],
            },
            "style_and_tone_are_not_fact_sources": True,
        },
        "auto_composition": {
            "headline": (
                "요청된 광고 콘셉트 또는 서로 다른 확인 사실·시각 사실"
                if has_campaign_concept
                else "verified_product_facts 또는 verified_visual_facts"
            ),
            "subcopy": (
                "verified_product_facts를 표현 범주 변경 없이 설명"
                if profile.generation_mode == "headline_and_subcopy"
                else "생성하지 않음"
            ),
            "candidate_diversity": "어순 변경이나 집중·그대로 교체만으로 구분하지 않음",
        },
        "output_limits": {
            "headline": {
                "target_length": "12~15자",
                "hard_max_length": profile.limits.headline,
            },
            "subcopy": (
                {"hard_max_length": profile.limits.subcopy}
                if profile.generation_mode == "headline_and_subcopy"
                else None
            ),
            "initial_option_count": 2,
        },
    }


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    profile = analyze_copy_input(ad_input, image_provided=False)
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "구조화된 상품 사실과 고신뢰 시각 사실만 사용해 서로 다른 광고 문구 선택지 2개를 작성하세요.\n"
        + json.dumps(_prompt_payload(profile, ad_input), ensure_ascii=False, indent=2)
    )


def build_replacement_prompt(
    ad_input: dict[str, Any],
    copies: list[AdCopy],
    option_indices: list[int],
    issues: list[CopyValidationIssue],
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=False)
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
            "상품 사실의 맛·향·풍미·식감 범주 유지",
            "제목과 보조 문구의 명령형 CTA 제거",
            "입력에 없는 강도·효용·고객 반응 제거",
            "같은 단어 반복과 후보 간 의미 중복 제거",
            "보조 문구에는 verified_product_facts 표현 우선 사용",
            "locked_options와 다른 관점의 제목 작성",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "실패한 선택지만 교체하세요. locked_options는 유지하고 replace_option_indices 순서의 교체 선택지만 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _sensory_categories(text: str) -> set[str]:
    compact = _normalize_compact(text)
    return {
        category
        for category, roots in _SENSORY_CATEGORIES.items()
        if any(root in compact for root in roots)
    }


def _headline_core(text: str) -> str:
    compact = _normalize_compact(text)
    previous_value = ""
    while compact != previous_value:
        previous_value = compact
        compact = _GENERIC_TITLE_SUFFIX_PATTERN.sub("", compact)
    return compact


def _visual_sensory_modifiers(
    text: str,
    verified_facts_text: str,
    ad_input: dict[str, Any],
) -> list[str]:
    compact_text = _normalize_compact(text)
    compact_facts = _normalize_compact(verified_facts_text)
    visual_terms = {
        term
        for fact in _visual_facts(ad_input)
        for term in _terms(str(fact.get("value") or ""))
        if len(term) >= 2
    }
    labels: list[str] = []
    for root, label in _RISKY_SENSORY_MODIFIERS.items():
        if root not in compact_text or root in compact_facts:
            continue
        if any(
            re.search(
                rf"(?:{re.escape(root)}.{{0,4}}{re.escape(term)}|{re.escape(term)}.{{0,4}}{re.escape(root)})",
                compact_text,
            )
            for term in visual_terms
        ):
            labels.append(label)
    return labels


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    profile = analyze_copy_input(ad_input)
    issues = list(previous.collect_banner_copy_issues(copies, ad_input))
    fact_categories = _sensory_categories(profile.verified_facts_text)

    for index, copy in enumerate(copies, start=1):
        for field, text in (("headline", copy["headline"]), ("subcopy", copy["body"])):
            if not text:
                continue
            if _LATIN_TEXT_PATTERN.search(text):
                issues.append(CopyValidationIssue(
                    "NON_KOREAN_COPY",
                    f"선택지 {index} {field}에 영문 문구가 포함되어 있습니다.",
                    index,
                    field,
                ))
            added_categories = sorted(_sensory_categories(text) - fact_categories)
            if added_categories:
                issues.append(CopyValidationIssue(
                    "SENSORY_CATEGORY_CHANGED",
                    f"선택지 {index} {field}가 입력에 없는 감각 범주를 추가했습니다: {', '.join(added_categories)}",
                    index,
                    field,
                ))
            if _CTA_PATTERN.search(text):
                issues.append(CopyValidationIssue(
                    "IMPERATIVE_CTA",
                    f"선택지 {index} {field}에 명령형 행동 유도 문구가 있습니다.",
                    index,
                    field,
                ))
            unsupported_benefits = [
                label
                for label, pattern in _UNSUPPORTED_BENEFIT_PATTERNS
                if pattern.search(text) and not pattern.search(profile.verified_facts_text)
            ]
            if unsupported_benefits:
                issues.append(CopyValidationIssue(
                    "UNVERIFIED_PRODUCT_BENEFIT",
                    f"선택지 {index} {field}에 근거 없는 상품 효용 표현이 있습니다: {', '.join(unsupported_benefits)}",
                    index,
                    field,
                ))
            repeated = _REPEATED_WORD_PATTERN.search(text)
            if repeated:
                issues.append(CopyValidationIssue(
                    "REPEATED_WORD",
                    f"선택지 {index} {field}에 같은 핵심어가 반복됩니다: {repeated.group('word')}",
                    index,
                    field,
                ))
            color_echo = _COLOR_ECHO_PATTERN.search(text)
            if color_echo:
                issues.append(CopyValidationIssue(
                    "REPEATED_COLOR",
                    f"선택지 {index} {field}에 같은 색상 의미가 반복됩니다: {color_echo.group('color')}",
                    index,
                    field,
                ))
            visual_modifiers = _visual_sensory_modifiers(
                text,
                profile.verified_facts_text,
                ad_input,
            )
            if visual_modifiers:
                issues.append(CopyValidationIssue(
                    "VISUAL_SENSORY_INFERENCE",
                    f"선택지 {index} {field}가 보이는 구성에 입력 근거 없는 감각 수식어를 붙였습니다: {', '.join(visual_modifiers)}",
                    index,
                    field,
                ))
        if _INCOMPLETE_HEADLINE_PATTERN.search(copy["headline"].strip()):
            issues.append(CopyValidationIssue(
                "INCOMPLETE_HEADLINE",
                f"선택지 {index} 제목이 뒤에 꾸밀 말이 필요한 형태로 끝납니다.",
                index,
                "headline",
            ))
        if _INCOMPLETE_COUNT_PATTERN.search(copy["headline"].strip()):
            issues.append(CopyValidationIssue(
                "INCOMPLETE_COUNT_PHRASE",
                f"선택지 {index} 제목의 수량 표현에 단위 또는 대상이 빠졌습니다.",
                index,
                "headline",
            ))

    if len(copies) == 2:
        first = copies[0]["headline"]
        second = copies[1]["headline"]
        first_core = _headline_core(first)
        second_core = _headline_core(second)
        if (
            first_core
            and first_core == second_core
            or title_lexical_similarity(first, second) >= 0.5
        ):
            issues.append(CopyValidationIssue(
                "LOW_CANDIDATE_DIVERSITY",
                "두 선택지의 제목이 어순이나 일반적인 종결 표현만 다르고 핵심 관점이 같습니다.",
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
    return base._fit_words(value, limit)


def _campaign_fallback_title(profile: Any) -> str | None:
    dimensions = parse_request_dimensions(profile.copy_request)
    contexts = list(dict.fromkeys(dimensions["usage_context"]))
    emotions = list(dict.fromkeys(dimensions["emotional_message"]))
    if contexts and emotions:
        context = contexts[0]
        emotion = emotions[0]
        if _normalize_compact(context) == _normalize_compact(emotion):
            return f"{emotion} 시간"
        if _normalize_compact(emotion) in _normalize_compact(context):
            return context
        if _normalize_compact(context) in _normalize_compact(emotion):
            return emotion
        return f"{context}의 {emotion}"
    if contexts:
        return contexts[0]
    if emotions:
        return f"{emotions[0]} 시간"
    return None


def _visual_fallback_titles(ad_input: dict[str, Any], limit: int) -> list[str]:
    facts = _visual_facts(ad_input)
    product_types = [
        str(fact.get("value") or "").strip()
        for fact in facts
        if fact.get("category") == "product_type"
    ]
    priority = {
        "count": 0,
        "shape": 1,
        "color": 2,
        "packaging": 3,
        "product_type": 4,
        "visible_component": 5,
        "container": 6,
        "arrangement": 7,
    }
    values: list[str] = []
    for fact in sorted(facts, key=lambda item: priority.get(str(item.get("category")), 99)):
        value = str(fact.get("value") or "").strip()
        if not value:
            continue
        if fact.get("category") == "count" and re.fullmatch(r"\d+", value):
            value = f"{value}개"
        if len(value) < 4 and product_types and value != product_types[0]:
            value = (
                f"{product_types[0]} {value}"
                if fact.get("category") == "count"
                else f"{value} {product_types[0]}"
            )
        fitted = _fit_words(value, limit)
        if len(fitted) >= 4 and _normalize_compact(fitted) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(fitted)
    return values


def _fallback_title_candidates(ad_input: dict[str, Any]) -> list[str]:
    profile = analyze_copy_input(ad_input)
    values: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        fitted = _fit_words(value, profile.limits.headline)
        if len(fitted) < 4:
            return
        if _REPEATED_WORD_PATTERN.search(fitted):
            return
        if _normalize_compact(fitted) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(fitted)

    campaign_title = _campaign_fallback_title(profile)
    if campaign_title:
        add(campaign_title)
    for fact in profile.verified_facts:
        add(fact)
        adjective_flavor = re.fullmatch(r"(.+?)한\s+풍미", fact)
        if adjective_flavor:
            add(f"{adjective_flavor.group(1)}함을 중심으로")
    for value in _visual_fallback_titles(ad_input, profile.limits.headline):
        add(value)
    add(f"{profile.verified_facts[0]} 포인트")
    add(f"{profile.verified_facts[0]} 그대로")
    return values


def _safe_fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    profile = analyze_copy_input(ad_input)
    body = base._fallback_body(profile)
    hashtags = build_deterministic_hashtags(ad_input)
    candidates = _fallback_title_candidates(ad_input)
    for first, second in combinations(candidates, 2):
        copies = [
            {
                "headline": first,
                "body": body if profile.generation_mode == "headline_and_subcopy" else "",
                "hashtags": hashtags.copy(),
            },
            {
                "headline": second,
                "body": body if profile.generation_mode == "headline_and_subcopy" else "",
                "hashtags": hashtags.copy(),
            },
        ]
        if not collect_banner_copy_issues(copies, ad_input):
            return copies
    raise CopyGenerationError("검증을 통과하는 안전 fallback 두 개를 구성할 수 없습니다.")


def _resolve_partial_fallback(
    ad_input: dict[str, Any],
    copies: list[AdCopy] | None,
    issues: list[CopyValidationIssue],
) -> tuple[list[AdCopy], tuple[int, ...]]:
    fallback = _safe_fallback_options(ad_input)
    if copies is None:
        return fallback, (1, 2)
    invalid_indices = {
        issue.option_index
        for issue in issues
        if issue.option_index is not None
    }
    if any(issue.option_index is None for issue in issues) or not invalid_indices:
        invalid_indices = {1, 2}
    resolved = [dict(copy) for copy in copies]
    for option_index in sorted(invalid_indices):
        other_index = 2 if option_index == 1 else 1
        replacement = next(
            (
                candidate
                for candidate in fallback
                if _normalize_compact(candidate["headline"])
                != _normalize_compact(resolved[other_index - 1]["headline"])
            ),
            fallback[option_index - 1],
        )
        resolved[option_index - 1] = dict(replacement)
    if collect_banner_copy_issues(resolved, ad_input):
        return fallback, (1, 2)
    return resolved, tuple(sorted(invalid_indices))


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    """구조화된 사실만 전송하고 실패 후보만 fallback으로 교체한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
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
            build_replacement_prompt(ad_input, copies, replacement_indices, issues)
            if targeted and copies is not None
            else build_banner_copy_prompt(ad_input)
        )
        prior_copies = [dict(copy) for copy in copies] if copies is not None else None
        prior_issues = list(issues)
        output_text = ""
        try:
            output_text = _generate_with_gemini(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
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
                "raw_image_sent": False,
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
                "raw_image_sent": False,
                "raw_output": output_text,
                "options": [],
                "issues": [error_message],
            })
            if not first_issues:
                first_issues = (error_message,)
            if targeted and prior_copies is not None:
                copies = prior_copies
                issues = prior_issues
            else:
                copies = None
                issues = []
            continue

        if not issues:
            return BannerCopyGenerationResult(
                copies=copies,
                generation_mode=profile.generation_mode,
                status="generated" if attempt == 1 else "retried",
                attempts=attempt,
                notice=None if attempt == 1 else "검증에 실패한 선택지만 다시 생성했습니다.",
                validation_issues=first_issues,
                attempt_history=tuple(history),
                prompt_version=COPY_PROMPT_VERSION,
            )
        if not first_issues:
            first_issues = tuple(issue.message for issue in issues)

    resolved, fallback_indices = _resolve_partial_fallback(ad_input, copies, issues)
    return BannerCopyGenerationResult(
        copies=resolved,
        generation_mode=profile.generation_mode,
        status="fallback",
        attempts=max_attempts,
        notice=(
            "모델 검증에 실패한 선택지만 확인된 입력 기반 fallback으로 교체했습니다. "
            f"교체 선택지: {', '.join(map(str, fallback_indices))}"
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=COPY_PROMPT_VERSION,
    )

