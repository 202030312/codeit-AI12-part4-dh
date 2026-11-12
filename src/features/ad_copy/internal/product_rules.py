"""상품명과 중복되는 시각 제품 종류를 메시지에서 제외하는 v1.10.14."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, permutations
import json
import re
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    _generate_with_gemini_generate_content as _generate_with_gemini,
    _generate_with_openai,
)
from src.features.ad_copy.internal.base_rules import (
    _normalize_compact,
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
import src.features.ad_copy.internal.ad_value_rules as ad_value_base
import src.features.ad_copy.internal.quality_recovery as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.14-product-type-dedup"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = previous.INTERNAL_CANDIDATE_COUNT
FINAL_OPTION_COUNT = previous.FINAL_OPTION_COUNT
MIN_HEADLINE_AD_SCORE = previous.MIN_HEADLINE_AD_SCORE
MIN_AD_SCORE = previous.MIN_AD_SCORE
SUBCOPY_RECOMMENDED_MIN = previous.SUBCOPY_RECOMMENDED_MIN
SUBCOPY_RECOMMENDED_MAX = previous.SUBCOPY_RECOMMENDED_MAX
SUBCOPY_HARD_MAX = previous.SUBCOPY_HARD_MAX


def _provider_copy_prompt_version(provider: str) -> str:
    """같은 v1.10.14 규칙을 실제 문구 생성 제공자 이름으로 기록한다."""

    return f"{provider}_copy_v1.10.14-product-type-dedup"


def _method_only_system_instructions(value: str) -> str:
    """기존 안전 규칙은 유지하되 보조 문구 형식을 하나로 강제하지 않는다."""

    value = re.sub(
        r"\n5\. '원통형 컵'[^\n]+\n",
        "\n",
        value,
    )
    value = re.sub(
        r"\n5\. '초콜릿 풍미'[^\n]+\n",
        (
            "\n5. 보조 문구는 배정된 근거와 목적에 맞춰 의미가 완결된 "
            "설명 문장·명사구·짧은 정보 나열 중 하나로 작성하세요.\n"
        ),
        value,
    )
    value = re.sub(
        r"\n6\. '느낄 수 있어요'[^\n]+\n",
        (
            "\n6. 종결형 어미를 일률적으로 붙이지 마세요. 직접 명령형 CTA는 "
            "입력에 판매·방문·행동 유도 요청이 명확할 때만 사용하세요.\n"
        ),
        value,
    )
    value = value.replace(
        (
            "보조 문구는 제목을 다시 말하는 짧은 라벨이 아니라 "
            "verified_product_facts의 특징을 소비자가 이해할 수 있게 설명하는 "
            "완결된 한 문장입니다."
        ),
        (
            "보조 문구는 제목을 다시 말하지 않고 배정된 두 번째 근거를 전달합니다. "
            "설명 문장·완결된 명사구·짧은 정보 나열 중 메시지와 지면에 맞는 "
            "형식을 선택합니다."
        ),
    )
    value = value.replace(
        (
            "- headline_and_subcopy: 제목은 요청된 광고 콘셉트가 있으면 그 메시지를, "
            "없으면 상품 또는 시각 사실을 사용합니다. 보조 문구는 "
            "verified_product_facts를 표현 변화 없이 우선 사용합니다."
        ),
        (
            "- headline_and_subcopy: role_assignment의 primary_message를 제목에, "
            "secondary_message를 보조 문구에 사용하고 두 메시지가 서로 다른 정보를 "
            "전달하게 합니다."
        ),
    )
    value = value.replace(
        (
            "- 제목과 보조 문구 모두 즐겨보세요·느껴보세요·만나보세요·선택하세요 "
            "같은 명령형 CTA를 사용하지 마세요."
        ),
        (
            "- 제목에는 명령형 CTA를 사용하지 마세요. 보조 문구의 명령형 CTA는 "
            "입력에 판매·방문·행동 유도 요청이 명확할 때만 허용합니다."
        ),
    )
    value = value.replace(
        (
            "보조 문구에는 verified_product_facts의 원문 핵심 표현을 최소 한 개 "
            "그대로 포함하세요. 입력에 없는 맛·향·식감·강도·재료는 추가하지 마세요."
        ),
        (
            "보조 문구에는 secondary_message의 핵심 표현을 포함하세요. 입력에 없는 "
            "맛·향·식감·강도·재료는 추가하지 마세요."
        ),
    )
    value = re.sub(r"\s*예:\s*[^.\n]+\.?", ".", value)
    value = re.sub(r"예를 들어\s*[^.\n]+\.?", "", value)
    value = value.replace(
        "수량·제품 구성·구별되는 토핑처럼 고객에게 의미 있는 정보",
        "구별되는 재료·토핑·장식·포장처럼 고객에게 의미 있는 정보",
    )
    value = value.replace(
        "가격·행사·혜택·후기·해시태그·최상급·이모지를 만들지 마세요.",
        "입력에 없는 가격·행사·혜택·후기·해시태그·최상급·이모지를 만들지 마세요.",
    )
    return value.replace("..", ".")


BANNER_COPY_SYSTEM_INSTRUCTIONS = _method_only_system_instructions(
    previous.BANNER_COPY_SYSTEM_INSTRUCTIONS
) + """

[v1.10.14 근거 메시지 분류·역할 배정과 보조 문구 형식 다양성]

1. message_inventory에 있는 근거 메시지만 사용하세요. 메시지는 상품·감성·상황·혜택 범주와 입력 출처를 가집니다.
2. role_assignment.primary_message를 제목에, secondary_message를 보조 문구에 배정하세요. 고정된 배너 유형을 선택하지 말고 주 메시지와 보조 메시지가 서로 보완되게 작성하세요.
3. 같은 상품명·재료명·제품 종류가 두 문구에 필요한 경우 반복할 수 있습니다. 그러나 같은 맛·향·식감·조합·혜택 주장을 어순이나 동의어만 바꿔 반복하지 마세요.
4. 연결된 복합 상품 특징은 하나의 주장입니다. 일부 감각 특징만 제목에 떼어 쓰고 보조 문구에서 전체 주장을 다시 설명하지 마세요.
5. secondary_message가 상품이 아니어도 근거가 있는 감성·상황·혜택이면 완결된 보조 문구로 사용할 수 있습니다.
6. headline_and_subcopy 요청은 안전한 두 번째 근거 메시지가 없거나 완결된 보조 문구를 만들 수 없을 때만 headline_only로 전환합니다.
7. 이미지에서 관찰된 개수는 그 자체로 제목 근거가 아닙니다. 사용자 입력에 판매 수량, 세트, 라인업, 포장 단위 또는 수량형 프로모션이 명시되고 관찰 수량과 일치할 때만 사용하세요.
8. 촬영 연출의 반복 배치, 토핑 개수, 판매 단위인지 확인되지 않은 수량은 사용하지 마세요.
9. 이미지의 재료·장식·포장을 상품 메시지로 사용할 수 있지만 이를 맛·향·풍미·식감으로 바꾸지 마세요.
10. 얼음·컵·테이블·물방울·일반 색상처럼 선택 이유가 되지 않는 시각 정보만으로 문구를 만들지 마세요.
11. 길이를 채우기 위해 새로운 감성·상황·맛·향·식감·강도·혜택을 만들지 마세요.
12. 제목은 최대 글자 수보다 2~4자 짧게 작성하고 단어·조사·어미를 자르지 마세요. 보조 문구는 배정된 메시지에 맞춰 의미가 완결된 설명 문장·명사구·짧은 정보 나열 중 적합한 형식을 선택하세요.
13. 재료의 조합이나 복합 식감처럼 관계 설명이 필요한 메시지는 문장형을 우선하고, 확인된 재료·풍미·식감의 간단한 정보는 명사구나 나열형을 사용할 수 있습니다.
14. 직접 명령형 CTA는 입력에 판매·방문·행동 유도 요청이 명확할 때만 사용하고, 요청이 없으면 설명형 또는 명사형으로 작성하세요.
15. 출력 전 각 문구의 배정 메시지, 같은 주장 반복, 복합 사실 보존, 수량 근거, 의미 완결성을 확인하고 실패 후보는 교체하세요.
16. 보조 문구에 안전하게 사용할 수 있는 형식이 두 가지 이상이면 내부 후보 4개에 가능한 한 두 가지 이상의 형식을 사용하세요.
17. 내부 후보와 최종 선택지에서 보조 문구 전체가 같은 후보를 반복하지 마세요. 최종 두 선택지는 서로 다른 형식을 우선하되, 안전한 후보가 한 형식뿐이면 같은 형식의 서로 다른 문구를 사용할 수 있습니다.
"""


_GENERIC_VISUAL_COMPONENTS = {
    "얼음", "물방울", "컵", "유리컵", "유리잔", "잔", "용기", "빨대",
    "표면", "컵 표면", "유리 표면", "음료", "액체", "거품", "테이블",
    "접시", "배경",
}
_LOW_VALUE_VISUAL_COMPONENT_ROOTS = ("부스러기",)
_SENSORY_ROOTS = ("맛", "향", "풍미", "식감", "질감")
_RELATION_ROOTS = ("조합", "조화", "만남", "함께", "어우러")
_EXPERIENCE_ROOTS = ("느낄수", "느껴", "돋보", "특징", "살린")
_INCOMPLETE_HEADLINE_END_PATTERN = re.compile(
    r"(?:위한|담은|더한|더해진|올린|얹은|보이는|느껴지는|돋보이는|전하는|"
    r"구성한|완성한|필요한|어울리는|함께하는|있는|없는|같은|새로운|"
    r"의)$"
)
_MALFORMED_MODIFIER_PARTICLE_PATTERN = re.compile(
    r"(?:한|난|운|로운|러운|스러운|적인)(?:이|가|을|를|은|는)(?:\s|$)"
)
_VALID_SINGLE_SYLLABLE_HEADLINE_ENDINGS = {
    "맛", "향", "잔", "결", "빵", "차", "컵", "봄", "밤", "쉼", "힘",
    "입", "점", "끼", "빛", "색", "날", "때", "꿈", "멋", "정", "폼",
}
_COMMON_HEADLINE_NOUNS = {
    "디저트", "케이크", "음료", "메뉴", "커피", "라떼", "스무디",
    "베이커리", "페이스트리", "브레드", "아메리카노", "밀크셰이크",
}
_KOREAN_COUNT_NUMBERS = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}
_COUNT_COUNTERS = (
    "개", "잔", "종", "가지", "세트", "팩", "병", "박스", "구", "입",
    "피스", "조각", "접시", "컵", "묶음", "구성",
)
_COUNT_TOKEN_PATTERN = re.compile(
    rf"(?<![0-9A-Za-z가-힣])(?P<number>\d+|{'|'.join(_KOREAN_COUNT_NUMBERS)})"
    rf"\s*(?:{ '|'.join(_COUNT_COUNTERS) })(?:의)?"
)
_PROMOTION_QUANTITY_PATTERN = re.compile(r"(?<!\d)(\d+)\s*\+\s*(\d+)(?!\d)")
_CTA_REQUEST_PATTERN = re.compile(
    r"(?:CTA|행동\s*유도|구매\s*유도|주문\s*유도|방문\s*유도|"
    r"선택\s*유도|판매\s*유도|클릭\s*유도)",
    re.IGNORECASE,
)
_CTA_DENIAL_PATTERN = re.compile(
    r"(?:CTA|행동\s*유도|구매\s*유도|주문\s*유도|방문\s*유도|"
    r"선택\s*유도|판매\s*유도|클릭\s*유도).{0,8}"
    r"(?:없이|제외|금지|사용하지|넣지\s*말)",
    re.IGNORECASE,
)
_CTA_END_PATTERN = re.compile(
    r"(?:만나|즐겨|느껴|확인|선택|주문|방문|경험|놓치지\s*말아).*세요[.!]?$"
)
_INCOMPLETE_SUBCOPY_END_PATTERN = re.compile(
    r"(?:담은|더한|올린|얹은|채운|살린|곁들인|어우러진|돋보이는|"
    r"느껴지는|완성한|위한|어울리는|함께하는|가득한|풍부한|"
    r"부드러운|바삭한|촉촉한|달콤한|고소한|상큼한|진한|깊은|은은한|"
    r"하고|하며|하면서|이지만|지만)$"
)
_STANDALONE_PARTICLE_END_PATTERN = re.compile(
    r"(?:^|\s)(?:이|가|을|를|은|는|의|와|과|로|으로|에|에서)$"
)
_INFORMATION_LIST_SEPARATOR_PATTERN = re.compile(r"\s*(?:,|·|/|\|)\s*")
_GENERAL_DESCRIPTIVE_END_PATTERN = re.compile(
    r"(?:습니다|[가-힣]니다|해요|예요|이에요|돼요|있어요|없어요|져요|려요|아요|어요)[.!]?$"
)


def _count_number(value: str) -> int | None:
    match = re.search(r"\d+", value)
    if match:
        return int(match.group())
    match = _COUNT_TOKEN_PATTERN.search(value)
    if match:
        return _KOREAN_COUNT_NUMBERS.get(match.group("number"))
    return None


def _commercial_text(ad_input: dict[str, Any]) -> str:
    """이미지 분석값을 제외하고 사용자가 제공한 상품·캠페인 텍스트만 모은다."""

    values: list[str] = []
    product = ad_input.get("product", {})
    campaign = ad_input.get("campaign", {})
    for value in (
        product.get("name"),
        product.get("features"),
        campaign.get("focus_feature"),
        campaign.get("copy_request"),
        campaign.get("promotion"),
        campaign.get("benefit"),
    ):
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return " ".join(values)


def _commercial_quantity_numbers(ad_input: dict[str, Any]) -> set[int]:
    text = _commercial_text(ad_input)
    numbers = {
        int(match.group("number"))
        if match.group("number").isdigit()
        else _KOREAN_COUNT_NUMBERS[match.group("number")]
        for match in _COUNT_TOKEN_PATTERN.finditer(text)
    }
    for match in _PROMOTION_QUANTITY_PATTERN.finditer(text):
        numbers.add(int(match.group(1)) + int(match.group(2)))
    return numbers


def _visual_count_is_commercially_grounded(
    ad_input: dict[str, Any],
    value: str,
) -> bool:
    number = _count_number(value)
    return bool(number and number in _commercial_quantity_numbers(ad_input))


def _headline_count_numbers(headline: str) -> set[int]:
    values: set[int] = set()
    for match in _COUNT_TOKEN_PATTERN.finditer(headline):
        token = match.group("number")
        values.add(int(token) if token.isdigit() else _KOREAN_COUNT_NUMBERS[token])
    return values


def _promotion_quantities(text: str) -> set[tuple[int, int]]:
    return {
        (int(match.group(1)), int(match.group(2)))
        for match in _PROMOTION_QUANTITY_PATTERN.finditer(text)
    }


def headline_uses_ungrounded_quantity(
    headline: str,
    ad_input: dict[str, Any],
) -> bool:
    used = _headline_count_numbers(headline)
    commercial = _commercial_quantity_numbers(ad_input)
    used_promotions = _promotion_quantities(headline)
    commercial_promotions = _promotion_quantities(_commercial_text(ad_input))
    return bool(
        (used - commercial)
        or (used_promotions - commercial_promotions)
    )


def headline_uses_ungrounded_visual_count(
    headline: str,
    ad_input: dict[str, Any],
) -> bool:
    """이전 보고서 필드와의 호환을 위한 수량 검증 별칭."""

    return headline_uses_ungrounded_quantity(headline, ad_input)


_BENEFIT_PATTERN = re.compile(
    r"(?:\d[\d,]*\s*(?:%|원)\s*(?:할인|혜택)?|\d+\s*\+\s*\d+|"
    r"할인|증정|무료|기간\s*한정|시즌\s*한정|한정\s*판매|저당|제로\s*슈거|"
    r"무카페인|디카페인|세트\s*구성|구독\s*혜택)"
)
_TONE_MESSAGE_LABELS = {
    "따뜻한": "따뜻한 분위기",
    "친근한": "친근한 분위기",
    "발랄한": "발랄한 분위기",
    "차분한": "차분한 분위기",
    "활기찬": "활기찬 분위기",
    "세련된": "세련된 분위기",
    "신뢰감 있는": "신뢰감 있는 인상",
    "전문적인": "전문적인 인상",
    "자연스러운": "자연스러운 분위기",
}
_STYLE_MESSAGE_LABELS = {
    "감성적": "감성적인 분위기",
    "고급스러운": "고급스러운 분위기",
    "유머러스": "유쾌한 분위기",
    "트렌디한": "트렌디한 분위기",
    "강렬한": "강렬한 인상",
}


def _benefit_messages(ad_input: dict[str, Any]) -> list[str]:
    text = _commercial_text(ad_input)
    values: list[str] = []
    for match in _BENEFIT_PATTERN.finditer(text):
        start = max(0, match.start() - 18)
        end = min(len(text), match.end() + 10)
        phrase = re.split(r"[,。.\n]", text[start:end])[-1]
        phrase = re.sub(
            r"(?:강조|표현|작성|알려|보여)\S*.*$",
            "",
            phrase,
        ).strip(" ,.! ")
        if phrase and _normalize_compact(phrase) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(phrase)
    return values


def _product_type_repeats_menu_name(
    ad_input: dict[str, Any],
    product_type: str,
) -> bool:
    """상품명에 이미 포함된 시각 제품 종류인지 일반 규칙으로 판정한다."""

    menu_name = str(ad_input.get("product", {}).get("name") or "")
    normalized_menu = _normalize_compact(menu_name)
    normalized_type = _normalize_compact(product_type)
    return bool(
        normalized_menu
        and len(normalized_type) >= 2
        and normalized_type in normalized_menu
    )


def build_message_inventory(ad_input: dict[str, Any]) -> list[dict[str, Any]]:
    """입력 근거를 상품·감성·상황·혜택 메시지 풀로 정리한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    dimensions = parse_request_dimensions(profile.copy_request)
    values: list[dict[str, Any]] = []

    def add(
        message_type: str,
        value: str,
        source: str,
        priority: int,
        *,
        subtype: str | None = None,
    ) -> None:
        cleaned = " ".join(value.strip(" ,.! ").split())
        if not cleaned:
            return
        signature = (message_type, _normalize_compact(cleaned))
        if any(
            (item["type"], _normalize_compact(str(item["value"]))) == signature
            for item in values
        ):
            return
        values.append({
            "id": f"m{len(values) + 1}",
            "type": message_type,
            "subtype": subtype,
            "value": cleaned,
            "source": source,
            "priority": priority,
        })

    for phrase in _benefit_messages(ad_input):
        add("benefit", phrase, "explicit_commercial_input", 110)

    contexts = {str(item) for item in dimensions["usage_context"]}
    emotions = {str(item) for item in dimensions["emotional_message"]}
    for phrase in ad_value_base._campaign_phrases(profile.copy_request):
        if _BENEFIT_PATTERN.search(phrase):
            add("benefit", phrase, "copy_request", 120)
        elif any(_normalize_compact(item) in _normalize_compact(phrase) for item in contexts):
            add("situation", phrase, "copy_request", 120)
        else:
            add("emotion", phrase, "copy_request", 120)
    for value in contexts:
        add("situation", value, "copy_request", 100)
    for value in emotions:
        add("emotion", value, "copy_request", 100)

    product_fact = assigned_subcopy_fact(ad_input)
    if product_fact:
        message_type = "benefit" if _BENEFIT_PATTERN.search(product_fact) else "product"
        priority = 105 if message_type == "benefit" else 90
        add(message_type, product_fact, "verified_product_facts", priority)

    for tone in dimensions["tone"]:
        if tone in _TONE_MESSAGE_LABELS:
            add("emotion", _TONE_MESSAGE_LABELS[tone], "requested_tone", 70)
    for style in dimensions["style"]:
        if style in _STYLE_MESSAGE_LABELS:
            add("emotion", _STYLE_MESSAGE_LABELS[style], "requested_style", 65)

    for fact in _eligible_visual_facts(ad_input):
        value = str(fact.get("value") or "").strip()
        category = str(fact.get("category") or "")
        if (
            category == "product_type"
            and _product_type_repeats_menu_name(ad_input, value)
        ):
            continue
        if value:
            add(
                "product",
                value,
                "verified_visual_facts",
                55,
                subtype=category,
            )
    return values


def assign_message_roles(
    ad_input: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """가장 중요한 메시지와 가장 보완적인 두 번째 메시지를 고른다."""

    inventory = build_message_inventory(ad_input)
    if not inventory:
        return None, None, inventory
    primary = max(inventory, key=lambda item: int(item["priority"]))
    remaining = [item for item in inventory if item["id"] != primary["id"]]
    if not remaining:
        return primary, None, inventory

    def complement_score(item: dict[str, Any]) -> tuple[int, int]:
        score = int(item["priority"])
        if item["type"] != primary["type"]:
            score += 25
        if item["source"] != primary["source"]:
            score += 15
        if primary["type"] == "product" and item["type"] in {
            "emotion", "situation", "benefit",
        }:
            score += 35
        if primary["type"] != "product" and item["type"] == "product":
            score += 45
        return score, int(item["priority"])

    secondary = max(remaining, key=complement_score)
    return primary, secondary, inventory


def _message_is_reflected(
    text: str,
    message: dict[str, Any],
    ad_input: dict[str, Any],
) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    value = str(message["value"])
    if message["source"] == "verified_product_facts":
        profile = analyze_copy_input(ad_input, image_provided=False)
        return bool(profile.verified_facts) and all(
            base._fact_is_covered_strict(cleaned, fact)
            for fact in profile.verified_facts
        )
    if message["source"] == "verified_visual_facts":
        return ad_value_base._fact_is_reflected(
            cleaned,
            value,
            str(message.get("subtype") or ""),
        )
    message_terms = [
        term for term in ad_value_base._terms(value)
        if len(term) >= 2
        and _normalize_compact(term) not in ad_value_base._GENERIC_FACT_TERMS
    ]
    compact = _normalize_compact(cleaned)
    return bool(message_terms) and all(
        _normalize_compact(term) in compact for term in message_terms
    )


def _same_claim_repeated(copy: AdCopy, ad_input: dict[str, Any]) -> bool:
    if not copy["headline"].strip() or not copy["body"].strip():
        return False
    _, _, inventory = assign_message_roles(ad_input)
    if any(
        _message_is_reflected(copy["headline"], message, ad_input)
        and _message_is_reflected(copy["body"], message, ad_input)
        for message in inventory
    ):
        return True
    profile = analyze_copy_input(ad_input, image_provided=False)
    return any(
        fact_reused_in_headline(copy["headline"], fact)
        and base._fact_is_covered_strict(copy["body"], fact)
        for fact in profile.verified_facts
    )


def _fact_terms(fact: str) -> list[str]:
    return [
        term
        for term in ad_value_base._terms(fact)
        if len(term) >= 2
        and _normalize_compact(term) not in ad_value_base._GENERIC_FACT_TERMS
        and not any(root in _normalize_compact(term) for root in _SENSORY_ROOTS)
    ]


def _modifier_roots(fact: str) -> set[str]:
    roots = {
        match.group(1)
        for match in re.finditer(r"([가-힣]{2,}?)(?:한|함)(?:\s|$)", fact)
    }
    return {_normalize_compact(root) for root in roots if len(root) >= 2}


def fact_reused_in_headline(headline: str, fact: str) -> bool:
    """감각·관계 범주를 보존한 채 표현만 바꾼 제목의 사실 재사용을 찾는다."""

    title_compact = _normalize_compact(headline)
    fact_compact = _normalize_compact(fact)
    if not title_compact or not fact_compact:
        return False
    terms = _fact_terms(fact)
    matched_terms = [
        term for term in terms if _normalize_compact(term) in title_compact
    ]

    if any(root in fact_compact for root in _RELATION_ROOTS):
        return len(matched_terms) >= min(2, len(terms)) and len(matched_terms) >= 2

    sensory_roots = [root for root in _SENSORY_ROOTS if root in fact_compact]
    if sensory_roots:
        if not matched_terms:
            return False
        if any(root in title_compact for root in sensory_roots):
            return True
        if any(root in title_compact for root in _EXPERIENCE_ROOTS):
            return True
        modifiers = _modifier_roots(fact)
        if modifiers and any(root in title_compact for root in modifiers):
            return True
        return False

    return base._fact_is_covered_strict(headline, fact)


def has_assigned_fact_repetition(copy: AdCopy, ad_input: dict[str, Any]) -> bool:
    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.generation_mode != "headline_and_subcopy" or not copy["body"]:
        return False
    return any(
        fact_reused_in_headline(copy["headline"], fact)
        and base._fact_is_covered_strict(copy["body"], fact)
        for fact in profile.verified_facts
    )


def _eligible_visual_facts(ad_input: dict[str, Any]) -> list[dict[str, str]]:
    facts = [dict(fact) for fact in ad_value_base._visual_facts(ad_input)]
    eligible: list[dict[str, str]] = []
    component_terms: set[str] = set()
    for fact in facts:
        category = str(fact.get("category") or "")
        value = str(fact.get("value") or "").strip()
        normalized = _normalize_compact(value)
        if category == "count":
            if _visual_count_is_commercially_grounded(ad_input, value):
                eligible.append(fact)
        elif category == "visible_component":
            if any(
                root in normalized
                for root in _LOW_VALUE_VISUAL_COMPONENT_ROOTS
            ):
                continue
            terms = {
                _normalize_compact(term)
                for term in ad_value_base._terms(value)
                if len(term) >= 2
            }
            meaningful = {
                term for term in terms
                if term not in {_normalize_compact(item) for item in _GENERIC_VISUAL_COMPONENTS}
            }
            if meaningful:
                eligible.append(fact)
                component_terms.update(meaningful)
        elif category == "packaging" and normalized:
            eligible.append(fact)
        elif category == "product_type" and normalized not in {"음료", "커피", "디저트", "빵"}:
            eligible.append(fact)

    for fact in facts:
        if fact in eligible or str(fact.get("category") or "") != "arrangement":
            continue
        normalized = _normalize_compact(str(fact.get("value") or ""))
        if any(term in normalized for term in component_terms):
            eligible.append(fact)
    return eligible


def _title_reflects_eligible_visual(title: str, ad_input: dict[str, Any]) -> bool:
    return any(
        ad_value_base._fact_is_reflected(
            title,
            str(fact.get("value") or ""),
            str(fact.get("category") or ""),
        )
        for fact in _eligible_visual_facts(ad_input)
    )


def _campaign_title_candidates(ad_input: dict[str, Any]) -> list[str]:
    profile = analyze_copy_input(ad_input, image_provided=False)
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.strip(" ,.!").split())
        if (
            4 <= len(cleaned) <= profile.limits.headline
            and _normalize_compact(cleaned) not in {
                _normalize_compact(item) for item in values
            }
        ):
            values.append(cleaned)

    for phrase in ad_value_base._campaign_phrases(profile.copy_request):
        add(phrase)
    dimensions = parse_request_dimensions(profile.copy_request)
    contexts = [str(item) for item in dimensions["usage_context"]]
    emotions = [str(item) for item in dimensions["emotional_message"]]
    if contexts and emotions:
        add(f"{contexts[0]}의 {emotions[0]}")
        add(f"{emotions[0]}이 필요한 {contexts[0]}")
    return values


def eligible_headline_candidates(ad_input: dict[str, Any]) -> list[str]:
    """보조 문구의 상품 사실과 겹치지 않는 콘셉트·시각 제목 후보를 만든다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.split())
        if not 4 <= len(cleaned) <= profile.limits.headline:
            return
        if headline_uses_ungrounded_quantity(cleaned, ad_input):
            return
        if is_truncated_headline(cleaned, ad_input):
            return
        if any(fact_reused_in_headline(cleaned, fact) for fact in profile.verified_facts):
            return
        if _normalize_compact(cleaned) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(cleaned)

    for value in _campaign_title_candidates(ad_input):
        add(value)
    for value in ad_value_base._meaningful_visual_titles(
        ad_input,
        profile.limits.headline,
    ):
        if _title_reflects_eligible_visual(value, ad_input):
            add(value)
    for fact in _eligible_visual_facts(ad_input):
        category = str(fact.get("category") or "")
        value = str(fact.get("value") or "").strip()
        if (
            category in {"visible_component", "packaging"}
            and value
            and not any(root in _normalize_compact(value) for root in _SENSORY_ROOTS)
        ):
            add(value)
    return values


def explicit_cta_requested(ad_input: dict[str, Any]) -> bool:
    """사용자가 판매·방문·행동 유도형 보조 문구를 명시했는지 확인한다."""

    request = str(ad_input.get("campaign", {}).get("copy_request") or "").strip()
    return bool(
        request
        and _CTA_REQUEST_PATTERN.search(request)
        and not _CTA_DENIAL_PATTERN.search(request)
    )


def classify_subcopy_form(
    text: str,
    ad_input: dict[str, Any] | None = None,
) -> str | None:
    """보조 문구가 의미상 완결된 허용 형식인지 분류한다."""

    original = " ".join(str(text or "").split())
    cleaned = original.strip(" ,.!?·/|")
    if len(cleaned) < 4:
        return None
    if previous.is_descriptive_subcopy(original):
        return "descriptive_sentence"
    if _GENERAL_DESCRIPTIVE_END_PATTERN.search(original):
        return "descriptive_sentence"
    if _CTA_END_PATTERN.search(original):
        if ad_input is not None and explicit_cta_requested(ad_input):
            return "explicit_cta"
        return None
    if (
        _INCOMPLETE_SUBCOPY_END_PATTERN.search(cleaned)
        or _STANDALONE_PARTICLE_END_PATTERN.search(cleaned)
    ):
        return None

    items = [
        item.strip()
        for item in _INFORMATION_LIST_SEPARATOR_PATTERN.split(cleaned)
        if item.strip()
    ]
    if len(items) >= 2:
        if all(
            len(item) >= 2
            and not _INCOMPLETE_SUBCOPY_END_PATTERN.search(item)
            and not _STANDALONE_PARTICLE_END_PATTERN.search(item)
            for item in items
        ):
            return "information_list"
        return None
    return "complete_noun_phrase"


def is_complete_subcopy(
    text: str,
    ad_input: dict[str, Any] | None = None,
) -> bool:
    return classify_subcopy_form(text, ad_input) is not None


def _message_requires_relational_sentence(message: dict[str, Any] | None) -> bool:
    if not message:
        return False
    value = _normalize_compact(str(message.get("value") or ""))
    return bool(
        any(root in value for root in _RELATION_ROOTS)
        or re.search(r"(?:겉|밖|외부).*(?:속|안|내부)", value)
        or re.search(r"(?:속|안|내부).*(?:겉|밖|외부)", value)
        or re.search(r"(?:하고|하며|하면서|반면)", value)
    )


def _message_has_parallel_information(
    message: dict[str, Any] | None,
    ad_input: dict[str, Any],
) -> bool:
    if not message:
        return False
    if message.get("source") == "verified_product_facts":
        profile = analyze_copy_input(ad_input, image_provided=False)
        if len(profile.verified_facts) >= 2:
            return True
    value = str(message.get("value") or "")
    return len([
        item for item in _INFORMATION_LIST_SEPARATOR_PATTERN.split(value)
        if item.strip()
    ]) >= 2


def eligible_subcopy_forms(ad_input: dict[str, Any]) -> tuple[str, ...]:
    """배정 메시지에 안전하게 사용할 수 있는 보조 문구 형식을 반환한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.generation_mode != "headline_and_subcopy":
        return ()
    _, secondary, _ = assign_message_roles(ad_input)
    if secondary is None:
        return ()
    forms = ["descriptive_sentence", "complete_noun_phrase"]
    if _message_has_parallel_information(secondary, ad_input):
        forms.append("information_list")
    if explicit_cta_requested(ad_input):
        forms.append("explicit_cta")
    return tuple(forms)


def subcopy_form_fitness(
    text: str,
    ad_input: dict[str, Any],
) -> tuple[str | None, int]:
    """메시지 구조와 보조 문구 형식의 적합도를 계산한다."""

    form = classify_subcopy_form(text, ad_input)
    if form is None:
        return None, 0
    _, secondary, _ = assign_message_roles(ad_input)
    relation_required = _message_requires_relational_sentence(secondary)
    parallel_allowed = _message_has_parallel_information(secondary, ad_input)
    if form == "explicit_cta":
        return form, 10 if explicit_cta_requested(ad_input) else 0
    if relation_required:
        return form, {
            "descriptive_sentence": 10,
            "complete_noun_phrase": 5,
            "information_list": 2 if parallel_allowed else 0,
        }.get(form, 0)
    return form, {
        "complete_noun_phrase": 10,
        "descriptive_sentence": 7,
        "information_list": 10 if parallel_allowed else 0,
    }.get(form, 0)


def _subcopy_pair_is_diverse(
    left: AdCopy,
    right: AdCopy,
    ad_input: dict[str, Any],
) -> bool:
    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.generation_mode != "headline_and_subcopy":
        return True
    if _normalize_compact(left["body"]) == _normalize_compact(right["body"]):
        return False
    return True


# 이전 버전의 외부 호출 이름은 유지하되 판정 범위를 허용 형식 전체로 넓힌다.
is_descriptive_subcopy = is_complete_subcopy


def assigned_subcopy_fact(ad_input: dict[str, Any]) -> str:
    """보조 문구에 배정할 상품 특징을 복합 표현 그대로 보존한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    raw_feature = str(ad_input.get("product", {}).get("features") or "").strip()
    raw_feature = re.sub(r"\s+", " ", raw_feature).strip(" ,.;")
    if len(profile.verified_facts) > 1 and raw_feature:
        return raw_feature
    if profile.verified_facts:
        return profile.verified_facts[0].strip()
    return raw_feature


def _descriptive_fallback_subcopy(ad_input: dict[str, Any]) -> str:
    """상품 특징 전체를 자르지 않고 의미가 완결된 안전 문구로 만든다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    fact = assigned_subcopy_fact(ad_input).rstrip(" .")
    if not fact:
        return ""
    value = fact if is_complete_subcopy(fact, ad_input) else (
        f"{fact}{ad_value_base._subject_particle(fact)} 특징이에요."
    )
    limit = min(profile.limits.subcopy, SUBCOPY_HARD_MAX)
    if len(value) > limit or not is_complete_subcopy(value, ad_input):
        return ""
    if not all(base._fact_is_covered_strict(value, item) for item in profile.verified_facts):
        return ""
    return value


def _message_fallback_subcopy(
    message: dict[str, Any],
    ad_input: dict[str, Any],
) -> str:
    if message["source"] == "verified_product_facts":
        return _descriptive_fallback_subcopy(ad_input)
    value = str(message["value"]).rstrip(" .")
    if not value:
        return ""
    if is_complete_subcopy(value, ad_input):
        body = value
    elif message["type"] == "benefit":
        body = f"{value}{ad_value_base._subject_particle(value)} 적용되는 혜택이에요."
    elif message["source"] == "verified_visual_facts":
        body = f"{value}{ad_value_base._subject_particle(value)} 돋보이는 구성이에요."
    else:
        body = f"{value}{ad_value_base._and_particle(value)} 어울리는 메뉴예요."
    profile = analyze_copy_input(ad_input, image_provided=False)
    limit = min(profile.limits.subcopy, SUBCOPY_HARD_MAX)
    if len(body) > limit or not is_complete_subcopy(body, ad_input):
        return ""
    return body


def _message_fallback_subcopies(
    message: dict[str, Any],
    ad_input: dict[str, Any],
) -> list[str]:
    """서로 다른 안전 형식의 fallback 보조 문구를 만든다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    limit = min(profile.limits.subcopy, SUBCOPY_HARD_MAX)
    value = str(message.get("value") or "").rstrip(" .")
    candidates: list[str] = []

    def add(candidate: str) -> None:
        cleaned = " ".join(candidate.split())
        if not cleaned or len(cleaned) > limit:
            return
        if not is_complete_subcopy(cleaned, ad_input):
            return
        if not _message_is_reflected(cleaned, message, ad_input):
            return
        if _normalize_compact(cleaned) not in {
            _normalize_compact(item) for item in candidates
        }:
            candidates.append(cleaned)

    add(_message_fallback_subcopy(message, ad_input))
    if message.get("source") == "verified_product_facts":
        fact = assigned_subcopy_fact(ad_input).rstrip(" .")
        add(fact)
        add(f"{fact}{ad_value_base._subject_particle(fact)} 특징이에요.")
    elif message.get("type") == "benefit":
        add(value)
        add(f"{value}{ad_value_base._subject_particle(value)} 적용되는 혜택이에요.")
    elif message.get("source") == "verified_visual_facts":
        add(value)
        add(
            f"{value}{ad_value_base._subject_particle(value)} "
            "돋보이는 구성"
        )
        add(f"{value}{ad_value_base._subject_particle(value)} 돋보이는 구성이에요.")
    else:
        add(value)
        add(f"{value}{ad_value_base._and_particle(value)} 어울리는 메뉴예요.")
    if explicit_cta_requested(ad_input):
        add(f"{value}{ad_value_base._object_particle(value)} 만나보세요.")

    _, secondary, _ = assign_message_roles(ad_input)
    if _message_requires_relational_sentence(secondary):
        order = {
            "descriptive_sentence": 0,
            "complete_noun_phrase": 1,
            "information_list": 2,
            "explicit_cta": 3,
        }
    else:
        order = {
            "complete_noun_phrase": 0,
            "descriptive_sentence": 1,
            "information_list": 2,
            "explicit_cta": 3,
        }
    return sorted(
        candidates,
        key=lambda item: order.get(classify_subcopy_form(item, ad_input) or "", 9),
    )


def is_truncated_headline(
    text: str,
    ad_input: dict[str, Any] | None = None,
) -> bool:
    """길이 제한 때문에 단어나 수식 구문이 중간에서 잘린 제목을 찾는다."""

    cleaned = text.strip(" .,!?")
    if not cleaned:
        return False
    if _INCOMPLETE_HEADLINE_END_PATTERN.search(cleaned):
        return True
    if _MALFORMED_MODIFIER_PARTICLE_PATTERN.search(cleaned):
        return True
    words = re.findall(r"[0-9A-Za-z가-힣]+", cleaned)
    if not words:
        return False
    last = words[-1]
    if (
        len(last) == 1
        and bool(re.fullmatch(r"[가-힣]", last))
        and last not in _VALID_SINGLE_SYLLABLE_HEADLINE_ENDINGS
    ):
        return True
    if len(last) >= 2 and any(
        noun != last and noun.startswith(last) for noun in _COMMON_HEADLINE_NOUNS
    ):
        return True
    if ad_input is None or len(last) < 3:
        return False
    source_values = [
        str(ad_input.get("product", {}).get("features") or ""),
        *[
            str(fact.get("value") or "")
            for fact in ad_value_base._visual_facts(ad_input)
        ],
    ]
    source_words = {
        word
        for value in source_values
        for word in re.findall(r"[0-9A-Za-z가-힣]+", value)
    }
    truncation_suffixes = ("운", "한", "하는", "인", "적인", "스러운", "로운")
    return any(
        source_word != last
        and source_word.startswith(last)
        and source_word[len(last):].startswith(truncation_suffixes)
        for source_word in source_words
    )


def _prepare_effective_input(
    ad_input: dict[str, Any],
) -> tuple[dict[str, Any], bool, str | None, list[str]]:
    prepared = deepcopy(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    primary, secondary, inventory = assign_message_roles(prepared)
    candidates = [
        str(item["value"]) for item in inventory
        if item["id"] != (secondary or {}).get("id")
    ]
    if profile.generation_mode != "headline_and_subcopy":
        return prepared, False, None, candidates
    safe_subcopy = (
        _message_fallback_subcopy(secondary, prepared)
        if secondary is not None else ""
    )
    same_claim = False
    if primary is not None and secondary is not None and safe_subcopy:
        title_candidates = _message_title_candidates(primary, prepared)
        probe_title = (
            title_candidates[0]
            if title_candidates else str(primary.get("value") or "")
        )
        same_claim = _same_claim_repeated(
            {"headline": probe_title, "body": safe_subcopy, "hashtags": []},
            prepared,
        )
    if (
        primary is not None
        and secondary is not None
        and safe_subcopy
        and not same_claim
    ):
        return prepared, False, None, candidates

    prepared.setdefault("campaign", {})["generation_mode"] = "headline_only"
    if secondary is None:
        reason = (
            "제목과 다른 정보를 더할 두 번째 근거 메시지가 없어 "
            "제목만 모드로 자동 전환했습니다."
        )
    elif not safe_subcopy:
        reason = (
            "두 번째 근거 메시지를 글자 수 안에서 완결된 보조 문구로 "
            "안전하게 설명할 수 없어 제목만 모드로 자동 전환했습니다."
        )
    elif same_claim:
        reason = (
            "두 번째 근거 메시지가 제목의 주 메시지와 같은 주장을 반복해 "
            "제목만 모드로 자동 전환했습니다."
        )
    else:
        reason = "메시지 역할을 안전하게 배정할 수 없어 제목만 모드로 자동 전환했습니다."
    return prepared, True, reason, candidates


def _headline_has_separate_source(title: str, ad_input: dict[str, Any]) -> bool:
    profile = analyze_copy_input(ad_input, image_provided=False)
    compact = _normalize_compact(title)
    campaign = any(
        _normalize_compact(phrase) in compact
        for phrase in ad_value_base._campaign_phrases(profile.copy_request)
    )
    return campaign or _title_reflects_eligible_visual(title, ad_input)


def _deduplicate_issues(issues: list[CopyValidationIssue]) -> list[CopyValidationIssue]:
    result: list[CopyValidationIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for issue in issues:
        key = (issue.code, issue.message, issue.option_index, issue.field)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    effective_input, _, _, _ = _prepare_effective_input(ad_input)
    profile = analyze_copy_input(effective_input, image_provided=False)
    primary, secondary, _ = assign_message_roles(effective_input)
    inherited = list(previous.collect_banner_copy_issues(copies, effective_input))
    issues: list[CopyValidationIssue] = []
    for issue in inherited:
        option = (
            copies[issue.option_index - 1]
            if issue.option_index is not None
            and 1 <= issue.option_index <= len(copies)
            else None
        )
        if (
            option is not None
            and issue.code == "SUBCOPY_NOT_DESCRIPTIVE"
            and is_complete_subcopy(option["body"], effective_input)
        ):
            continue
        if (
            option is not None
            and issue.code in {"SUBCOPY_CTA", "IMPERATIVE_CTA"}
            and issue.field == "subcopy"
            and explicit_cta_requested(effective_input)
            and classify_subcopy_form(option["body"], effective_input) == "explicit_cta"
        ):
            continue
        if option is not None and primary is not None and secondary is not None:
            headline_has_primary = _message_is_reflected(
                option["headline"], primary, effective_input
            )
            body_has_secondary = _message_is_reflected(
                option["body"], secondary, effective_input
            )
            if (
                issue.code == "SUBCOPY_MISSING_FACTS"
                and primary["source"] == "verified_product_facts"
                and headline_has_primary
                and body_has_secondary
            ):
                continue
            if (
                issue.code == "UNVERIFIED_SUBCOPY_EXPRESSION"
                and secondary["type"] in {"emotion", "situation", "benefit"}
                and body_has_secondary
            ):
                continue
            if (
                issue.code == "DISALLOWED_SUBCOPY_CONTENT"
                and secondary["type"] == "benefit"
                and body_has_secondary
            ):
                continue
            if (
                issue.code == "DISALLOWED_HEADLINE_CONTENT"
                and primary["type"] == "benefit"
                and headline_has_primary
            ):
                continue
            if (
                issue.code == "HEADLINE_NOT_GROUNDED"
                and headline_has_primary
            ):
                continue
        issues.append(issue)

    for index, copy in enumerate(copies, start=1):
        if is_truncated_headline(copy["headline"], effective_input):
            issues.append(CopyValidationIssue(
                "TRUNCATED_HEADLINE",
                f"선택지 {index} 제목이 단어나 수식 구문 중간에서 잘렸습니다.",
                index,
                "headline",
            ))
        if headline_uses_ungrounded_quantity(copy["headline"], effective_input):
            issues.append(CopyValidationIssue(
                "UNGROUNDED_QUANTITY_TITLE",
                (
                    f"선택지 {index} 제목이 사용자 입력의 판매 수량·세트·라인업·"
                    "포장 단위·프로모션 근거 없이 수량 표현을 사용합니다."
                ),
                index,
                "headline",
            ))

    if profile.generation_mode != "headline_and_subcopy":
        return _deduplicate_issues(issues)

    for index, copy in enumerate(copies, start=1):
        body_form = classify_subcopy_form(copy["body"], effective_input)
        if copy["body"] and body_form is None:
            issues.append(CopyValidationIssue(
                "SUBCOPY_NOT_COMPLETE",
                (
                    f"선택지 {index} 보조 문구가 허용된 설명 문장·완결 명사구·"
                    "정보 나열 형식이 아니거나 끝맺음이 불완전합니다."
                ),
                index,
                "subcopy",
            ))
        elif (
            copy["body"]
            and body_form not in eligible_subcopy_forms(effective_input)
        ):
            issues.append(CopyValidationIssue(
                "SUBCOPY_FORM_NOT_ELIGIBLE",
                (
                    f"선택지 {index} 보조 문구 형식 {body_form}은 배정 메시지 구조에 "
                    "적합하지 않습니다."
                ),
                index,
                "subcopy",
            ))
        if primary is None or not _message_is_reflected(
            copy["headline"], primary, effective_input
        ):
            issues.append(CopyValidationIssue(
                "HEADLINE_MISSING_PRIMARY_MESSAGE",
                f"선택지 {index} 제목에 배정된 주 메시지가 반영되지 않았습니다.",
                index,
                "headline",
            ))
        if secondary is None or not _message_is_reflected(
            copy["body"], secondary, effective_input
        ):
            issues.append(CopyValidationIssue(
                "SUBCOPY_MISSING_SECONDARY_MESSAGE",
                f"선택지 {index} 보조 문구에 배정된 두 번째 메시지가 반영되지 않았습니다.",
                index,
                "subcopy",
            ))
        missing_pair_facts = [
            fact for fact in profile.verified_facts
            if not base._fact_is_covered_strict(
                f"{copy['headline']} {copy['body']}",
                fact,
            )
        ]
        if missing_pair_facts:
            issues.append(CopyValidationIssue(
                "PAIR_MISSING_PRODUCT_FACTS",
                f"선택지 {index}의 제목과 보조 문구에 확인된 상품 특징이 빠졌습니다: {', '.join(missing_pair_facts)}",
                index,
            ))
        if _same_claim_repeated(copy, effective_input):
            issues.append(CopyValidationIssue(
                "SAME_CLAIM_REPEATED",
                f"선택지 {index}의 제목과 보조 문구가 같은 주장을 반복합니다.",
                index,
            ))
    return _deduplicate_issues(issues)


def _candidate_hard_issues(
    copy: AdCopy,
    ad_input: dict[str, Any],
    *,
    candidate_index: int,
) -> list[CopyValidationIssue]:
    issues = collect_banner_copy_issues([copy, dict(copy)], ad_input)
    return [
        CopyValidationIssue(
            issue.code,
            issue.message.replace("선택지 1", f"내부 후보 {candidate_index}"),
            candidate_index,
            issue.field,
        )
        for issue in issues
        if issue.option_index == 1
    ]


def score_ad_candidate(
    copy: AdCopy,
    ad_input: dict[str, Any],
) -> tuple[int, int, int, tuple[str, ...], dict[str, int]]:
    score, headline_score, subcopy_score, flags, base_breakdown = (
        previous.score_ad_candidate(copy, ad_input)
    )
    breakdown = dict(base_breakdown)
    profile = analyze_copy_input(ad_input, image_provided=False)
    primary, secondary, _ = assign_message_roles(ad_input)
    if (
        profile.generation_mode == "headline_and_subcopy"
        and copy["body"].strip()
        and is_complete_subcopy(copy["body"], ad_input)
        and "subcopy_descriptive_sentence" not in breakdown
    ):
        breakdown["subcopy_complete_allowed_form"] = 12
        subcopy_score += 12
    if profile.generation_mode == "headline_and_subcopy" and copy["body"].strip():
        form, fitness = subcopy_form_fitness(copy["body"], ad_input)
        if form is not None and fitness:
            breakdown["subcopy_form_fitness"] = fitness
            subcopy_score += fitness
    if (
        profile.generation_mode == "headline_and_subcopy"
        and primary is not None
        and secondary is not None
        and _message_is_reflected(copy["headline"], primary, ad_input)
        and _message_is_reflected(copy["body"], secondary, ad_input)
        and not _same_claim_repeated(copy, ad_input)
    ):
        breakdown["message_role_complementarity"] = 16
        score += 16
        headline_score += 16
    return sum(breakdown.values()), headline_score, subcopy_score, flags, breakdown


def _rank_candidates(
    candidates: list[AdCopy],
    ad_input: dict[str, Any],
    *,
    attempt: int,
) -> tuple[list[previous.previous.RankedCandidate], list[dict[str, Any]]]:
    valid: list[previous.previous.RankedCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for index, copy in enumerate(candidates, start=1):
        hard_issues = _candidate_hard_issues(copy, ad_input, candidate_index=index)
        score, headline_score, subcopy_score, flags, breakdown = score_ad_candidate(
            copy,
            ad_input,
        )
        ranking_issues: list[str] = []
        if not hard_issues and headline_score < MIN_HEADLINE_AD_SCORE:
            ranking_issues.append(
                f"제목 광고성 점수 {headline_score}점이 최소 {MIN_HEADLINE_AD_SCORE}점보다 낮습니다."
            )
        if not hard_issues and score < MIN_AD_SCORE:
            ranking_issues.append(
                f"전체 광고성 점수 {score}점이 최소 {MIN_AD_SCORE}점보다 낮습니다."
            )
        diagnostics.append({
            "candidate_index": index,
            "headline": copy["headline"],
            "subcopy": copy["body"],
            "subcopy_form": classify_subcopy_form(copy["body"], ad_input),
            "hard_issues": [issue.message for issue in hard_issues],
            "ranking_issues": ranking_issues,
            "ad_score": score,
            "headline_ad_score": headline_score,
            "subcopy_score": subcopy_score,
            "quality_flags": list(flags),
            "score_breakdown": breakdown,
        })
        if not hard_issues:
            valid.append(previous.previous.RankedCandidate(
                copy=dict(copy),
                candidate_index=index,
                attempt=attempt,
                score=score,
                headline_score=headline_score,
                subcopy_score=subcopy_score,
                flags=flags,
                score_breakdown=breakdown,
            ))
    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.generation_mode == "headline_and_subcopy" and valid:
        unique_bodies = {
            _normalize_compact(item.copy["body"])
            for item in valid
        }
        diversity_messages: list[str] = []
        if len(unique_bodies) < 2:
            diversity_messages.append(
                "내부 후보의 보조 문구 전체가 동일해 최종 두 선택지를 구성할 수 없습니다."
            )
        if diversity_messages:
            for item in diagnostics:
                if not item["hard_issues"]:
                    item["ranking_issues"].extend(diversity_messages)
    return valid, diagnostics


def select_top_two_candidates(
    candidates: list[previous.previous.RankedCandidate],
    ad_input: dict[str, Any],
) -> tuple[list[AdCopy], list[dict[str, Any]]]:
    """제목과 보조 문구가 모두 다른 최종 두 선택지를 고른다."""

    qualified = [candidate for candidate in candidates if candidate.qualifies]
    best: tuple[
        int,
        previous.previous.RankedCandidate,
        previous.previous.RankedCandidate,
        float,
        int,
        int,
    ] | None = None
    profile = analyze_copy_input(ad_input, image_provided=False)
    for left, right in combinations(qualified, 2):
        if _normalize_compact(left.copy["headline"]) == _normalize_compact(
            right.copy["headline"]
        ):
            continue
        if not _subcopy_pair_is_diverse(left.copy, right.copy, ad_input):
            continue
        similarity = title_lexical_similarity(
            left.copy["headline"], right.copy["headline"]
        )
        headline_adjustment = 8
        if similarity >= 0.85:
            headline_adjustment = -24
        elif similarity >= 0.5:
            headline_adjustment = -10
        elif ad_value_base._has_visual_angle(
            left.copy["headline"], ad_input
        ) != ad_value_base._has_visual_angle(right.copy["headline"], ad_input):
            headline_adjustment = 12

        subcopy_adjustment = 0
        if profile.generation_mode == "headline_and_subcopy":
            left_form = classify_subcopy_form(left.copy["body"], ad_input)
            right_form = classify_subcopy_form(right.copy["body"], ad_input)
            subcopy_adjustment = 18 if left_form != right_form else 0
        pair_score = (
            left.score + right.score + headline_adjustment + subcopy_adjustment
        )
        candidate = (
            pair_score,
            left,
            right,
            similarity,
            headline_adjustment,
            subcopy_adjustment,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return [], []

    (
        _, left, right, similarity, headline_adjustment, subcopy_adjustment,
    ) = best
    selected = sorted(
        (left, right),
        key=lambda item: (
            -item.score,
            -item.headline_score,
            item.attempt,
            item.candidate_index,
        ),
    )
    metadata = [{
        "attempt": item.attempt,
        "candidate_index": item.candidate_index,
        "headline": item.copy["headline"],
        "subcopy": item.copy["body"],
        "subcopy_form": classify_subcopy_form(item.copy["body"], ad_input),
        "ad_score": item.score,
        "headline_ad_score": item.headline_score,
        "subcopy_score": item.subcopy_score,
        "quality_flags": list(item.flags),
        "pair_similarity": round(similarity, 3),
        "headline_diversity_adjustment": headline_adjustment,
        "subcopy_diversity_adjustment": subcopy_adjustment,
    } for item in selected]
    return [dict(item.copy) for item in selected], metadata


def _prompt_payload(ad_input: dict[str, Any], headline_sources: list[str]) -> dict[str, Any]:
    payload = previous._prompt_payload(ad_input)
    profile = analyze_copy_input(ad_input, image_provided=False)
    primary, secondary, inventory = assign_message_roles(ad_input)
    allowed_subcopy_forms = list(eligible_subcopy_forms(ad_input))
    payload.pop("low_ad_value_visuals", None)
    payload.pop("headline_source_priority", None)
    payload["message_inventory"] = inventory
    payload["message_routing_method"] = {
        "classification": "입력 근거를 상품·감성·상황·혜택 메시지로 분류",
        "primary_selection": "사용자가 직접 강조한 메시지와 구매 결정 가치, 근거 강도를 함께 고려해 주 메시지 선택",
        "secondary_selection": "주 메시지와 다른 정보를 더하면서 함께 읽을 때 하나의 광고가 되는 두 번째 메시지 선택",
        "fixed_template": "사용하지 않음. 입력별 메시지 중요도와 상호 보완성으로 역할 결정",
        "grounding": "message_inventory 밖의 감각·상황·효과·혜택을 추가하지 않음",
    }
    payload["role_assignment"] = {
        "primary_message": primary,
        "primary_role": "headline",
        "secondary_message": secondary,
        "secondary_role": (
            "subcopy" if profile.generation_mode == "headline_and_subcopy" else None
        ),
        "entity_repetition": "상품명·재료명·제품 종류의 필요한 반복 허용",
        "claim_repetition": "같은 맛·향·식감·조합·혜택 주장의 재서술 금지",
        "compound_fact": "연결된 복합 특징은 한 주장으로 보존",
    }
    payload["auto_composition"] = {
        "headline": "role_assignment.primary_message를 짧은 광고 메시지로 압축",
        "subcopy": (
            "role_assignment.secondary_message를 새로운 정보가 있는 완결된 문장·명사구·짧은 정보 나열 중 적합한 형식으로 전달"
            if profile.generation_mode == "headline_and_subcopy"
            else "생성하지 않음"
        ),
        "candidate_diversity": (
            "동일 역할 배정을 유지하되 내부 후보 4개에서 가능한 한 두 가지 이상의 안전한 형식을 사용하고 "
            "같은 보조 문구 전체를 반복하지 않음"
        ),
    }
    payload["subcopy_form_routing"] = {
        "eligible_forms": allowed_subcopy_forms,
        "relation_explanation": (
            "재료 관계·복합 식감처럼 요소 사이의 관계가 핵심이면 설명 문장형 우선"
        ),
        "simple_message": "하나의 간단한 메시지는 의미가 완결된 명사구 허용",
        "parallel_information": (
            "서로 독립적으로 확인된 정보가 두 개 이상일 때만 짧은 정보 나열 허용"
        ),
        "cta": (
            "명시적 행동 유도 요청이 있을 때만 허용"
            if explicit_cta_requested(ad_input)
            else "명시적 행동 유도 요청이 없으므로 사용 금지"
        ),
        "internal_candidate_preference": (
            "eligible_forms가 두 가지 이상이면 내부 후보 4개에 가능한 한 두 가지 이상의 형식을 사용"
        ),
        "final_pair_selection": (
            "동일한 보조 문구 전체는 선택하지 않고, 서로 다른 형식 조합을 높은 가점으로 우선하되 "
            "한 형식만 안전하면 같은 형식의 서로 다른 문구를 허용"
        ),
    }
    payload["quantity_policy"] = {
        "allowed": "사용자 입력의 판매 수량·세트·라인업·포장 단위·수량형 프로모션과 관찰 수량이 일치하는 경우",
        "forbidden": "사진 속 반복 배치, 토핑 개수 또는 판매 단위인지 확인되지 않은 관찰 수량",
        "validation": "상업적 수량 근거가 없으면 개수 표현을 제거하고 더 높은 우선순위의 확인 근거로 제목을 다시 작성",
    }
    if "subcopy_policy" in payload:
        for obsolete_key in (
            "verified_fact_required",
            "bare_noun_phrase",
            "non_imperative_experience",
            "imperative_cta",
        ):
            payload["subcopy_policy"].pop(obsolete_key, None)
        payload["subcopy_policy"]["role"] = (
            "제목에 배정되지 않은 두 번째 근거 메시지를 새로운 정보로 전달"
        )
        payload["subcopy_policy"]["grounded_message_required"] = True
        payload["subcopy_policy"]["allowed_forms"] = [
            "의미가 완결된 설명 문장",
            "의미가 완결된 명사구",
            "확인된 재료·풍미·식감의 짧은 정보 나열",
        ]
        payload["subcopy_policy"]["format_selection"] = (
            "재료 조합·복합 식감처럼 관계 설명이 필요하면 문장형을 우선하고, "
            "간단한 특징·플레이버 정보는 명사구나 나열형을 허용"
        )
        payload["subcopy_policy"]["completion"] = (
            "종결형 어미를 강제하지 않되 뒤에 수식 대상이나 서술어가 필요한 표현은 금지"
        )
        payload["subcopy_policy"]["sensory_grounding"] = (
            "입력에서 확인된 재료·풍미·식감만 사용하고 이미지의 재료·장식을 감각 속성으로 변환하지 않음"
        )
        payload["subcopy_policy"]["imperative_cta"] = (
            "입력에 명시적 판매·방문·행동 유도 요청이 있을 때만 허용"
            if explicit_cta_requested(ad_input)
            else "명시적 행동 유도 요청이 없으므로 사용 금지"
        )
        payload["subcopy_policy"]["duplicate_across_candidates"] = (
            "동일한 보조 문구 전체 반복 금지"
        )
    payload["headline_completion_policy"] = {
        "target_headroom": "최대 글자 수보다 2~4자 짧게 작성",
        "word_truncation": "금지. 단어·조사·어미를 자르지 말고 전체 제목을 더 짧게 다시 작성",
    }
    if profile.generation_mode == "headline_and_subcopy":
        payload["pair_validation"] = {
            "primary_in_headline": True,
            "secondary_in_subcopy": True,
            "product_fact_across_pair": "확인된 상품 사실은 제목과 보조 문구 전체에서 보존",
            "same_claim_rephrasing": "금지",
            "shared_entity_terms": "허용",
        }
    else:
        payload["headline_only_policy"] = {
            "allowed_sources": [
                "verified_product_facts",
                "copy_request의 명시적 콘셉트",
                "고신뢰 시각 정보",
            ],
            "product_fact_reuse": "보조 문구가 없으므로 제목에서 사용 가능",
            "priority": "상품 장점·요청 콘셉트를 일반 용기·배경 묘사보다 우선",
        }
    return payload


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    effective_input, switched, reason, sources = _prepare_effective_input(ad_input)
    effective_profile = analyze_copy_input(effective_input, image_provided=False)
    prefix = f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
    if switched:
        prefix += (
            f"자동 구성: {reason} 보조 문구는 출력하지 말고, 확인된 상품 특징을 "
            "광고성 있는 제목 근거로 사용할 수 있습니다.\n"
        )
    elif effective_profile.generation_mode == "headline_and_subcopy":
        prefix += (
            "message_inventory를 별도 배너 유형으로 나누지 말고 role_assignment에 따라 조합하세요. "
            "primary_message는 제목, secondary_message는 보조 문구에 반영하세요. "
            "보조 문구 형식은 메시지에 맞춰 선택하고, 내부 후보 4개에는 가능한 한 두 가지 이상의 형식을 사용하세요. "
            "동일한 보조 문구 전체를 반복하거나 종결형 어미를 일률적으로 붙이지 마세요. "
            "상품명·재료명의 필요한 반복은 허용하지만 같은 주장의 재서술은 금지합니다.\n"
        )
    else:
        prefix += (
            "제목만 정확히 4개 작성하세요. 확인된 상품 특징·요청 콘셉트를 일반적인 "
            "용기·배경 묘사보다 우선하세요.\n"
        )
    return prefix + json.dumps(
        _prompt_payload(effective_input, sources),
        ensure_ascii=False,
        indent=2,
    )


def build_candidate_retry_prompt(
    ad_input: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    headline_sources: list[str],
) -> str:
    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.generation_mode == "headline_only":
        retry_priority = [
            "hard_issues가 있었던 표현을 반복하지 않음",
            "상업적 근거 없는 이미지 개수를 제거하고 제목 우선순위의 다른 확인 근거로 교체",
            "verified_product_facts나 명시적 요청 콘셉트를 제목에 구체적으로 반영",
            "일반적인 컵·얼음·테이블 묘사보다 확인된 상품 장점을 우선",
            "제목 끝의 단어·조사·어미를 자르지 않고 최대 글자 수보다 2~4자 짧게 다시 작성",
            f"headline_ad_score가 {MIN_HEADLINE_AD_SCORE}점 미만인 제목을 다른 광고성 제목으로 교체",
        ]
        instruction = "제목만 포함한 새로운 내부 후보 4개를 반환하세요.\n"
    else:
        retry_priority = [
            "primary_message를 제목에, secondary_message를 보조 문구에 다시 배정",
            "상업적 근거 없는 이미지 개수를 제거하고 제목 우선순위의 다른 확인 근거로 교체",
            "상품명·재료명 반복은 허용하되 같은 감각·조합·혜택 주장 반복 제거",
            "복합 상품 특징의 일부만 떼어 다른 문구에서 다시 설명하지 않음",
            "보조 문구는 secondary_message에 맞는 완결 문장·명사구·짧은 정보 나열 중 적합한 형식으로 다시 작성",
            "안전한 형식이 두 가지 이상이면 내부 후보 4개에 가능한 한 두 가지 이상의 형식을 사용",
            "동일한 보조 문구 전체를 반복하지 않음",
            "종결형 어미를 억지로 붙이지 않고 뒤에 수식 대상이나 서술어가 필요한 불완전 표현 제거",
            "제목 끝의 단어·조사·어미를 자르지 않고 전체 제목을 더 짧게 다시 작성",
            "입력에 없는 광고 상황·상품 특징을 새로 만들지 않음",
        ]
        instruction = "역할이 겹친 후보를 버리고 새로운 내부 후보 4개를 반환하세요.\n"
    context = {
        **_prompt_payload(ad_input, headline_sources),
        "retry": True,
        "previous_candidates": diagnostics,
        "retry_priority": retry_priority,
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        + instruction
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _compound_fact_title_variants(fact: str) -> list[str]:
    """복합 사실의 모든 성분을 유지한 짧은 제목 변형만 만든다."""

    values: list[str] = []
    outer_inner = re.fullmatch(
        r"겉은\s+(.+?)하고\s+속은\s+(.+?)한\s+식감",
        fact,
    )
    if outer_inner:
        outer, inner = outer_inner.groups()
        values.extend([
            f"{outer}한 겉, {inner}한 속",
            f"겉은 {outer}, 속은 {inner}",
        ])

    paired_sensory = re.fullmatch(
        r"(.+?)하고\s+(.+?)한\s+(식감|풍미)",
        fact,
    )
    if paired_sensory:
        first, second, category = paired_sensory.groups()
        values.extend([
            f"{first}함과 {second}함",
            f"{first}·{second}한 {category}",
        ])

    combination = re.fullmatch(r"(.+?)(와|과)(.+?)의\s*조합", fact)
    if combination:
        left, connector, right = combination.groups()
        values.extend([
            f"{left}{connector} {right}",
            f"{right}{ad_value_base._and_particle(right)} {left}의 조화",
        ])

    if re.search(r"(?:와|과|하고|이며)", fact):
        values.append(
            re.sub(r"\s*(?:와|과|하고|이며)\s*", ", ", fact)
        )
    return values


def _headline_only_fallback_candidates(ad_input: dict[str, Any]) -> list[str]:
    """제목 우선순위와 복합 사실 보존을 지키는 결정론적 후보를 만든다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.strip(" ,.! ").split())
        if not 4 <= len(cleaned) <= profile.limits.headline:
            return
        if is_truncated_headline(cleaned, ad_input):
            return
        if headline_uses_ungrounded_quantity(cleaned, ad_input):
            return
        if _normalize_compact(cleaned) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(cleaned)

    for value in _campaign_title_candidates(ad_input):
        add(value)
    compound_fact = assigned_subcopy_fact(ad_input)
    fallback_facts = [compound_fact] if compound_fact else list(profile.verified_facts)
    for fact in fallback_facts:
        add(fact)
        for value in _compound_fact_title_variants(fact):
            add(value)
        for value in ad_value_base._fact_variants(fact):
            if base._fact_is_covered_strict(value, fact):
                add(value)
        add(f"{fact}의 매력")
        add(f"{fact} 그대로")
    for value in eligible_headline_candidates(ad_input):
        add(value)
    return values


def _message_title_candidates(
    message: dict[str, Any],
    ad_input: dict[str, Any],
) -> list[str]:
    profile = analyze_copy_input(ad_input, image_provided=False)
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.strip(" ,.! ").split())
        if not 4 <= len(cleaned) <= profile.limits.headline:
            return
        if is_truncated_headline(cleaned, ad_input):
            return
        if headline_uses_ungrounded_quantity(cleaned, ad_input):
            return
        if not _message_is_reflected(cleaned, message, ad_input):
            return
        if _normalize_compact(cleaned) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(cleaned)

    value = str(message["value"])
    add(value)
    if message["type"] in {"emotion", "situation"}:
        if "의 " in value:
            first, second = value.split("의 ", 1)
            add(f"{first}, {second}")
        add(f"{value}{ad_value_base._object_particle(value)} 담은 순간")
        add(f"{value}{ad_value_base._and_particle(value)} 함께하는 시간")
    elif message["type"] == "benefit":
        add(f"{value} 혜택")
        add(f"{value}{ad_value_base._subject_particle(value)} 있는 선택")
    elif message["source"] == "verified_visual_facts":
        add(f"{value}{ad_value_base._subject_particle(value)} 돋보이는 구성")
    else:
        for candidate in _compound_fact_title_variants(value):
            add(candidate)
        for candidate in ad_value_base._fact_variants(value):
            if base._fact_is_covered_strict(candidate, value):
                add(candidate)
        add(f"{value}의 매력")
        add(f"{value} 그대로")
    return values


def _safe_fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    ad_input, _, _, _ = _prepare_effective_input(ad_input)
    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.generation_mode != "headline_and_subcopy":
        hashtags = ad_value_base.build_deterministic_hashtags(ad_input)
        candidates = _headline_only_fallback_candidates(ad_input)
        for first, second in combinations(candidates, 2):
            fallback = [
                {"headline": first, "body": "", "hashtags": hashtags.copy()},
                {"headline": second, "body": "", "hashtags": hashtags.copy()},
            ]
            if not collect_banner_copy_issues(fallback, ad_input):
                return fallback
        raise CopyGenerationError(
            "제목 우선순위와 복합 사실 보존 검증을 통과하는 fallback 두 개를 만들 수 없습니다."
        )

    primary, secondary, _ = assign_message_roles(ad_input)
    if primary is None or secondary is None:
        raise CopyGenerationError(
            "제목과 보조 문구에 배정할 두 개의 근거 메시지가 없습니다."
        )
    bodies = _message_fallback_subcopies(secondary, ad_input)
    if len(bodies) < 2:
        raise CopyGenerationError(
            "두 번째 근거 메시지로 서로 다른 형식의 안전한 보조 문구 fallback 두 개를 만들 수 없습니다."
        )
    hashtags = ad_value_base.build_deterministic_hashtags(ad_input)
    candidates = _message_title_candidates(primary, ad_input)
    body_pairs = sorted(
        permutations(bodies, 2),
        key=lambda pair: (
            classify_subcopy_form(pair[0], ad_input)
            == classify_subcopy_form(pair[1], ad_input)
        ),
    )
    for first, second in combinations(candidates, 2):
        for first_body, second_body in body_pairs:
            copies = [
                {
                    "headline": first,
                    "body": first_body,
                    "hashtags": hashtags.copy(),
                },
                {
                    "headline": second,
                    "body": second_body,
                    "hashtags": hashtags.copy(),
                },
            ]
            if not _subcopy_pair_is_diverse(copies[0], copies[1], ad_input):
                continue
            if not collect_banner_copy_issues(copies, ad_input):
                return copies
    raise CopyGenerationError(
        "최종 검증을 통과하는 메시지 배정 fallback 두 개를 만들 수 없습니다."
    )


def _resolve_with_fallback(
    ad_input: dict[str, Any],
    ranked: list[previous.previous.RankedCandidate],
) -> tuple[list[AdCopy], tuple[int, ...], list[dict[str, Any]]]:
    fallback = _safe_fallback_options(ad_input)
    qualified = [candidate for candidate in ranked if candidate.qualifies]
    if not qualified:
        return fallback, (1, 2), []
    best = max(
        qualified,
        key=lambda item: (item.score, item.headline_score, -item.attempt, -item.candidate_index),
    )
    replacement_candidates = [
        copy for copy in fallback
        if _normalize_compact(copy["headline"])
        != _normalize_compact(best.copy["headline"])
        and _subcopy_pair_is_diverse(best.copy, copy, ad_input)
    ]
    best_form = classify_subcopy_form(best.copy["body"], ad_input)
    replacement = next(
        (
            copy for copy in replacement_candidates
            if classify_subcopy_form(copy["body"], ad_input) != best_form
        ),
        replacement_candidates[0] if replacement_candidates else None,
    )
    if replacement is None:
        return fallback, (1, 2), []
    resolved = [dict(best.copy), dict(replacement)]
    if collect_banner_copy_issues(resolved, ad_input):
        if collect_banner_copy_issues(fallback, ad_input):
            raise CopyGenerationError("fallback 최종 검증에 실패했습니다.")
        return fallback, (1, 2), []
    return resolved, (2,), [{
        "attempt": best.attempt,
        "candidate_index": best.candidate_index,
        "headline": best.copy["headline"],
        "ad_score": best.score,
        "headline_ad_score": best.headline_score,
        "subcopy_score": best.subcopy_score,
        "quality_flags": list(best.flags),
    }]


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    effective_input, switched, switch_reason, headline_sources = _prepare_effective_input(ad_input)
    profile = analyze_copy_input(effective_input, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider not in {"gemini", "openai"}:
        raise CopyGenerationError("배너 문구 생성은 Gemini 또는 OpenAI 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    ranked_pool: list[previous.previous.RankedCandidate] = []
    history: list[dict[str, Any]] = []
    first_issues: tuple[str, ...] = ()
    previous_diagnostics: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        prompt = (
            build_banner_copy_prompt(effective_input)
            if attempt == 1
            else build_candidate_retry_prompt(
                effective_input,
                previous_diagnostics,
                headline_sources,
            )
        )
        output_text = ""
        usage: dict[str, int] = {}
        try:
            response_schema = previous._response_schema(profile)
            if config.provider == "openai":
                output_text, usage = _generate_with_openai(
                    prompt=prompt,
                    model_id=model_id,
                    api_key=api_key,
                    client=client,
                    system_instructions=BANNER_COPY_SYSTEM_INSTRUCTIONS,
                    response_schema=response_schema,
                )
            else:
                output_text = _generate_with_gemini(
                    prompt=prompt,
                    model_id=model_id,
                    api_key=api_key,
                    client=client,
                    system_instructions=BANNER_COPY_SYSTEM_INSTRUCTIONS,
                    response_schema=response_schema,
                )
            returned = base.parse_banner_copy_options(
                output_text,
                effective_input,
                expected_count=INTERNAL_CANDIDATE_COUNT,
            )
            ranked, diagnostics = _rank_candidates(returned, effective_input, attempt=attempt)
            ranked_pool.extend(ranked)
            previous_diagnostics = diagnostics
            selected, selection = select_top_two_candidates(
                ranked_pool,
                effective_input,
            )
            hard_messages = [
                message for item in diagnostics for message in item["hard_issues"]
            ]
            history.append({
                "attempt": attempt,
                "repair": attempt > 1,
                "internal_candidate_count": len(returned),
                "hard_valid_candidate_count": len(ranked),
                "qualified_candidate_count": sum(candidate.qualifies for candidate in ranked),
                "raw_image_sent": False,
                "raw_output": output_text,
                "candidates": diagnostics,
                "selected": selection,
                "issues": hard_messages,
                "usage": usage,
                "auto_headline_only": switched,
                "auto_mode_reason": switch_reason,
            })
            if not first_issues and hard_messages:
                first_issues = tuple(hard_messages)
            if len(selected) == FINAL_OPTION_COUNT:
                final_issues = collect_banner_copy_issues(selected, effective_input)
                if not final_issues:
                    notices = []
                    if switch_reason:
                        notices.append(switch_reason)
                    if attempt > 1:
                        notices.append("역할 분리와 제목 최소 점수를 통과한 후보 중 상위 2개를 선택했습니다.")
                    return BannerCopyGenerationResult(
                        copies=selected,
                        generation_mode=profile.generation_mode,
                        status="generated" if attempt == 1 else "retried",
                        attempts=attempt,
                        notice=" ".join(notices) or None,
                        validation_issues=first_issues,
                        attempt_history=tuple(history),
                        prompt_version=_provider_copy_prompt_version(config.provider),
                    )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            history.append({
                "attempt": attempt,
                "repair": attempt > 1,
                "internal_candidate_count": 0,
                "hard_valid_candidate_count": 0,
                "qualified_candidate_count": 0,
                "raw_image_sent": False,
                "raw_output": output_text,
                "candidates": [],
                "selected": [],
                "issues": [error_message],
                "usage": usage,
                "auto_headline_only": switched,
                "auto_mode_reason": switch_reason,
            })
            if not first_issues:
                first_issues = (error_message,)

    resolved, fallback_indices, selection = _resolve_with_fallback(effective_input, ranked_pool)
    if history:
        history[-1]["selected"] = selection
    notices = []
    if switch_reason:
        notices.append(switch_reason)
    notices.append(
        "역할 분리와 제목 최소 점수를 통과한 내부 후보가 부족해 확인된 입력 기반 fallback으로 보완했습니다. "
        f"교체 선택지: {', '.join(map(str, fallback_indices))}"
    )
    return BannerCopyGenerationResult(
        copies=resolved,
        generation_mode=profile.generation_mode,
        status="fallback",
        attempts=max_attempts,
        notice=" ".join(notices),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=_provider_copy_prompt_version(config.provider),
    )


attach_visual_context = previous.attach_visual_context


def extract_verified_visual_facts(**kwargs: Any) -> Any:
    ad_value_base._generate_with_gemini = _generate_with_gemini
    return ad_value_base.extract_verified_visual_facts(**kwargs)
