"""광고 콘셉트 요청을 반영해 제목 후보만 생성하고 검증하는 흐름."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
from src.features.ad_copy.validation import _fact_is_reflected, validate_ad_copies


TITLE_CANDIDATE_PROMPT_VERSION = "gemini_copy_v1.7.0-context-aware-banner-titles"
OPTIONAL_BODY_PROMPT_VERSION = "gemini_copy_v1.4.0-optional-body-schema"
MAX_TITLE_LEXICAL_SIMILARITY = 0.75

GenerationStatus = Literal["generated", "retried", "fallback"]
BodyStatus = Literal["generated", "retried", "skipped", "omitted"]


class TitleInputError(CopyGenerationError):
    """제목 생성 전에 사용자가 입력을 보완해야 할 때 발생한다."""


@dataclass(frozen=True)
class TitleInputProfile:
    menu_name: str
    menu_terms: tuple[str, ...]
    selling_point: str
    distinct_facts: tuple[str, ...]
    headline_facts: tuple[str, ...]
    remaining_facts: tuple[str, ...]
    required_advantage_terms: tuple[str, ...]
    optional_information: dict[str, Any]
    input_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestConstraints:
    max_length: int | None = None
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    has_campaign_concept: bool = False
    allow_menu_name: bool = False
    requires_multi_fact_coverage: bool = False


@dataclass(frozen=True)
class TitleValidationIssue:
    code: str
    message: str
    candidate_index: int | None = None
    missing_terms: tuple[str, ...] = ()
    compared_with: int | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class TitleGenerationResult:
    copies: list[AdCopy]
    status: GenerationStatus
    attempts: int
    notice: str | None = None
    validation_issues: tuple[str, ...] = ()
    model_copy_count: int = 0
    fallback_copy_count: int = 0
    prompt_version: str = TITLE_CANDIDATE_PROMPT_VERSION


@dataclass(frozen=True)
class OptionalBodyResult:
    body: str
    status: BodyStatus
    attempts: int
    notice: str | None = None
    prompt_version: str = OPTIONAL_BODY_PROMPT_VERSION


_PARTICLES = (
    "에서는", "에게서", "으로", "에서", "에게", "처럼", "부터", "까지",
    "와", "과", "을", "를", "이", "가", "은", "는", "의", "에", "도", "만",
)
_GENERIC_INPUT_TERMS = {
    "특징", "특징인", "메뉴", "상품", "디저트", "음료",
    "두", "가지", "요소", "재료",
}
_RELATION_TERMS = {"조합", "조화", "만남", "어우러짐", "어우러진", "관계"}
_RELATION_PATTERN = re.compile(r"(?:조합|조화|만남|어우러짐|어우러진)")
_CANONICAL_ROOTS = {
    "바삭": "바삭", "촉촉": "촉촉", "쫄깃": "쫄깃", "부드러": "부드러",
    "달콤": "달콤", "쌉싸름": "쌉싸름", "담백": "담백", "고소": "고소",
    "깔끔": "깔끔", "진한": "진하", "진하게": "진하", "진함": "진하",
    "새콤": "새콤", "상큼": "상큼", "매콤": "매콤", "짭짤": "짭짤",
    "어우러": "관계", "조합": "관계", "조화": "관계", "만남": "관계",
}
_UNVERIFIED_SENSORY_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "촉촉한": ("촉촉",),
    "시원한": ("시원",),
    "청량한": ("청량",),
    "톡 쏘는": ("톡쏘",),
    "차가운": ("차갑", "차가운"),
    "따뜻한": ("따뜻",),
    "부드러운": ("부드러", "부드럽"),
    "달콤한": ("달콤",),
    "고소한": ("고소",),
    "바삭한": ("바삭",),
    "쫄깃한": ("쫄깃",),
    "상큼한": ("상큼",),
    "새콤한": ("새콤",),
    "쌉싸름한": ("쌉싸름",),
    "담백한": ("담백",),
    "매콤한": ("매콤",),
    "짭짤한": ("짭짤",),
    "향긋한": ("향긋",),
    "개운한": ("개운",),
    "크리미한": ("크리미",),
    "진한": ("진한", "진하게", "진함"),
    "은은한": ("은은",),
    "풍부한": ("풍부",),
    "깊은": ("깊은", "깊게", "깊이"),
}
_UNVERIFIED_FACT_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "가득": ("가득",),
    "퍼지는": ("퍼지",),
    "감도는": ("감돌",),
    "번지는": ("번지",),
    "본연의 맛": ("본연의맛", "본연"),
    "맛을 살린": ("맛을살린",),
    "씹을수록": ("씹을수록",),
    "첫": ("첫",),
    "마지막": ("마지막",),
}
_UNREQUESTED_EMBELLISHMENTS: dict[str, tuple[str, ...]] = {
    "고요한": ("고요",),
    "극적인": ("극적",),
    "매력적인": ("매력적",),
    "특별한": ("특별",),
}
_STYLE_TERMS: dict[str, tuple[str, ...]] = {
    "감성적": ("감성적", "감성적으로"),
    "직설적": ("직설적", "직설적으로"),
    "유머러스": ("유머러스", "재미있", "위트"),
    "고급스러운": ("고급스럽", "품격"),
    "트렌디한": ("트렌디", "힙한"),
    "간결한": ("간결", "짧게", "복잡한 표현 없이"),
    "강렬한": ("강렬", "짧고 강하게"),
}
_TONE_TERMS: dict[str, tuple[str, ...]] = {
    "친근한": ("친근",),
    "신뢰감 있는": ("신뢰",),
    "발랄한": ("발랄", "경쾌"),
    "차분한": ("차분",),
    "전문적인": ("전문적",),
    "긴박한": ("긴박",),
    "따뜻한": ("따뜻",),
    "활기찬": ("활기",),
    "세련된": ("세련",),
    "자연스러운": ("자연스럽",),
}
_TARGET_AUDIENCE_TERMS = (
    "직장인", "학생", "부모", "가족", "여성", "남성", "고객",
    "연인", "커플", "아이", "어린이", "청소년", "대학생", "1인 가구",
)
_USAGE_CONTEXT_TERMS = (
    "아침", "점심", "오후", "저녁", "밤", "출근", "출근길", "퇴근",
    "퇴근길", "주말", "휴식", "데이트", "모임", "선물", "기념일",
    "일상", "브런치", "간식", "회의", "공부", "운동", "여행",
    "봄", "여름", "가을", "겨울",
)
_EMOTIONAL_MESSAGE_TERMS = (
    "여유", "휴식", "설렘", "품격", "활력", "응원", "위로", "기분전환",
    "즐거움", "행복", "만족", "특별한 순간", "여백", "낭만", "추억",
)
_CHANNEL_TERMS = (
    "인스타그램", "SNS", "네이버", "상세페이지", "배너", "포스터",
    "메뉴판", "전단", "디지털 사이니지",
)
_PRODUCT_CLAIM_NOUNS = (
    "맛", "향", "풍미", "식감", "온도", "재료", "원산지", "효능",
    "신선도", "원두", "커피", "라떼", "음료", "빵", "크림", "치즈",
    "과일", "디저트", "케이크", "쿠키", "초콜릿",
)
_CAMPAIGN_CONTEXT_NOUNS = tuple(dict.fromkeys(
    _USAGE_CONTEXT_TERMS
    + _EMOTIONAL_MESSAGE_TERMS
    + (
        "하루", "순간", "시간", "마음", "기분", "분위기", "시작", "마무리",
    )
))
_PROMOTION_OR_PRICE_PATTERN = re.compile(
    r"(?:\d[\d,]*(?:원|%|퍼센트)|할인|특가|증정|무료|이벤트|행사|오늘만|기간)"
)
_EMOJI_PATTERN = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
)


def _normalize_compact(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _canonical_term(value: str) -> str:
    word = _normalize_compact(value)
    for particle in _PARTICLES:
        if word.endswith(particle) and len(word) > len(particle):
            word = word[: -len(particle)]
            break
    for source, target in _CANONICAL_ROOTS.items():
        if source in word:
            return target
    for ending in (
        "스러운", "스럽고", "로운", "롭고", "하게", "하고",
        "하다", "한", "함", "운",
    ):
        if word.endswith(ending) and len(word) > len(ending) + 1:
            word = word[: -len(ending)]
            break
    return word


def _terms(value: str) -> list[str]:
    return [
        term
        for raw in re.findall(r"[0-9A-Za-z가-힣]+", value)
        if (term := _canonical_term(raw))
    ]


def _same_or_contained_term(left: str, right: str) -> bool:
    return bool(
        left and right and (
            left == right
            or (len(left) >= 2 and left in right)
            or (len(right) >= 2 and right in left)
        )
    )


def _request_mentions_phrase(phrase: str, request: str) -> bool:
    phrase_terms = _terms(phrase)
    request_terms = _terms(request)
    return bool(phrase_terms) and all(
        any(
            _same_or_contained_term(phrase_term, request_term)
            for request_term in request_terms
        )
        for phrase_term in phrase_terms
    )


def _remove_menu_terms(terms: list[str], menu_terms: tuple[str, ...]) -> list[str]:
    return [
        term for term in terms
        if not any(_same_or_contained_term(term, menu_term) for menu_term in menu_terms)
    ]


def _fact_required_terms(fact: str, menu_terms: tuple[str, ...]) -> list[str]:
    values = _remove_menu_terms(_terms(fact), menu_terms)
    category_terms = {"맛", "향", "풍미", "식감", "결"}
    specific_terms = [
        term
        for term in values
        if term not in category_terms and term not in _GENERIC_INPUT_TERMS
    ]
    if specific_terms:
        values = specific_terms
    return list(dict.fromkeys(
        term for term in values
        if term not in _GENERIC_INPUT_TERMS
        and (len(term) >= 2 or term in {"맛", "향", "결", "겉", "속"})
    ))


def _select_headline_facts(
    facts: list[str],
    title_request: str,
) -> tuple[list[str], list[str]]:
    if not facts:
        return [], []
    request_terms = set(_terms(title_request))
    matched = [
        fact for fact in facts
        if request_terms and request_terms.intersection(_terms(fact))
    ]
    headline_facts = matched or [facts[0]]
    remaining = [fact for fact in facts if fact not in headline_facts]
    return headline_facts, remaining


def _requested_title_limit(title_request: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*자\s*이내", title_request)
    return min(20, int(match.group(1))) if match else None


_CAMPAIGN_CONCEPT_TERMS = (
    "직장인", "학생", "부모", "가족", "여성", "남성", "고객", "대상",
    "아침", "점심", "오후", "저녁", "밤", "출근", "퇴근", "주말",
    "휴식", "여유", "도전", "응원", "선물", "데이트", "모임", "일상",
    "기념", "계절", "봄", "여름", "가을", "겨울",
)
_FORBIDDEN_REQUEST_MARKERS = (
    "사용하지", "쓰지", "넣지", "제외", "금지", "빼고", "없이",
)
_REQUIRED_REQUEST_MARKERS = (
    "반드시", "꼭", "넣어", "포함", "사용해",
)


def _clean_constraint_phrase(value: str) -> str:
    return " ".join(value.strip(" \t\r\n.,:;·-").split())


def _matched_labels(
    text: str,
    vocabulary: dict[str, tuple[str, ...]],
) -> list[str]:
    normalized = _normalize_compact(text)
    return [
        label
        for label, variants in vocabulary.items()
        if any(_normalize_compact(variant) in normalized for variant in variants)
    ]


def parse_request_dimensions(
    title_request: str,
    tone_instruction: str | None = None,
) -> dict[str, Any]:
    """자유 입력을 상품 사실과 분리된 광고 작성 차원으로 정리한다."""

    combined = " ".join(
        value.strip()
        for value in (tone_instruction or "", title_request)
        if value and value.strip()
    )
    normalized = _normalize_compact(combined)
    request_normalized = _normalize_compact(title_request)

    targets = [
        term for term in _TARGET_AUDIENCE_TERMS
        if _normalize_compact(term) in request_normalized
    ]
    targets.extend(re.findall(r"\d{1,2}대", title_request))
    for match in re.finditer(
        r"([0-9A-Za-z가-힣 ]{1,20}?)(?:을|를)\s*대상(?:으로)?",
        title_request,
    ):
        phrase = _clean_constraint_phrase(match.group(1))
        if phrase:
            targets.append(phrase)

    usage_contexts = [
        term for term in _USAGE_CONTEXT_TERMS
        if _normalize_compact(term) in request_normalized
    ]
    emotional_messages = [
        term for term in _EMOTIONAL_MESSAGE_TERMS
        if _normalize_compact(term) in request_normalized
    ]
    channels = [
        term for term in _CHANNEL_TERMS
        if _normalize_compact(term) in request_normalized
    ]
    creative_scope = (
        "campaign_concept"
        if targets or usage_contexts or emotional_messages
        else "fact_focused"
    )
    return {
        "style": _matched_labels(combined, _STYLE_TERMS),
        "tone": _matched_labels(combined, _TONE_TERMS),
        "tone_instruction": tone_instruction or None,
        "target_audience": list(dict.fromkeys(targets)),
        "usage_context": list(dict.fromkeys(usage_contexts)),
        "emotional_message": list(dict.fromkeys(emotional_messages)),
        "channel": list(dict.fromkeys(channels)),
        "creative_scope": creative_scope,
        "has_campaign_concept": creative_scope == "campaign_concept",
    }


def parse_request_constraints(title_request: str) -> RequestConstraints:
    """자연어 요청 중 코드로 확인할 수 있는 명시 조건만 보수적으로 추출한다."""

    required: list[str] = []
    forbidden: list[str] = []
    quote_pattern = re.compile(r'["“”\'‘’]([^"“”\'‘’]{1,30})["“”\'‘’]')
    for match in quote_pattern.finditer(title_request):
        phrase = _clean_constraint_phrase(match.group(1))
        tail = title_request[match.end() : match.end() + 35]
        forbidden_positions = [
            tail.find(marker)
            for marker in _FORBIDDEN_REQUEST_MARKERS
            if marker in tail
        ]
        required_positions = [
            tail.find(marker)
            for marker in _REQUIRED_REQUEST_MARKERS
            if marker in tail
        ]
        nearest_forbidden = min(forbidden_positions, default=10_000)
        nearest_required = min(required_positions, default=10_000)
        if nearest_forbidden < nearest_required:
            forbidden.append(phrase)
        elif nearest_required < nearest_forbidden:
            required.append(phrase)

    field_patterns = (
        (r"(?:반드시\s*넣을|필수)\s*(?:단어|표현)?\s*[:：]\s*([^\n.]+)", required),
        (r"(?:제외할|금지할)\s*(?:단어|표현)?\s*[:：]\s*([^\n.]+)", forbidden),
    )
    for pattern, target in field_patterns:
        for match in re.finditer(pattern, title_request):
            for phrase in re.split(r"[,，/]", match.group(1)):
                cleaned = _clean_constraint_phrase(phrase)
                if cleaned:
                    target.append(cleaned)

    request_dimensions = parse_request_dimensions(title_request)
    has_campaign_concept = bool(request_dimensions["has_campaign_concept"])
    allow_menu_name = bool(
        re.search(
            r"메뉴명.{0,24}(?:넣|포함|드러|보이|강조|한눈)",
            title_request,
        )
    ) and not bool(
        re.search(
            r"메뉴명.{0,16}(?:사용하지|쓰지|넣지|제외|금지|빼)",
            title_request,
        )
    )
    requires_multi_fact_coverage = bool(
        re.search(
            r"(?:모두|둘\s*다|두\s*(?:가지|재료|요소)|균형|대비|함께|동시에)",
            title_request,
        )
    )
    return RequestConstraints(
        max_length=_requested_title_limit(title_request),
        required_phrases=tuple(dict.fromkeys(filter(None, required))),
        forbidden_phrases=tuple(dict.fromkeys(filter(None, forbidden))),
        has_campaign_concept=has_campaign_concept,
        allow_menu_name=allow_menu_name,
        requires_multi_fact_coverage=requires_multi_fact_coverage,
    )


def analyze_title_input(
    ad_input: dict[str, Any],
    *,
    image_provided: bool = False,
) -> TitleInputProfile:
    """메뉴 단어, 제목 필수 장점, 선택 정보를 모델 호출 전에 분리한다."""

    product = ad_input.get("product", {})
    campaign = ad_input.get("campaign", {})
    menu_name = str(product.get("name") or "").strip()
    selling_point = str(
        campaign.get("focus_feature") or product.get("features") or ""
    ).strip()
    title_request = str(campaign.get("copy_request") or "").strip()
    request_constraints = parse_request_constraints(title_request)
    tone_instruction = str(campaign.get("tone") or "").strip()
    request_dimensions = parse_request_dimensions(
        title_request,
        tone_instruction,
    )
    menu_terms = tuple(dict.fromkeys(_terms(menu_name)))
    facts = split_distinct_feature_facts(selling_point)
    headline_facts, remaining_facts = _select_headline_facts(facts, title_request)
    required_terms: list[str] = []
    for fact in headline_facts:
        required_terms.extend(_fact_required_terms(fact, menu_terms))
    required_terms = list(dict.fromkeys(required_terms))

    issues: list[str] = []
    requested_limit = request_constraints.max_length
    if requested_limit is not None and requested_limit < 6:
        issues.append("제목 요청 길이는 6자 이상으로 입력해주세요.")

    return TitleInputProfile(
        menu_name=menu_name,
        menu_terms=menu_terms,
        selling_point=selling_point,
        distinct_facts=tuple(facts),
        headline_facts=tuple(headline_facts),
        remaining_facts=tuple(remaining_facts),
        required_advantage_terms=tuple(required_terms),
        optional_information={
            "copy_request": title_request or None,
            "request_role": (
                "스타일·톤·타깃 고객·상황·강조점·글자 수·필수어·금지어를 "
                "지정하는 광고 작성 지시"
            ),
            "request_dimensions": request_dimensions,
            "request_constraints": {
                "max_length": request_constraints.max_length,
                "required_phrases": list(request_constraints.required_phrases),
                "forbidden_phrases": list(request_constraints.forbidden_phrases),
                "has_campaign_concept": request_constraints.has_campaign_concept,
                "allow_menu_name": request_constraints.allow_menu_name,
                "requires_multi_fact_coverage": (
                    request_constraints.requires_multi_fact_coverage
                ),
            },
            "image_provided": image_provided,
            "image_role": "명확하게 보이는 색상·형태·개수·포장 확인용 보조 정보",
        },
        input_issues=tuple(issues),
    )


def validate_title_input(ad_input: dict[str, Any]) -> list[str]:
    return list(analyze_title_input(ad_input).input_issues)


TITLE_CANDIDATE_SYSTEM_INSTRUCTIONS = """당신은 다양한 업종과 상품에 적용할 수 있는 한국어 광고 배너 제목 카피라이터입니다.

상품 이미지, 메뉴 정보, 확인된 상품 사실과 사용자의 문구 요청 사항을 바탕으로 1080×1080 광고 배너에 사용할 짧은 제목 후보만 작성하세요.

[입력의 역할]

- product_image: 이미지에서 명확하게 확인되는 상품 종류·색상·형태·개수·구성·포장만 참고하는 보조 정보
- menu: 상품 식별 정보. 배너에 별도로 표시되므로 기본적으로 전체 메뉴명을 제목에서 반복하지 않음
- verified_product_facts: 맛·향·식감·온도·재료·구성·제조 방식·원산지·효능 등 상품 특성을 표현할 수 있는 유일한 사실 근거
- copy_request.style: 감성적·직설적·유머러스·고급스러운·트렌디한 등 문구의 표현 방식
- copy_request.tone: 친근한·신뢰감 있는·발랄한·차분한·전문적인 등 고객에게 말하는 분위기
- copy_request.target_audience: 광고가 말을 거는 대상 고객
- copy_request.usage_context: 광고를 연결할 시간·장소·행동·상황
- copy_request.emotional_message: 여유·휴식·설렘·품격·활력 등 강조할 감성 메시지
- copy_request.channel: 제목이 사용될 매체
- copy_request의 나머지 필드: 글자 수·필수 표현·금지 표현·메뉴명 사용 여부·여러 사실 동시 반영 여부

[사실과 광고 콘셉트 구분]

1. 맛·향·식감·온도·재료·제조 방식·원산지·효능·품질은 verified_product_facts에서 확인된 내용만 표현하세요.
2. 스타일과 톤은 상품 사실이 아니라 어휘 선택·문장 리듬·말투를 정하는 지시입니다.
3. target_audience, usage_context, emotional_message가 있으면 이를 추상적인 광고 콘셉트로 재구성할 수 있습니다.
4. 광고 콘셉트를 표현할 때 사용자의 문장을 그대로 복사할 필요는 없습니다. 의미를 유지하는 은유·연상·바꿔쓰기를 사용할 수 있습니다.
5. 광고 콘셉트를 새로운 상품 특성이나 효능으로 바꾸면 안 됩니다.

[작성 모드]

- creative_scope가 fact_focused이면 확인된 상품 사실을 중심으로 작성하고, 입력에 없는 고객·상황·감정은 만들지 마세요.
- creative_scope가 campaign_concept이면 사용자가 입력한 고객·상황·감성 메시지를 중심으로 추상적이고 기억하기 쉬운 광고 제목을 만들 수 있습니다.
- campaign_concept 모드에서도 상품 자체의 맛·향·식감·온도·효능을 새로 만들면 안 됩니다.

[수식 표현 판단]

특정 단어를 무조건 금지하지 말고 무엇을 꾸미는지 판단하세요.

- 상황·감정을 꾸미는 표현은 입력된 광고 콘셉트와 연결되면 허용합니다.
  예: 깊은 여유, 따뜻한 순간, 활력 가득한 하루, 고요한 아침
- 상품 특성을 꾸미는 표현은 verified_product_facts에서 확인될 때만 허용합니다.
  예: 깊은 풍미, 따뜻한 음료, 풍부한 향, 입안에 퍼지는 맛

‘본연의 맛’, ‘씹을수록’, ‘갓 만든’, ‘신선한’, ‘집중력을 높이는’처럼 상품 경험·제조 상태·효능을 주장하는 표현도 verified_product_facts에 근거가 있어야 합니다.

[후보 작성 방법]

- 기본적으로 제목 후보 2개를 작성하세요.
- 첫 번째 후보는 사용자가 가장 강조한 사실 또는 광고 콘셉트를 명확하게 전달하세요.
- 두 번째 후보는 같은 근거를 다른 관점·어순·리듬으로 표현하세요.
- 확인된 사실이나 입력된 광고 콘셉트만으로 충분히 다른 표현이 가능할 때만 세 번째 후보를 작성하세요.
- requires_multi_fact_coverage가 true이면 첫 번째 후보에 요청된 여러 사실을 함께 반영하세요.
- 사실성이 후보 간 다양성보다 우선합니다.

[메뉴명 규칙]

- allow_menu_name이 false이면 전체 메뉴명을 제목에 반복하지 마세요.
- allow_menu_name이 true이면 후보 중 적어도 하나에 전체 메뉴명을 자연스럽게 포함하세요.
- 사실 표현에 필요한 재료명이나 메뉴명의 일부 단어는 사용할 수 있습니다.

[일반 예시]

- 요청: 고급스럽고 차분하게, 직장인을 대상으로 아침의 여유를 강조
- 허용: ‘하루의 품격을 깨우는 한 잔’
- 금지: ‘집중력을 높이는 깊고 풍부한 커피’

‘하루·품격·한 잔’은 입력된 고객·상황·감성 메시지를 재구성한 광고 콘셉트입니다. 반면 ‘집중력을 높인다·깊고 풍부하다’는 확인되지 않은 효능과 상품 특성입니다.

[공통 규칙]

1. 제목만 생성하고 body·본문·설명·후기·closing·해시태그는 생성하지 마세요.
2. 각 제목은 기본적으로 공백 포함 20자 이하이며 사용자가 더 짧게 요청하면 그 제한을 따르세요.
3. 제목은 짧더라도 독립적으로 이해되는 완결된 구여야 합니다.
4. 사용자가 지정한 필수 표현과 금지 표현을 지키세요.
5. 가격·할인·이벤트·기간·혜택을 넣지 마세요.
6. 근거 없는 최상급·인기·고객 반응을 추가하지 마세요.
7. 의미 없는 영어·별점·인증·이모지·명령형 CTA를 넣지 마세요.

[출력 형식]

아래 JSON 객체만 반환하세요.

{
  "headlines": ["제목 후보 1", "제목 후보 2", "선택적인 제목 후보 3"]
}

설명, 유형명, 검토 과정, body, 본문, 후기, closing, 해시태그와 마크다운 코드 블록은 출력하지 마세요."""


TITLE_CANDIDATE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headlines": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {"type": "string", "minLength": 1, "maxLength": 20},
        }
    },
    "required": ["headlines"],
    "additionalProperties": False,
}


OPTIONAL_BODY_SYSTEM_INSTRUCTIONS = """당신은 선택된 광고 제목을 보완하는 짧은 한국어 배너 본문을 작성합니다.

입력 JSON에는 menu_name, selected_headline, used_facts, remaining_facts가 명시됩니다. remaining_facts에 있는 확인 사실만 사용하세요. remaining_facts가 비어 있거나 자연스럽게 보완할 내용이 없으면 빈 문자열을 반환하세요.

[규칙]

1. remaining_facts에 적힌 사실만 사용하세요.
2. used_facts와 selected_headline의 내용을 다른 말로 반복하지 마세요.
3. 메뉴명, 매장명, 가격, 할인, 행사, 후기, closing과 행동 권유 문구를 넣지 마세요.
4. 입력에 없는 맛·향·식감·재료·제조 방식·신선도·효능과 강도 표현을 추가하지 마세요.
5. 본문은 공백 포함 34자 이하의 자연스러운 한 문장으로 작성하세요.
6. 제목만으로 충분하면 body를 빈 문자열로 반환하세요.

아래 JSON 객체만 반환하세요.

{
  "body": "제목에 없는 특징을 보완하는 문장 또는 빈 문자열"
}

설명, 후기, closing, 해시태그와 마크다운 코드 블록은 출력하지 마세요."""


OPTIONAL_BODY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body": {"type": "string", "minLength": 0, "maxLength": 34},
    },
    "required": ["body"],
    "additionalProperties": False,
}


def _profile_payload(profile: TitleInputProfile) -> dict[str, Any]:
    return {
        "product_image": {
            "provided": profile.optional_information["image_provided"],
            "role": profile.optional_information["image_role"],
        },
        "menu": {"name": profile.menu_name, "terms": list(profile.menu_terms)},
        "verified_product_facts": {
            "source_text": profile.selling_point,
            "headline_facts": list(profile.headline_facts),
            "fact_terms_for_validation": list(profile.required_advantage_terms),
        },
        "copy_request": {
            "raw": profile.optional_information["copy_request"],
            "role": profile.optional_information["request_role"],
            **profile.optional_information["request_dimensions"],
            **profile.optional_information["request_constraints"],
        },
    }


def build_title_candidate_prompt(
    ad_input: dict[str, Any],
    retry_feedback: str | None = None,
    *,
    image_provided: bool = False,
) -> str:
    profile = analyze_title_input(ad_input, image_provided=image_provided)
    prompt = (
        f"문구 프롬프트 버전: {TITLE_CANDIDATE_PROMPT_VERSION}\n"
        "다음 구조화 입력으로 광고 제목 후보 2~3개만 작성하세요.\n"
        + json.dumps(_profile_payload(profile), ensure_ascii=False, indent=2)
    )
    if retry_feedback:
        prompt += (
            "\n\n이전 응답의 아래 문제를 수정하되 입력에 없는 사실은 추가하지 마세요.\n- "
            + retry_feedback
        )
    return prompt


def _replacement_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "headlines": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {"type": "string", "minLength": 1, "maxLength": 20},
            }
        },
        "required": ["headlines"],
        "additionalProperties": False,
    }


def _issue_feedback(issue: TitleValidationIssue) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": issue.code, "message": issue.message}
    if issue.missing_terms:
        payload["missing_terms"] = list(issue.missing_terms)
    if issue.compared_with is not None:
        payload["compared_with_candidate"] = issue.compared_with
    if issue.similarity is not None:
        payload["similarity"] = round(issue.similarity, 3)
    return payload


def build_title_replacement_prompt(
    ad_input: dict[str, Any],
    copies: list[AdCopy],
    candidate_indexes: list[int],
    issues: list[TitleValidationIssue],
    *,
    image_provided: bool = False,
) -> str:
    profile = analyze_title_input(ad_input, image_provided=image_provided)
    zero_based = {index - 1 for index in candidate_indexes}
    context = {
        **_profile_payload(profile),
        "locked_candidates": [
            {"candidate_index": index + 1, "headline": copy["headline"]}
            for index, copy in enumerate(copies)
            if index not in zero_based
        ],
        "candidate_indexes_to_replace": candidate_indexes,
        "violations": [
            {"candidate_index": issue.candidate_index, **_issue_feedback(issue)}
            for issue in issues
            if issue.candidate_index in candidate_indexes
        ],
    }
    return (
        f"문구 프롬프트 버전: {TITLE_CANDIDATE_PROMPT_VERSION}\n"
        "검증에 실패한 후보만 교체하세요. locked_candidates는 그대로 유지하고, "
        "candidate_indexes_to_replace 순서에 맞춰 새 제목만 headlines 배열에 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _parse_headline_values(
    output_text: str,
    minimum: int,
    maximum: int,
) -> list[str]:
    try:
        payload = json.loads(output_text)
        raw_headlines = payload["headlines"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CopyGenerationError("AI 응답을 제목 후보 형식으로 읽을 수 없습니다.") from exc
    if not isinstance(raw_headlines, list) or not minimum <= len(raw_headlines) <= maximum:
        if minimum == maximum:
            raise CopyGenerationError(f"광고 제목은 정확히 {minimum}개여야 합니다.")
        raise CopyGenerationError(f"광고 제목 후보는 {minimum}~{maximum}개여야 합니다.")

    headlines: list[str] = []
    signatures: set[str] = set()
    for raw_headline in raw_headlines:
        if not isinstance(raw_headline, str) or not raw_headline.strip():
            raise CopyGenerationError("비어 있는 광고 제목이 있습니다.")
        headline = " ".join(raw_headline.split())
        if len(headline) > 20:
            raise CopyGenerationError("광고 제목은 공백 포함 20자 이하여야 합니다.")
        signature = _normalize_compact(headline)
        if signature in signatures:
            raise CopyGenerationError("동일한 광고 제목 후보가 반복되었습니다.")
        signatures.add(signature)
        headlines.append(headline)
    return headlines


def _headlines_to_copies(
    headlines: list[str],
    ad_input: dict[str, Any],
) -> list[AdCopy]:
    hashtags = build_deterministic_hashtags(ad_input)
    return [
        {"headline": headline, "body": "", "hashtags": hashtags.copy()}
        for headline in headlines
    ]


def parse_title_candidates(output_text: str, ad_input: dict[str, Any]) -> list[AdCopy]:
    return _headlines_to_copies(_parse_headline_values(output_text, 2, 3), ad_input)


def _term_is_reflected(
    term: str,
    headline_terms: set[str],
    headline: str,
) -> bool:
    normalized_headline = _normalize_compact(headline)
    return any(
        _same_or_contained_term(term, value) for value in headline_terms
    ) or (len(term) >= 2 and term in normalized_headline)


def _headline_reflects_fact(
    headline: str,
    fact: str,
    menu_terms: tuple[str, ...],
) -> bool:
    """제목이 한 사실의 핵심어 중 적어도 하나를 보존하는지 확인한다."""

    headline_terms = set(_terms(headline))
    fact_terms = _fact_required_terms(fact, menu_terms)
    if fact_terms:
        return any(
            _term_is_reflected(term, headline_terms, headline)
            for term in fact_terms
        )
    return _fact_is_reflected(headline, fact)


def _matching_expression_labels(
    text: str,
    source: str,
    expressions: dict[str, tuple[str, ...]],
) -> list[str]:
    """text에는 있지만 source에는 없는 표현 묶음의 표시명을 반환한다."""

    normalized_text = _normalize_compact(text)
    normalized_source = _normalize_compact(source)
    return [
        label
        for label, variants in expressions.items()
        if any(_normalize_compact(variant) in normalized_text for variant in variants)
        and not any(
            _normalize_compact(variant) in normalized_source
            for variant in variants
        )
    ]


def _nearest_term_distance(
    normalized_text: str,
    variants: tuple[str, ...],
    terms: tuple[str, ...],
) -> float:
    distances: list[float] = []
    for variant in variants:
        normalized_variant = _normalize_compact(variant)
        if not normalized_variant:
            continue
        start = 0
        while (variant_index := normalized_text.find(normalized_variant, start)) >= 0:
            variant_center = variant_index + len(normalized_variant) / 2
            for term in terms:
                normalized_term = _normalize_compact(term)
                if not normalized_term:
                    continue
                term_start = 0
                while (term_index := normalized_text.find(normalized_term, term_start)) >= 0:
                    term_center = term_index + len(normalized_term) / 2
                    distances.append(abs(variant_center - term_center))
                    term_start = term_index + 1
            start = variant_index + 1
    return min(distances, default=float("inf"))


def _context_aware_unsupported_labels(
    text: str,
    verified_facts: str,
    expressions: dict[str, tuple[str, ...]],
    *,
    creative_scope: str,
    product_terms: tuple[str, ...],
    context_terms: tuple[str, ...],
    allow_contextual_use: bool,
) -> list[str]:
    """상품 수식과 광고 콘셉트 수식을 가까운 명사 기준으로 구분한다."""

    normalized_text = _normalize_compact(text)
    normalized_facts = _normalize_compact(verified_facts)
    unsupported: list[str] = []
    for label, variants in expressions.items():
        normalized_variants = tuple(
            _normalize_compact(variant) for variant in variants
        )
        if not any(
            variant and variant in normalized_text
            for variant in normalized_variants
        ):
            continue
        if any(
            variant and variant in normalized_facts
            for variant in normalized_variants
        ):
            continue
        if allow_contextual_use and creative_scope == "campaign_concept":
            context_distance = _nearest_term_distance(
                normalized_text,
                variants,
                context_terms,
            )
            product_distance = _nearest_term_distance(
                normalized_text,
                variants,
                product_terms,
            )
            if context_distance <= 8 and context_distance < product_distance:
                continue
        unsupported.append(label)
    return unsupported


def title_lexical_similarity(left: str, right: str) -> float:
    """조사·활용형·관계 동의어를 정규화한 Jaccard 어휘 유사도."""

    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def collect_title_validation_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[TitleValidationIssue]:
    profile = analyze_title_input(ad_input)
    issues: list[TitleValidationIssue] = []
    title_request = str(profile.optional_information.get("copy_request") or "")
    request_constraints = parse_request_constraints(title_request)
    request_dimensions = profile.optional_information["request_dimensions"]
    creative_scope = str(request_dimensions["creative_scope"])

    for message in validate_ad_copies(copies, ad_input):
        if "입력한 실제 메뉴 특징이 구체적으로 반영되지 않았습니다" in message:
            # 후보별 모든 핵심어를 강제하는 공통 검증 대신 아래의 사실 단위
            # 제목 검증을 사용한다.
            continue
        if "입력 특징으로 확인되지 않는 수식어" in message:
            # 형태만 보는 공통 검증은 광고 제목의 분위기 표현을 상품 사실로
            # 오인한다. 아래의 명시적 사실·수식 표현 검사로 대체한다.
            continue
        if "입력 근거가 없는 감각 표현" in message:
            # 공통 검증은 tone까지 상품 사실의 출처로 간주하므로 제목 전용
            # 검증에서는 verified_product_facts만 근거로 다시 검사한다.
            continue
        match = re.search(r"문구\s+(\d+)", message)
        issues.append(
            TitleValidationIssue(
                code="BASE_VALIDATION",
                message=message,
                candidate_index=int(match.group(1)) if match else None,
            )
        )

    requested_limit = request_constraints.max_length
    normalized_menu = _normalize_compact(profile.menu_name)
    candidate_facts = profile.distinct_facts or profile.headline_facts
    coverage_facts = profile.headline_facts or candidate_facts
    product_terms = tuple(dict.fromkeys(
        _PRODUCT_CLAIM_NOUNS + profile.menu_terms
    ))
    context_terms = tuple(dict.fromkeys(
        _CAMPAIGN_CONTEXT_NOUNS
        + tuple(request_dimensions["target_audience"])
        + tuple(request_dimensions["usage_context"])
        + tuple(request_dimensions["emotional_message"])
    ))

    for index, copy in enumerate(copies, start=1):
        headline = copy["headline"].strip()
        if requested_limit is not None and len(headline) > requested_limit:
            issues.append(
                TitleValidationIssue(
                    code="TITLE_TOO_LONG",
                    message=f"제목 후보 {index}가 요청한 {requested_limit}자를 초과했습니다.",
                    candidate_index=index,
                )
            )

        normalized_headline = _normalize_compact(headline)
        if (
            not request_constraints.allow_menu_name
            and normalized_menu
            and normalized_menu in normalized_headline
        ):
            issues.append(
                TitleValidationIssue(
                    code="FULL_MENU_NAME_REPEATED",
                    message=f"제목 후보 {index}에 별도 표시되는 전체 메뉴명이 반복되었습니다.",
                    candidate_index=index,
                )
            )

        reflected_facts = [
            fact
            for fact in candidate_facts
            if _headline_reflects_fact(
                headline,
                fact,
                profile.menu_terms,
            )
        ]
        if (
            creative_scope == "fact_focused"
            and candidate_facts
            and not reflected_facts
        ):
            issues.append(
                TitleValidationIssue(
                    code="MISSING_FACT_GROUNDING",
                    message=(
                        f"제목 후보 {index}에 확인된 상품 장점이 "
                        "구체적으로 반영되지 않았습니다."
                    ),
                    candidate_index=index,
                )
            )
        if (
            index == 1
            and request_constraints.requires_multi_fact_coverage
            and len(coverage_facts) >= 2
        ):
            missing_facts = [
                fact for fact in coverage_facts
                if fact not in reflected_facts
            ]
            if missing_facts:
                issues.append(
                    TitleValidationIssue(
                        code="MISSING_REQUESTED_FACT_COVERAGE",
                        message=(
                            "첫 번째 제목에 함께 표현하도록 요청한 사실이 "
                            "빠졌습니다: " + ", ".join(missing_facts)
                        ),
                        candidate_index=index,
                    )
                )

        unsupported_sensory = _context_aware_unsupported_labels(
            headline,
            profile.selling_point,
            _UNVERIFIED_SENSORY_EXPRESSIONS,
            creative_scope=creative_scope,
            product_terms=product_terms,
            context_terms=context_terms,
            allow_contextual_use=True,
        )
        unsupported_factual_phrases = _context_aware_unsupported_labels(
            headline,
            profile.selling_point,
            _UNVERIFIED_FACT_EXPRESSIONS,
            creative_scope=creative_scope,
            product_terms=product_terms,
            context_terms=context_terms,
            allow_contextual_use=False,
        )
        unsupported_claims = list(dict.fromkeys(
            unsupported_sensory + unsupported_factual_phrases
        ))
        if unsupported_claims:
            issues.append(
                TitleValidationIssue(
                    code="UNVERIFIED_PRODUCT_EXPRESSION",
                    message=(
                        f"제목 후보 {index}에 확인되지 않은 상품 표현이 있습니다: "
                        + ", ".join(unsupported_claims)
                    ),
                    candidate_index=index,
                )
            )
        unrequested_embellishments = _context_aware_unsupported_labels(
            headline,
            profile.selling_point,
            _UNREQUESTED_EMBELLISHMENTS,
            creative_scope=creative_scope,
            product_terms=product_terms,
            context_terms=context_terms,
            allow_contextual_use=True,
        )
        if unrequested_embellishments:
            issues.append(
                TitleValidationIssue(
                    code="UNREQUESTED_EMBELLISHMENT",
                    message=(
                        f"제목 후보 {index}에 요청하지 않은 과장·분위기 표현이 있습니다: "
                        + ", ".join(unrequested_embellishments)
                    ),
                    candidate_index=index,
                )
            )
        for phrase in request_constraints.required_phrases:
            if _normalize_compact(phrase) not in normalized_headline:
                issues.append(
                    TitleValidationIssue(
                        code="MISSING_REQUESTED_PHRASE",
                        message=(
                            f"제목 후보 {index}에 반드시 넣도록 요청한 표현 "
                            f"'{phrase}'이 없습니다."
                        ),
                        candidate_index=index,
                    )
                )
        for phrase in request_constraints.forbidden_phrases:
            if _normalize_compact(phrase) in normalized_headline:
                issues.append(
                    TitleValidationIssue(
                        code="FORBIDDEN_REQUESTED_PHRASE",
                        message=(
                            f"제목 후보 {index}에 제외하도록 요청한 표현 "
                            f"'{phrase}'이 있습니다."
                        ),
                        candidate_index=index,
                    )
                )
        if _PROMOTION_OR_PRICE_PATTERN.search(headline):
            issues.append(
                TitleValidationIssue(
                    code="PRICE_OR_EVENT_TERM",
                    message=f"제목 후보 {index}에 가격·행사 표현이 포함되었습니다.",
                    candidate_index=index,
                )
            )
        if _EMOJI_PATTERN.search(headline):
            issues.append(
                TitleValidationIssue(
                    code="EMOJI",
                    message=f"제목 후보 {index}에 이모지가 포함되었습니다.",
                    candidate_index=index,
                )
            )

    for (left_index, left), (right_index, right) in combinations(
        enumerate(copies, start=1), 2
    ):
        similarity = title_lexical_similarity(
            left["headline"], right["headline"]
        )
        left_terms = _terms(left["headline"])
        right_terms = _terms(right["headline"])
        same_start = bool(
            left_terms and right_terms and left_terms[0] == right_terms[0]
        )
        left_signature = _normalize_compact(left["headline"])
        right_signature = _normalize_compact(right["headline"])
        # 사실 입력이 짧을 때는 같은 핵심어를 재사용하는 것이 안전하다.
        # 따라서 어휘 유사도는 진단 정보로만 계산하고, 완전히 동일한 후보만
        # 하드 실패로 처리한다.
        if left_signature == right_signature:
            issues.append(
                TitleValidationIssue(
                    code="TOO_SIMILAR",
                    message=(
                        f"제목 후보 {right_index}가 후보 {left_index}과 너무 유사합니다 "
                        f"(어휘 유사도 {similarity:.0%})."
                    ),
                    candidate_index=right_index,
                    compared_with=left_index,
                    similarity=similarity,
                )
            )

    deduplicated: list[TitleValidationIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for issue in issues:
        key = (
            issue.code,
            issue.message,
            issue.candidate_index,
            issue.compared_with,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(issue)
    return deduplicated


def validate_title_candidates(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[str]:
    return [
        issue.message
        for issue in collect_title_validation_issues(copies, ad_input)
    ]


def _fit_fallback_title(value: str, limit: int) -> str:
    """단어 중간 절단을 피하면서 fallback 제목을 길이에 맞춘다."""

    normalized = " ".join(value.split()).strip(" .!?。！？,，")
    if len(normalized) <= limit:
        return normalized
    shortened = normalized[:limit].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" ·,，")


def _fallback_title_copies(ad_input: dict[str, Any]) -> list[AdCopy]:
    profile = analyze_title_input(ad_input)
    request = str(profile.optional_information.get("copy_request") or "")
    constraints = parse_request_constraints(request)
    request_dimensions = profile.optional_information["request_dimensions"]
    limit = constraints.max_length or 20
    facts = list(profile.headline_facts or profile.distinct_facts)
    source = profile.selling_point or (facts[0] if facts else "") or profile.menu_name
    primary_source = (
        source
        if constraints.requires_multi_fact_coverage
        else (facts[0] if facts else source)
    )
    candidates: list[str] = []
    signatures: set[str] = set()

    def add(value: str) -> None:
        fitted = _fit_fallback_title(value, limit)
        if 0 < len(fitted) < 6:
            suffix = " 그대로"
            available = max(1, limit - len(suffix))
            fitted = (
                _fit_fallback_title(fitted, available) + suffix
            )
        signature = _normalize_compact(fitted)
        if fitted and signature and signature not in signatures:
            signatures.add(signature)
            candidates.append(fitted)

    if constraints.required_phrases:
        required_text = " ".join(constraints.required_phrases)
        if request_dimensions["creative_scope"] == "campaign_concept":
            add(f"{required_text}, 한 잔의 순간")
            add(f"한 잔의 {required_text}")
        else:
            add(f"{required_text} · {primary_source}")

    if constraints.allow_menu_name:
        add(f"{profile.menu_name} · {primary_source}")

    add(primary_source)
    for fact in facts:
        add(fact)

    if facts:
        relation_variant = re.sub(r"조합", "조화", facts[0])
        if relation_variant != facts[0]:
            add(relation_variant)

    if len(candidates) < 2:
        suffix = " 그대로"
        available = max(1, limit - len(suffix))
        core = _fit_fallback_title(primary_source, available)
        add(f"{core}{suffix}")

    if len(candidates) < 2:
        suffix = "에 집중"
        available = max(1, limit - len(suffix))
        core = _fit_fallback_title(primary_source, available)
        add(f"{core}{suffix}")

    return _headlines_to_copies(candidates[:2], ad_input)


def generate_safe_title_candidates(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> TitleGenerationResult:
    profile = analyze_title_input(
        ad_input,
        image_provided=image_bytes is not None,
    )
    if profile.input_issues:
        raise TitleInputError(" ".join(profile.input_issues))

    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider != "gemini":
        raise CopyGenerationError("제목 후보 생성은 Gemini 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    copies: list[AdCopy] | None = None
    issues: list[TitleValidationIssue] = []
    first_issue_messages: tuple[str, ...] = ()
    retry_feedback: str | None = None

    for attempt in range(1, max_attempts + 1):
        replacing_indexes = sorted(
            {
                issue.candidate_index
                for issue in issues
                if issue.candidate_index is not None
            }
        )
        targeted_retry = bool(copies is not None and replacing_indexes)
        if targeted_retry:
            prompt = build_title_replacement_prompt(
                ad_input,
                copies,
                replacing_indexes,
                issues,
                image_provided=image_bytes is not None,
            )
            response_schema = _replacement_schema(len(replacing_indexes))
        else:
            prompt = build_title_candidate_prompt(
                ad_input,
                retry_feedback,
                image_provided=image_bytes is not None,
            )
            response_schema = TITLE_CANDIDATE_RESPONSE_SCHEMA

        try:
            output_text = _generate_with_gemini(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                system_instructions=TITLE_CANDIDATE_SYSTEM_INSTRUCTIONS,
                response_schema=response_schema,
            )
            if targeted_retry and copies is not None:
                replacements = _parse_headline_values(
                    output_text,
                    len(replacing_indexes),
                    len(replacing_indexes),
                )
                updated = [dict(copy) for copy in copies]
                for candidate_index, headline in zip(
                    replacing_indexes,
                    replacements,
                ):
                    updated[candidate_index - 1]["headline"] = headline
                copies = updated
            else:
                copies = parse_title_candidates(output_text, ad_input)
        except Exception as exc:
            error_message = (
                "모델 요청 또는 응답 처리 오류: "
                f"{type(exc).__name__}: {exc}"
            )
            if not first_issue_messages:
                first_issue_messages = (error_message,)
            retry_feedback = error_message
            copies = None
            issues = []
            continue

        issues = collect_title_validation_issues(copies, ad_input)
        if not issues:
            return TitleGenerationResult(
                copies=copies,
                status="generated" if attempt == 1 else "retried",
                attempts=attempt,
                notice=(
                    None
                    if attempt == 1
                    else "검증에 실패한 제목 후보만 다시 생성했습니다."
                ),
                validation_issues=first_issue_messages,
                model_copy_count=len(copies),
                fallback_copy_count=0,
            )
        if not first_issue_messages:
            first_issue_messages = tuple(issue.message for issue in issues)
        retry_feedback = " ".join(issue.message for issue in issues)

    fallback_copies = _fallback_title_copies(ad_input)
    return TitleGenerationResult(
        copies=fallback_copies,
        status="fallback",
        attempts=max_attempts,
        notice="모델 제목이 검증을 통과하지 못해 입력 정보 기반 제목을 사용했습니다.",
        validation_issues=(
            first_issue_messages
            or tuple(issue.message for issue in issues)
        ),
        model_copy_count=0,
        fallback_copy_count=len(fallback_copies),
    )


def build_optional_body_input(
    ad_input: dict[str, Any],
    headline: str,
) -> dict[str, Any]:
    """선택 제목이 소비한 사실과 Body에 허용할 남은 사실을 명시한다."""

    profile = analyze_title_input(ad_input)
    headline_terms = set(_terms(headline))
    used_facts: list[str] = []
    for fact in profile.distinct_facts:
        required_terms = _fact_required_terms(fact, profile.menu_terms)
        is_reflected = (
            all(
                _term_is_reflected(term, headline_terms, headline)
                for term in required_terms
            )
            if required_terms
            else _fact_is_reflected(headline, fact)
        )
        if is_reflected:
            used_facts.append(fact)
    remaining_facts = [
        fact
        for fact in profile.distinct_facts
        if fact not in used_facts
    ]
    return {
        "menu_name": profile.menu_name,
        "selected_headline": headline,
        "used_facts": used_facts,
        "remaining_facts": remaining_facts,
    }


def _remaining_facts(ad_input: dict[str, Any], headline: str) -> list[str]:
    return list(build_optional_body_input(ad_input, headline)["remaining_facts"])


def build_optional_body_prompt(
    ad_input: dict[str, Any],
    headline: str,
    remaining_facts: list[str] | None = None,
    retry_feedback: str | None = None,
) -> str:
    context = build_optional_body_input(ad_input, headline)
    if remaining_facts is not None:
        context["remaining_facts"] = remaining_facts
        context["used_facts"] = [
            fact
            for fact in analyze_title_input(ad_input).distinct_facts
            if fact not in remaining_facts
        ]
    prompt = (
        f"본문 프롬프트 버전: {OPTIONAL_BODY_PROMPT_VERSION}\n"
        "다음 Body 전용 입력 JSON으로 필요한 경우에만 짧은 body를 작성하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )
    if retry_feedback:
        prompt += "\n\n이전 응답 문제를 수정하세요.\n- " + retry_feedback
    return prompt


def generate_optional_banner_body(
    ad_input: dict[str, Any],
    headline: str,
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> OptionalBodyResult:
    remaining_facts = _remaining_facts(ad_input, headline)
    if not remaining_facts:
        return OptionalBodyResult(
            body="",
            status="skipped",
            attempts=0,
            notice="제목과 다른 확인 정보가 없어 본문을 생략했습니다.",
        )

    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider != "gemini" or (not api_key.strip() and client is None):
        return OptionalBodyResult(
            body="",
            status="omitted",
            attempts=0,
            notice="본문을 안전하게 생성할 수 없어 제목과 메뉴명만 사용했습니다.",
        )

    retry_feedback: str | None = None
    for attempt in range(1, max_attempts + 1):
        prompt = build_optional_body_prompt(
            ad_input,
            headline,
            remaining_facts,
            retry_feedback,
        )
        try:
            output_text = _generate_with_gemini(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                system_instructions=OPTIONAL_BODY_SYSTEM_INSTRUCTIONS,
                response_schema=OPTIONAL_BODY_RESPONSE_SCHEMA,
            )
            payload = json.loads(output_text)
            body = " ".join(str(payload["body"]).split())
        except Exception:
            retry_feedback = "응답 형식 또는 모델 요청을 완료하지 못했습니다."
            continue
        if not body:
            return OptionalBodyResult(
                body="",
                status="skipped",
                attempts=attempt,
                notice="추가할 확인 정보가 없어 본문을 생략했습니다.",
            )

        copy: AdCopy = {
            "headline": headline,
            "body": body,
            "hashtags": build_deterministic_hashtags(ad_input),
        }
        issues = validate_ad_copies([copy], ad_input)
        if not issues and any(_fact_is_reflected(body, fact) for fact in remaining_facts):
            return OptionalBodyResult(
                body=body,
                status="generated" if attempt == 1 else "retried",
                attempts=attempt,
                notice="제목에 담기지 않은 확인 정보를 본문으로 보완했습니다.",
            )
        retry_feedback = " ".join(issues) or "남은 확인 특징이 본문에 반영되지 않았습니다."

    return OptionalBodyResult(
        body="",
        status="omitted",
        attempts=max_attempts,
        notice="검증된 본문을 만들 수 없어 제목과 메뉴명만 사용했습니다.",
    )
