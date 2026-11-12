"""AI 광고 문구를 검증하고 실패 시 안전한 기본 문구를 제공한다."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Any, Literal

from src.features.ad_copy.generator import (
    COPY_PROMPT_VERSION,
    AdCopy,
    build_deterministic_hashtags,
    generate_ad_copies,
    split_distinct_feature_facts,
)


GenerationStatus = Literal["generated", "retried", "mixed", "fallback"]

MIN_HEADLINE_CHARS = 6
MAX_HEADLINE_CHARS = 20
MAX_BODY_CHARS = 34
PROMOTION_LABEL_TERMS = (
    "할인",
    "특가",
    "증정",
    "무료",
    "행사",
    "평일",
    "주말",
    "오전",
    "오후",
    "이후",
    "이전",
    "테이크아웃",
    "포장",
)
TERMINAL_PUNCTUATION = (".", "!", "?", "。", "！", "？")
KOREAN_PARTICLES = (
    "에서",
    "에게",
    "으로",
    "와",
    "과",
    "을",
    "를",
    "이",
    "가",
    "은",
    "는",
    "의",
    "에",
    "도",
    "만",
)


@dataclass(frozen=True)
class SafeGenerationResult:
    """검증과 재시도를 포함한 최종 광고 문구 생성 결과."""

    copies: list[AdCopy]
    status: GenerationStatus
    attempts: int
    notice: str | None = None
    validation_issues: tuple[str, ...] = ()
    model_copy_count: int = 3
    fallback_copy_count: int = 0
    prompt_version: str = COPY_PROMPT_VERSION


# 근거가 없을 때 소비자를 오인시킬 가능성이 큰 표현들이다. 입력에 같은 표현이
# 명시되어 있으면 허용하고, 그렇지 않으면 재생성 대상으로 분류한다.
RISKY_CLAIM_TERMS = (
    "할인",
    "특가",
    "무료",
    "증정",
    "한정",
    "오늘만",
    "최저",
    "저렴",
    "가성비",
    "합리적",
    "최고",
    "최상",
    "극상",
    "1위",
    "유일",
    "건강",
    "다이어트",
    "면역",
    "피로",
    "효능",
    "칼로리",
    "무설탕",
    "비건",
    "유기농",
    "국산",
    "인증",
    "수제",
    "직접 만든",
    "당일",
    "신선",
    "오늘 구운",
    "갓 구운",
    "매일 마셔도 질리지 않는",
    "누구나 좋아하는",
    "꼭 먹어야 하는",
    "언제 마셔도 부담 없는",
)

# 입력 근거와 무관하게 수정해야 하는 맞춤법·문장 오류다.
INVALID_COPY_EXPRESSIONS = (
    "디저트에요",
    "향이 입안에 퍼져보세요",
    "퍼져보세요",
    "맛을 맛보세요",
    "특징인",
    "의,",
    "을,",
    "를,",
    "조합을 느껴보세요",
    "향을 맛보세요",
    "특징적인",
    "특징",
)

MODIFIER_ENDINGS = (
    "로운",
    "스러운",
    "적인",
    "하게",
    "한",
    "좋은",
    "깊은",
    "있는",
    "없는",
    "다른",
)
NEUTRAL_MODIFIER_ROOTS = (
    "편안",
    "특별",
    "즐거",
    "여유",
    "가득",
)

# 대표 실험에서 확인된, 상품 특징 없이도 어느 상품에나 붙일 수 있는 제목이다.
# 프롬프트 지시가 실패했을 때 한 번 더 생성하도록 보수적으로 막는다.
GENERIC_HEADLINE_EXPRESSIONS = (
    "활기찬 오후를 채우는 맛",
    "바쁜 일상 속에서 만나는 휴식",
    "이색적인 풍미로 채우는 오후",
)

GENERIC_BODY_EXPRESSIONS = (
    "풍미를 즐겨보세요.",
    "매력을 느껴보세요.",
)

# 이 말들은 제품명이나 사진만으로 추론하면 사실 오류가 되기 쉽다. 사용자가
# 특징·행사·분위기에 같은 감각을 직접 입력했을 때만 허용한다.
GROUNDED_SENSORY_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "촉촉한": ("촉촉",),
    "시원한": ("시원",),
    "청량감": ("청량",),
    "톡 쏘는": ("톡쏘",),
    "차가운": ("차갑", "차가운", "차갑게"),
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
    "진한": ("진한", "진하게"),
    "은은한": ("은은",),
    "풍부한": ("풍부",),
}

INCOMPLETE_HEADLINE_ENDINGS = (
    "가득한",
    "마시는",
    "즐기는",
    "담은",
    "채우는",
    "전해지는",
    "느껴지는",
    "퍼지는",
    "스며드는",
    "어우러지는",
    "선사하는",
)

GENERIC_FEATURE_WORDS = {
    "특징",
    "특징인",
    "조합",
    "풍미",
    "식감",
    "메뉴",
    "상품",
    "디저트",
    "음료",
    "베이커리",
}


def _source_texts(ad_input: dict[str, Any]) -> list[str]:
    product = ad_input.get("product", {})
    values = (
        ad_input.get("store", {}).get("name"),
        product.get("name"),
        product.get("base_price", product.get("price")),
        product.get("sale_price"),
        product.get("features"),
        ad_input.get("campaign", {}).get("promotion"),
        ad_input.get("campaign", {}).get("tone"),
    )
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _normalize_token(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _number_tokens(value: str) -> set[str]:
    return {
        token.replace(",", "").replace(".", "")
        for token in re.findall(r"\d+(?:[,.]\d+)*", value)
    }


def _display_length(value: str) -> int:
    """모델이 만든 불규칙한 공백을 정리한 뒤 화면용 글자 수를 센다."""
    return len(" ".join(value.split()))


def _feature_keywords(value: str | None) -> set[str]:
    """입력 특징에서 실제 메뉴를 구분하는 핵심 단어와 어간을 추린다."""
    if not value:
        return set()
    keywords: set[str] = set()
    for raw_word in re.findall(r"[0-9A-Za-z가-힣]+", value):
        word = _normalize_token(raw_word)
        for particle in KOREAN_PARTICLES:
            if word.endswith(particle) and len(word) > len(particle):
                word = word[: -len(particle)]
                break
        if (len(word) < 2 and word not in {"향", "맛", "결"}) or word in GENERIC_FEATURE_WORDS:
            continue
        keywords.add(word)
    return keywords


def _fact_anchor_terms(value: str) -> set[str]:
    """서로 다른 특징을 구분할 때 사용할 핵심 어간을 만든다."""
    anchors: set[str] = set()
    for raw_word in re.findall(r"[0-9A-Za-z가-힣]+", value):
        word = _normalize_token(raw_word)
        for particle in KOREAN_PARTICLES:
            if word.endswith(particle) and len(word) > len(particle):
                word = word[: -len(particle)]
                break
        for ending in ("스러운", "로운", "한", "함", "운"):
            if word.endswith(ending) and len(word) > len(ending) + 1:
                word = word[: -len(ending)]
                break
        if word in GENERIC_FEATURE_WORDS or len(word) < 2:
            continue
        anchors.add(word)
    return anchors


def _fact_is_reflected(text: str, fact: str) -> bool:
    normalized_text = _normalize_token(text)
    return any(
        anchor in normalized_text
        or (len(anchor) >= 3 and anchor[:2] in normalized_text)
        for anchor in _fact_anchor_terms(fact)
    )


def _has_feature_grounding(
    text: str,
    features: str | None,
    product_name: str | None = None,
) -> bool:
    relation_match = re.fullmatch(
        r"\s*(.+?)(?:와|과)\s*(.+?)의\s*조합\s*",
        features or "",
    )
    if relation_match:
        normalized_text = _normalize_token(text)
        relation_is_preserved = any(
            term in normalized_text
            for term in ("조합", "조화", "어우러", "만남", "만나")
        )
        operand_keywords = [
            _feature_keywords(operand)
            for operand in relation_match.groups()
        ]
        operands_are_preserved = all(
            keywords
            and any(keyword in normalized_text for keyword in keywords)
            for keywords in operand_keywords
        )
        return relation_is_preserved and operands_are_preserved

    keywords = _feature_keywords(features)
    if product_name:
        product_words = {
            _normalize_token(word)
            for word in re.findall(r"[0-9A-Za-z가-힣]+", product_name)
        }
        keywords = {
            keyword
            for keyword in keywords
            if not any(
                keyword == product_word
                or keyword in product_word
                or product_word in keyword
                for product_word in product_words
                if product_word
            )
        }
    if not keywords:
        return not features or "조합" not in features or "조합" in text
    normalized_text = _normalize_token(text)
    return any(
        keyword in normalized_text
        or (len(keyword) >= 3 and keyword[:2] in normalized_text)
        for keyword in keywords
    )


def _contains_product_reference(text: str, product_name: str) -> bool:
    normalized_text = _normalize_token(text)
    normalized_product = _normalize_token(product_name)
    # 상품명 전체의 반복만 차단한다. 상품명에 포함된 단어가 입력 특징에 적힌
    # 실제 재료일 수 있으므로, 서로 떨어져 등장하는 재료명까지 상품명 반복으로
    # 판정하지 않는다.
    return bool(normalized_product and normalized_product in normalized_text)


def _unsupported_modifiers(text: str, features: str | None) -> list[str]:
    """입력 특징에 없는 수식어를 형태 기반으로 찾는다."""
    normalized_features = _normalize_token(features or "")
    unsupported: list[str] = []
    for raw_word in re.findall(r"[0-9A-Za-z가-힣]+", text):
        word = _normalize_token(raw_word)
        if not word or not word.endswith(MODIFIER_ENDINGS):
            continue
        if any(root in word for root in NEUTRAL_MODIFIER_ROOTS):
            continue
        stem = re.sub(r"(?:로운|스러운|적인|하게|좋은|깊은|있는|없는|다른|한)$", "", word)
        if stem in normalized_features or word in normalized_features:
            continue
        if len(stem) >= 2 and stem[:2] in normalized_features:
            continue
        unsupported.append(raw_word)
    return unsupported


def _allowed_hashtag_terms(source_texts: list[str]) -> set[str]:
    allowed: set[str] = set()
    for source in source_texts:
        normalized_source = _normalize_token(source)
        if normalized_source:
            allowed.add(normalized_source)

        words = [
            _normalize_token(word)
            for word in re.findall(r"[0-9A-Za-z가-힣]+", source)
        ]
        words = [word for word in words if word]
        word_variants: list[set[str]] = []
        for word in words:
            variants = {word}
            for particle in KOREAN_PARTICLES:
                if word.endswith(particle) and len(word) > len(particle):
                    variants.add(word[: -len(particle)])
            word_variants.append(variants)
        # 한 입력 항목 안에 있는 단어의 생략·결합까지만 허용한다.
        for size in range(1, min(len(word_variants), 5) + 1):
            for selected_groups in combinations(word_variants, size):
                combined = {""}
                for group in selected_groups:
                    combined = {
                        prefix + variant
                        for prefix in combined
                        for variant in group
                    }
                allowed.update(combined)
    return allowed


def validate_ad_copies(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[str]:
    """입력에 근거가 없는 수치·위험 주장·해시태그를 찾아 반환한다."""
    sources = _source_texts(ad_input)
    source_text = " ".join(sources).casefold()
    allowed_numbers = _number_tokens(source_text)
    allowed_hashtags = _allowed_hashtag_terms(sources)
    issues: list[str] = []

    for index, copy in enumerate(copies, start=1):
        headline = copy["headline"].strip()
        body = copy["body"].strip()
        combined = f"{headline} {body}".casefold()

        headline_length = _display_length(headline)
        body_length = _display_length(body)

        if headline_length < MIN_HEADLINE_CHARS:
            issues.append(
                f"문구 {index}의 제목이 {MIN_HEADLINE_CHARS}자보다 짧습니다."
            )
        if headline_length > MAX_HEADLINE_CHARS:
            issues.append(
                f"문구 {index}의 제목이 {MAX_HEADLINE_CHARS}자를 넘습니다."
            )
        if body_length > MAX_BODY_CHARS:
            issues.append(f"문구 {index}의 본문이 {MAX_BODY_CHARS}자를 넘습니다.")
        if "\n" in copy["headline"] or "\n" in copy["body"]:
            issues.append(f"문구 {index}에 불필요한 줄바꿈이 있습니다.")
        if headline.endswith(TERMINAL_PUNCTUATION):
            issues.append(f"문구 {index}의 제목 끝에 불필요한 문장부호가 있습니다.")
        last_headline_word = headline.split()[-1] if headline.split() else ""
        if last_headline_word.endswith(INCOMPLETE_HEADLINE_ENDINGS):
            issues.append(f"문구 {index}의 제목이 완결되지 않은 관형형 표현입니다.")
        if headline.startswith("느껴지는"):
            issues.append(f"문구 {index}의 제목이 어색한 수동형으로 시작합니다.")
        if body and not body.endswith("요."):
            issues.append(
                f"문구 {index}의 본문이 자연스러운 해요체와 마침표로 끝나지 않습니다."
            )
        if body and body.endswith("보세요."):
            issues.append(
                f"문구 {index}의 본문이 소비자에게 행동을 요청하는 표현으로 끝납니다."
            )

        for expression in INVALID_COPY_EXPRESSIONS:
            if expression in combined:
                issues.append(
                    f"문구 {index}에 수정이 필요한 표현 '{expression}'이 있습니다."
                )
        if re.search(r"(?:을|를)\s*느껴져요\.$", body):
            issues.append(
                f"문구 {index}의 본문에서 목적격 조사와 '느껴져요'의 연결이 어색합니다."
            )
        if re.search(r"의\s+(?:한|두|세|네|여러)\s+가지", combined):
            issues.append(
                f"문구 {index}에 수량어의 위치가 어색한 표현이 있습니다."
            )
        if re.search(r"(?:풍미|식감|향|조합)\s+(?:매력|즐거움|시간|순간)", combined):
            issues.append(
                f"문구 {index}에 필요한 조사가 빠진 명사 나열이 있습니다."
            )
        if re.search(r"(?:이|가)\s*(?:매력|특징)$", headline):
            issues.append(
                f"문구 {index}의 제목이 서술어 없이 끝나는 어색한 표현입니다."
            )
        for noun in ("향", "풍미", "식감", "맛", "조합"):
            if body.count(noun) >= 2:
                issues.append(
                    f"문구 {index}의 본문에서 핵심 단어 '{noun}'이 반복됩니다."
                )
        for text_name, text in (("제목", headline), ("본문", body)):
            has_complete_pair = re.search(
                r"[0-9A-Za-z가-힣 ]+(?:와|과)\s*[0-9A-Za-z가-힣 ]+의 조합",
                text,
            )
            has_safe_generic_pair = any(
                phrase in text
                for phrase in ("두 가지 메뉴의 조합", "두 가지 풍미의 조합", "조합의 매력")
            )
            if "의 조합" in text and not has_complete_pair and not has_safe_generic_pair:
                issues.append(
                    f"문구 {index}의 {text_name}에서 조합의 두 대상이 완전하게 표현되지 않았습니다."
                )

        features = ad_input.get("product", {}).get("features")
        product_name = ad_input.get("product", {}).get("name")
        if (
            isinstance(features, str)
            and ("두 가지 메뉴" in combined or "두 메뉴" in combined)
            and "두 가지 메뉴" not in features
            and "두 메뉴" not in features
        ):
            issues.append(
                f"문구 {index}에서 한 상품의 재료를 두 가지 메뉴로 표현했습니다."
            )
        if isinstance(features, str) and not _has_feature_grounding(
            f"{headline} {body}",
            features,
            product_name if isinstance(product_name, str) else None,
        ):
            issues.append(
                f"문구 {index}에 입력한 실제 메뉴 특징이 구체적으로 반영되지 않았습니다."
            )
        distinct_facts = split_distinct_feature_facts(
            features if isinstance(features, str) else None
        )
        if len(distinct_facts) >= 2 and body:
            headline_fact_indexes = {
                fact_index
                for fact_index, fact in enumerate(distinct_facts)
                if _fact_is_reflected(headline, fact)
            }
            body_fact_indexes = {
                fact_index
                for fact_index, fact in enumerate(distinct_facts)
                if _fact_is_reflected(body, fact)
            }
            if not any(
                headline_index != body_index
                for headline_index in headline_fact_indexes
                for body_index in body_fact_indexes
            ):
                issues.append(
                    f"문구 {index}의 제목과 본문이 서로 다른 입력 특징을 나누어 설명하지 않습니다."
                )
        unsupported_modifiers = _unsupported_modifiers(f"{headline} {body}", features)
        if unsupported_modifiers:
            values = ", ".join(dict.fromkeys(unsupported_modifiers))
            issues.append(
                f"문구 {index}에 입력 특징으로 확인되지 않는 수식어({values})가 있습니다."
            )

        tone = ad_input.get("campaign", {}).get("tone")
        if (
            isinstance(tone, str)
            and _normalize_token(tone)
            and _normalize_token(tone) in _normalize_token(combined)
        ):
            issues.append(
                f"문구 {index}에 문체 지시용 광고 분위기 값이 직접 반복됩니다."
            )

        store_name = ad_input.get("store", {}).get("name")
        if (
            isinstance(store_name, str)
            and _normalize_token(store_name)
            and _normalize_token(store_name) in _normalize_token(headline)
        ):
            issues.append(f"문구 {index}의 제목에 매장명이 반복됩니다.")
        if (
            isinstance(store_name, str)
            and _normalize_token(store_name)
            and _normalize_token(store_name) in _normalize_token(body)
        ):
            issues.append(f"문구 {index}의 본문에 매장명이 반복됩니다.")

        product = ad_input.get("product", {})
        prices = (
            product.get("base_price", product.get("price")),
            product.get("sale_price"),
        )
        for price in prices:
            if isinstance(price, str) and price.strip():
                repeated_price_numbers = _number_tokens(body) & _number_tokens(price)
                if repeated_price_numbers or _normalize_token(price) in _normalize_token(body):
                    issues.append(
                        f"문구 {index}의 본문에 별도 표시되는 가격이 반복됩니다."
                    )
                    break

        promotion = ad_input.get("campaign", {}).get("promotion")
        if isinstance(promotion, str) and promotion.strip():
            promotion_numbers = _number_tokens(promotion)
            repeated_headline_numbers = _number_tokens(headline) & promotion_numbers
            repeated_headline_terms = [
                term
                for term in PROMOTION_LABEL_TERMS
                if term in promotion and term in headline
            ]
            if repeated_headline_numbers or repeated_headline_terms:
                issues.append(
                    f"문구 {index}의 제목에 행사 배지 정보가 반복됩니다."
                )
            repeated_body_numbers = _number_tokens(body) & promotion_numbers
            repeated_body_terms = [
                term
                for term in PROMOTION_LABEL_TERMS
                if term in promotion and term in body
            ]
            if repeated_body_numbers or repeated_body_terms:
                issues.append(
                    f"문구 {index}의 본문에 행사 배지 정보가 반복됩니다."
                )

        unexpected_numbers = _number_tokens(combined) - allowed_numbers
        if unexpected_numbers:
            values = ", ".join(sorted(unexpected_numbers))
            issues.append(f"문구 {index}에 입력에 없는 수치({values})가 있습니다.")

        for term in RISKY_CLAIM_TERMS:
            if term.casefold() in combined and term.casefold() not in source_text:
                issues.append(
                    f"문구 {index}에 입력 근거가 없는 표현 '{term}'이 있습니다."
                )

        normalized_combined = _normalize_token(combined)
        normalized_source = _normalize_token(source_text)
        for label, variants in GROUNDED_SENSORY_EXPRESSIONS.items():
            if any(variant in normalized_combined for variant in variants) and not any(
                variant in normalized_source for variant in variants
            ):
                issues.append(
                    f"문구 {index}에 입력 근거가 없는 감각 표현 '{label}'이 있습니다."
                )

        for hashtag in copy["hashtags"]:
            normalized_hashtag = _normalize_token(hashtag.removeprefix("#"))
            if not hashtag.startswith("#"):
                issues.append(f"문구 {index}의 해시태그 '{hashtag}'에 #이 없습니다.")
            elif not normalized_hashtag or normalized_hashtag not in allowed_hashtags:
                issues.append(
                    f"문구 {index}의 해시태그 '{hashtag}'는 입력에서 확인되지 않습니다."
                )

    return list(dict.fromkeys(issues))


def _shorten(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    shortened = normalized[:limit].rstrip()
    if " " not in shortened:
        return shortened
    return shortened.rsplit(" ", 1)[0]


def _fit_fallback_headline(value: str, product_name: str) -> str:
    headline = _shorten(value, MAX_HEADLINE_CHARS)
    if headline.endswith("함"):
        polite_headline = headline[:-1] + "해요"
        headline = (
            polite_headline
            if len(polite_headline) >= MIN_HEADLINE_CHARS
            else f"{headline}이 느껴져요"
        )
    if len(headline) < MIN_HEADLINE_CHARS:
        headline = _shorten(f"{headline} 그대로", MAX_HEADLINE_CHARS)
    if len(headline) < MIN_HEADLINE_CHARS:
        headline = _shorten(product_name, MAX_HEADLINE_CHARS)
    return headline


def _natural_feature_phrase(feature: str, product_name: str | None = None) -> str:
    """입력 메모를 광고 문장에서 바로 쓸 수 있는 명사구로 정리한다."""
    clean_feature = re.sub(r"[.!?。！？]+", "", " ".join(feature.split())).strip(" ,")
    meta_match = re.fullmatch(r"(.+?)(?:이|가) 특징인 .+", clean_feature)
    if meta_match:
        clean_feature = meta_match.group(1).strip()
    return clean_feature or "확인된 핵심 특징"


def _object_particle(value: str) -> str:
    korean_letters = re.findall(r"[가-힣]", value)
    if not korean_letters:
        return "을"
    return "을" if (ord(korean_letters[-1]) - 0xAC00) % 28 else "를"


def _subject_particle(value: str) -> str:
    korean_letters = re.findall(r"[가-힣]", value)
    if not korean_letters:
        return "이"
    return "이" if (ord(korean_letters[-1]) - 0xAC00) % 28 else "가"


def _fallback_body(feature: str) -> str:
    clean_feature = _natural_feature_phrase(feature)
    if any(term in clean_feature for term in ("색", "구성", "형태", "모양")):
        ending = " 한눈에 보여요."
    elif clean_feature.endswith("사용"):
        ingredient = clean_feature.removesuffix("사용").strip()
        ending = f"{_object_particle(ingredient)} 사용해요."
        clean_feature = ingredient
    else:
        ending = f"{_subject_particle(clean_feature)} 그대로 느껴져요."
    suffix = ending
    available = MAX_BODY_CHARS - len(suffix)
    core = _shorten(clean_feature, available)
    if any(term in core for term in ("색", "구성", "형태", "모양")):
        body = f"{core}{_subject_particle(core)} 한눈에 보여요."
    elif clean_feature != core and clean_feature.endswith("사용"):
        body = f"{core}{_object_particle(core)} 사용해요."
    elif ending.endswith("사용해요."):
        body = f"{core}{_object_particle(core)} 사용해요."
    else:
        body = f"{core}{_subject_particle(core)} 그대로 느껴져요."
    return body


def build_fallback_copies(ad_input: dict[str, Any]) -> list[AdCopy]:
    """입력 특징 수에 맞춰 결정론적 기본 광고 문구 한 종을 만든다."""
    product = ad_input.get("product", {})
    product_name = product.get("name") or "상품"
    features = product.get("features")
    feature_parts = split_distinct_feature_facts(
        features if isinstance(features, str) else None
    )

    hashtags = build_deterministic_hashtags(ad_input)

    headline_source = feature_parts[0] if feature_parts else product_name
    headline = _fit_fallback_headline(
        _natural_feature_phrase(headline_source, product_name),
        product_name,
    )
    body = _fallback_body(feature_parts[1]) if len(feature_parts) >= 2 else ""
    return [{"headline": headline, "body": body, "hashtags": hashtags.copy()}]


def _repair_single_product_name_repetition(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> tuple[list[AdCopy], bool]:
    """후보가 하나뿐일 때 상품명 반복 제목을 입력 특징 제목으로 교정한다."""
    if len(copies) != 1:
        return copies, False
    product_name = ad_input.get("product", {}).get("name")
    if not isinstance(product_name, str) or not product_name.strip():
        return copies, False

    copy = copies[0]
    headline_repeats = _contains_product_reference(copy["headline"], product_name)
    body_repeats = _contains_product_reference(copy["body"], product_name)
    if not headline_repeats and not body_repeats:
        return copies, False

    safe_copy = build_fallback_copies(ad_input)[0]
    repaired: AdCopy = {
        "headline": safe_copy["headline"] if headline_repeats else copy["headline"],
        "body": safe_copy["body"] if body_repeats else copy["body"],
        "hashtags": copy["hashtags"],
    }
    if validate_ad_copies([repaired], ad_input):
        return copies, False
    return [repaired], True


def _complete_required_fact_split(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> tuple[list[AdCopy], int]:
    """독립 사실이 둘 이상인데 본문이 비었으면 입력 기반 한 쌍으로 보완한다."""
    features = ad_input.get("product", {}).get("features")
    facts = split_distinct_feature_facts(
        features if isinstance(features, str) else None
    )
    if len(facts) < 2:
        return copies, 0

    safe_copy = build_fallback_copies(ad_input)[0]
    completed: list[AdCopy] = []
    repaired_count = 0
    signatures: set[tuple[str, str]] = set()
    for copy in copies:
        candidate = copy
        if not copy["body"].strip():
            candidate = safe_copy
            if not validate_ad_copies([candidate], ad_input):
                repaired_count += 1
            else:
                candidate = copy
        signature = (
            _normalize_token(candidate["headline"]),
            _normalize_token(candidate["body"]),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        completed.append(candidate)
    return completed, repaired_count


def generate_safe_ad_copies(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> SafeGenerationResult:
    """모델 결과를 검증하고 한 번 재시도한 뒤 필요하면 기본 문구를 반환한다."""
    if max_attempts < 1:
        raise ValueError("max_attempts는 1 이상이어야 합니다.")

    retry_feedback: str | None = None
    last_issues: tuple[str, ...] = ()
    failure_kind = "error"
    for attempt in range(1, max_attempts + 1):
        try:
            copies = generate_ad_copies(
                ad_input,
                model_id=model_id,
                api_key=api_key,
                client=client,
                retry_feedback=retry_feedback,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
        except Exception:
            failure_kind = "error"
            retry_feedback = "이전 모델 요청이 완료되지 않았습니다."
            continue

        valid_model_copies: list[AdCopy] = []
        rejected_issues: list[str] = []
        signatures: set[tuple[str, str]] = set()
        omitted_body_count = 0
        for copy in copies:
            candidate_issues = validate_ad_copies([copy], ad_input)
            if candidate_issues and copy["body"].strip():
                headline_only_copy: AdCopy = {
                    "headline": copy["headline"],
                    "body": "",
                    "hashtags": copy["hashtags"],
                }
                if not validate_ad_copies([headline_only_copy], ad_input):
                    rejected_issues.extend(candidate_issues)
                    copy = headline_only_copy
                    candidate_issues = []
                    omitted_body_count += 1
            if candidate_issues:
                rejected_issues.extend(candidate_issues)
                continue
            signature = (
                _normalize_token(copy["headline"]),
                _normalize_token(copy["body"]),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            valid_model_copies.append(copy)

        if valid_model_copies:
            valid_model_copies, fact_split_repaired_count = (
                _complete_required_fact_split(valid_model_copies, ad_input)
            )
            valid_model_copies, product_name_repaired = (
                _repair_single_product_name_repetition(
                    valid_model_copies,
                    ad_input,
                )
            )
            rejected_tuple = tuple(dict.fromkeys(rejected_issues))
            if attempt == 1:
                notices: list[str] = []
                if omitted_body_count:
                    notices.append(
                        f"문제 있는 본문 {omitted_body_count}종을 생략했습니다."
                    )
                if product_name_repaired:
                    notices.append(
                        "상품명이 반복된 제목을 입력 특징 기반 제목으로 교정했습니다."
                    )
                if fact_split_repaired_count:
                    notices.append(
                        "서로 다른 입력 사실을 제목과 본문으로 분리했습니다."
                    )
                if rejected_tuple and not notices:
                    notices.append(
                        f"검증을 통과한 모델 문구 {len(valid_model_copies)}종을 사용했습니다."
                    )
                return SafeGenerationResult(
                    valid_model_copies,
                    "mixed"
                    if product_name_repaired or fact_split_repaired_count
                    else "generated",
                    attempt,
                    notice=" ".join(notices) or None,
                    validation_issues=rejected_tuple,
                    model_copy_count=(
                        0
                        if product_name_repaired or fact_split_repaired_count
                        else len(valid_model_copies)
                    ),
                    fallback_copy_count=(
                        len(valid_model_copies)
                        if product_name_repaired or fact_split_repaired_count
                        else 0
                    ),
                )
            notice = (
                "입력 근거 검증 후 문구를 다시 생성했습니다."
                if last_issues
                else "일시적인 모델 오류 후 문구를 다시 생성했습니다."
            )
            if product_name_repaired:
                notice += " 상품명이 반복된 제목은 입력 특징 기반으로 교정했습니다."
            if fact_split_repaired_count:
                notice += " 서로 다른 입력 사실을 제목과 본문으로 분리했습니다."
            return SafeGenerationResult(
                valid_model_copies,
                "mixed"
                if product_name_repaired or fact_split_repaired_count
                else "retried",
                attempt,
                notice=notice,
                validation_issues=last_issues,
                model_copy_count=(
                    0
                    if product_name_repaired or fact_split_repaired_count
                    else len(valid_model_copies)
                ),
                fallback_copy_count=(
                    len(valid_model_copies)
                    if product_name_repaired or fact_split_repaired_count
                    else 0
                ),
            )

        failure_kind = "validation"
        last_issues = tuple(dict.fromkeys(rejected_issues))
        retry_feedback = " ".join(last_issues)

    notice = (
        "문구가 품질·근거 검증을 반복 통과하지 못해 안전한 기본 문구를 사용했습니다."
        if failure_kind == "validation"
        else "모델 호출이 완료되지 않아 안전한 기본 문구를 사용했습니다."
    )
    fallback_copies = build_fallback_copies(ad_input)
    return SafeGenerationResult(
        fallback_copies,
        "fallback",
        max_attempts,
        notice=notice,
        validation_issues=last_issues,
        model_copy_count=0,
        fallback_copy_count=len(fallback_copies),
    )
