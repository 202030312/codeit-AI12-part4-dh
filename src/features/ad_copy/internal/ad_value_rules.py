"""광고성 우선순위와 한국어 시각 재추출을 적용한 문구 생성 v1.10.3."""

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
    parse_request_dimensions,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.base_workflow as base
import src.features.ad_copy.visual_analysis as visual_base
import src.features.ad_copy.internal.structured_workflow as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.3-ad-value-korean-visual-retry"
VISUAL_PROMPT_VERSION = "gemini_visual_facts_v1.1.0-korean-retry"


BANNER_COPY_SYSTEM_INSTRUCTIONS = previous.BANNER_COPY_SYSTEM_INSTRUCTIONS.replace(
    "[자동 구성]",
    """[광고 제목 우선순위]

1. 사용자가 요청한 이용 상황·감성 메시지가 있으면 이를 가장 먼저 사용하세요.
2. 다음으로 verified_product_facts의 상품 장점을 사용하세요.
3. verified_visual_facts는 수량·제품 구성·구별되는 토핑처럼 고객에게 의미 있는 정보일 때만 사용하세요.
4. 컵·잔·접시·테이블·일반적인 색상·단순 도형만으로 제목을 만들지 마세요.
5. '원통형 컵', '갈색 음료', '음료 1잔', '테이블 위에 놓인 잔' 같은 사진 설명형 제목을 만들지 마세요.
6. 제목을 사진 관찰 보고서가 아니라 짧고 자연스러운 광고 메시지로 작성하세요.

[자동 구성]""",
).replace(
    "verified_visual_facts에 없는 담긴·올린·얹은·위에·아래에 같은 배치 표현을 만들지 마세요.",
    "verified_visual_facts에 없는 담긴·올린·얹은·위에·아래에 같은 배치 표현을 만들지 마세요. 특히 검증 실패 사유에 '담긴'이 있으면 repair에서 형태를 바꿔 반복하지 말고 상품 사실이나 요청 콘셉트로 완전히 교체하세요.",
).replace(
    "8. 질리지 않는·손이 가는·계속 찾는·재구매·인기 메뉴·누구나 좋아하는·중독되는·후회 없는 등 고객 반응·반복 행동·인기도·만족도 주장을 만들지 마세요.",
    "8. 질리지 않는·손이 가는·계속 찾는·재구매·인기 메뉴·누구나 좋아하는·중독되는·후회 없는 등 고객 반응·반복 행동·인기도·만족도 주장을 만들지 마세요.\n9. 입력 근거 없이 정통·진수·명품·격이 다른·프리미엄 같은 권위·우월성 표현을 만들지 마세요.",
)


_LATIN_TEXT_PATTERN = re.compile(r"[A-Za-z]{2,}")
_LOW_VALUE_EXACT_PATTERNS = (
    re.compile(r"^(?:[가-힣]+색\s*)?(?:원통형|구\s*형태|사각형|직육면체|부채꼴(?:\s*모양)?)(?:\s*(?:컵|잔|조각|형태|음료|디저트|빵))?$"),
    re.compile(r"^(?:구|원통|사각|직육면체|부채꼴|조개껍데기).*형태.*$"),
    re.compile(r"^(?:[가-힣]+색\s*)?(?:음료|컵|잔)\s*(?:1(?:개|잔)?)?$"),
    re.compile(r"^테이블\s*(?:위|위에).*(?:컵|잔)$"),
    re.compile(r"^(?:유리)?잔의\s*휘핑\s*크림$"),
    re.compile(r"^부채꼴.*(?:조각\s*)?형태$"),
    re.compile(r"^액체와\s*.+$"),
    re.compile(r"^(?:상단|하단)(?:이|가)?\s*보이는\s*.+$"),
    re.compile(r"^(?:갈색|흰색|하얀색|검은색|노란색|초록색|붉은색|보라색)(?:이|가)?\s*보이는\s*.+$"),
    re.compile(r"^.*(?:둥근|원형|사각형)\s*\d+개$"),
)
_GENERIC_PRODUCT_TYPES = {
    "음료", "디저트", "빵", "베이커리", "제품", "상품", "메뉴", "한잔", "한개",
}
_GENERIC_COMPONENT_TERMS = {
    "액체", "내용물", "토핑", "장식", "조각", "형태", "표면", "휘핑크림",
    "상단", "하단", "내부", "코팅", "갈색", "흰색", "하얀색", "노란색",
    "다진", "말린", "놓인", "잘린", "얹힌",
}
_GENERIC_FACT_TERMS = {
    "조합", "구성", "관계", "풍미", "식감", "향", "빵결", "메뉴", "음료", "디저트",
}
_SENSORY_ROOTS = ("맛", "향", "풍미", "식감", "빵 결", "빵결")
_UNVERIFIED_PRESTIGE_PATTERN = re.compile(
    r"(?:정통|진수|명품|격이\s*다른|프리미엄)"
)


def _visual_facts(ad_input: dict[str, Any]) -> list[dict[str, str]]:
    return previous._visual_facts(ad_input)


def _prompt_payload(profile: Any, ad_input: dict[str, Any]) -> dict[str, Any]:
    payload = previous._prompt_payload(profile, ad_input)
    payload["headline_source_priority"] = [
        "사용자가 명시한 이용 상황·감성 메시지",
        "verified_product_facts",
        "의미 있는 verified_visual_facts: 수량·제품 구성·구별되는 토핑",
    ]
    payload["low_ad_value_visuals"] = {
        "do_not_use_alone": [
            "컵·잔·접시·테이블",
            "일반적인 색상",
            "단순 도형·형태",
            "1개·1잔처럼 정보 가치가 낮은 수량",
        ],
        "allowed_when_meaningful": [
            "마카롱 5개",
            "블루베리 3개",
            "여러 구성 요소가 함께 보이는 경우",
        ],
    }
    payload["repair_policy"] = (
        "검증 실패 표현을 어형만 바꿔 반복하지 말고 다른 근거와 관점으로 교체"
    )
    return payload


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    profile = analyze_copy_input(ad_input, image_provided=False)
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "상품 장점과 요청 콘셉트를 우선해 서로 다른 광고 문구 선택지 2개를 작성하세요. "
        "시각 사실은 고객에게 의미 있는 구체적 정보일 때만 보조적으로 사용하세요.\n"
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
            "실패한 표현과 같은 단어·어형·사진 묘사를 반복하지 않음",
            "사용자가 요청한 상황·감성 콘셉트 사용",
            "verified_product_facts의 상품 장점 사용",
            "수량·구성처럼 의미 있는 시각 사실만 사용",
            "컵·잔·접시·테이블·일반 색상·단순 형태만으로 제목을 만들지 않음",
            "locked_options와 다른 관점 사용",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "실패한 선택지만 완전히 다른 안전한 관점으로 교체하세요. locked_options는 유지하고 "
        "replace_option_indices 순서의 교체 선택지만 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _campaign_phrases(copy_request: str) -> list[str]:
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.strip(" ,.").split())
        if 3 <= len(cleaned) <= 24 and _normalize_compact(cleaned) not in {
            _normalize_compact(item) for item in values
        }:
            values.append(cleaned)

    for pattern in (
        r"(?:^|[,。.]\s*)([^,。.]{2,30}?)(?:을|를)\s*강조",
        r"제목에는\s*([^,。.]{2,30}?)(?:을|를)\s*표현",
    ):
        for match in re.finditer(pattern, copy_request):
            add(match.group(1))
    dimensions = parse_request_dimensions(copy_request)
    for key in ("usage_context", "emotional_message"):
        for value in dimensions[key]:
            add(str(value))
    return values


def _strong_product_or_campaign_grounding(
    text: str,
    ad_input: dict[str, Any],
) -> bool:
    compact = _normalize_compact(text)
    profile = analyze_copy_input(ad_input)
    for fact in profile.verified_facts:
        fact_compact = _normalize_compact(fact)
        if fact_compact and fact_compact in compact:
            return True
        sensory = next((root for root in _SENSORY_ROOTS if root.replace(" ", "") in fact_compact), None)
        if sensory and sensory.replace(" ", "") not in compact:
            continue
        terms = {
            term
            for term in _terms(fact)
            if len(term) >= 2 and _normalize_compact(term) not in _GENERIC_FACT_TERMS
        }
        matched = sum(_normalize_compact(term) in compact for term in terms)
        if terms and matched >= min(2, len(terms)):
            return True
    return any(
        _normalize_compact(phrase) in compact
        for phrase in _campaign_phrases(profile.copy_request)
    )


def _fact_is_reflected(text: str, value: str, category: str) -> bool:
    compact = _normalize_compact(text)
    value_compact = _normalize_compact(value)
    if value_compact and value_compact in compact:
        return True
    if category == "count":
        number = re.search(r"\d+", value)
        return bool(number and number.group() in text)
    return any(
        len(term) >= 2 and _normalize_compact(term) in compact
        for term in _terms(value)
    )


def _meaningful_visual_score(text: str, ad_input: dict[str, Any]) -> int:
    score = 0
    for fact in _visual_facts(ad_input):
        category = str(fact.get("category") or "")
        value = str(fact.get("value") or "")
        if not _fact_is_reflected(text, value, category):
            continue
        if category == "count":
            number = re.search(r"\d+", value)
            if number and int(number.group()) >= 2:
                score += 2
        elif category == "product_type":
            if _normalize_compact(value) not in _GENERIC_PRODUCT_TYPES:
                score += 1
        elif category == "visible_component":
            terms = {
                _normalize_compact(term)
                for term in _terms(value)
                if len(term) >= 2
                and _normalize_compact(term) not in _GENERIC_COMPONENT_TERMS
            }
            if any(term in _normalize_compact(text) for term in terms):
                score += 1
        elif category == "packaging":
            score += 1
    return score


def _product_entity_score(text: str, ad_input: dict[str, Any]) -> int:
    compact = _normalize_compact(text)
    profile = analyze_copy_input(ad_input)
    for fact in profile.verified_facts:
        for term in _terms(fact):
            normalized = _normalize_compact(term)
            if (
                len(term) >= 2
                and normalized not in _GENERIC_FACT_TERMS
                and normalized in compact
            ):
                return 1
    return 0


def _is_low_ad_value_visual_title(text: str, ad_input: dict[str, Any]) -> bool:
    stripped = " ".join(text.split())
    if any(pattern.fullmatch(stripped) for pattern in _LOW_VALUE_EXACT_PATTERNS):
        return True
    visual_reflected = any(
        _fact_is_reflected(stripped, str(fact.get("value") or ""), str(fact.get("category") or ""))
        for fact in _visual_facts(ad_input)
    )
    return bool(
        visual_reflected
        and not _strong_product_or_campaign_grounding(stripped, ad_input)
        and (
            _meaningful_visual_score(stripped, ad_input)
            + _product_entity_score(stripped, ad_input)
        ) < 1
    )


def _has_visual_angle(text: str, ad_input: dict[str, Any]) -> bool:
    compact = _normalize_compact(text)
    return bool(
        any(marker in compact for marker in ("보이는", "속", "위", "아래", "옆", "개", "잔"))
        and any(
            _fact_is_reflected(
                text,
                str(fact.get("value") or ""),
                str(fact.get("category") or ""),
            )
            for fact in _visual_facts(ad_input)
        )
    )


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    profile = analyze_copy_input(ad_input)
    issues = []
    for issue in previous.collect_banner_copy_issues(copies, ad_input):
        if (
            issue.code == "HEADLINE_NOT_GROUNDED"
            and profile.generation_mode == "headline_and_subcopy"
        ):
            continue
        if (
            issue.code == "LOW_CANDIDATE_DIVERSITY"
            and len(copies) == 2
            and _has_visual_angle(copies[0]["headline"], ad_input)
            != _has_visual_angle(copies[1]["headline"], ad_input)
            and not any(
                _is_low_ad_value_visual_title(copy["headline"], ad_input)
                for copy in copies
            )
        ):
            continue
        issues.append(issue)
    for index, copy in enumerate(copies, start=1):
        for field, text in (("headline", copy["headline"]), ("subcopy", copy["body"])):
            if (
                text
                and _UNVERIFIED_PRESTIGE_PATTERN.search(text)
                and not _UNVERIFIED_PRESTIGE_PATTERN.search(profile.verified_facts_text)
            ):
                issues.append(CopyValidationIssue(
                    "UNVERIFIED_PRESTIGE_CLAIM",
                    f"선택지 {index} {field}에 입력 근거 없는 권위·우월성 표현이 있습니다.",
                    index,
                    field,
                ))
        if _is_low_ad_value_visual_title(copy["headline"], ad_input):
            issues.append(CopyValidationIssue(
                "LOW_AD_VALUE_VISUAL_TITLE",
                f"선택지 {index} 제목이 상품 장점보다 일반적인 사진 묘사에 머뭅니다.",
                index,
                "headline",
            ))
        if re.search(r"(?:형태|모양)의\s*\d+(?:개|구)$", copy["headline"].strip()):
            issues.append(CopyValidationIssue(
                "INCOMPLETE_COUNT_PHRASE",
                f"선택지 {index} 제목의 수량 앞에 실제 상품 대상이 없습니다.",
                index,
                "headline",
            ))
        if re.search(r"(?:이|가|을|를|은|는)$", copy["headline"].strip()):
            issues.append(CopyValidationIssue(
                "INCOMPLETE_HEADLINE",
                f"선택지 {index} 제목이 조사로 끝나 문장이 완결되지 않았습니다.",
                index,
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


def _subject_particle(value: str) -> str:
    if not value:
        return "이"
    code = ord(value[-1]) - 0xAC00
    return "이" if 0 <= code <= 11171 and code % 28 else "가"


def _object_particle(value: str) -> str:
    if not value:
        return "을"
    code = ord(value[-1]) - 0xAC00
    return "을" if 0 <= code <= 11171 and code % 28 else "를"


def _and_particle(value: str) -> str:
    if not value:
        return "과"
    code = ord(value[-1]) - 0xAC00
    return "과" if 0 <= code <= 11171 and code % 28 else "와"


def _specific_fact_terms(ad_input: dict[str, Any]) -> list[str]:
    profile = analyze_copy_input(ad_input)
    values: list[str] = []
    for fact in profile.verified_facts:
        adjective_roots = {
            _normalize_compact(value)
            for value in re.findall(r"([가-힣]+?)한(?:\s|$)", fact)
        }
        for term in _terms(fact):
            normalized = _normalize_compact(term)
            if (
                len(term) >= 2
                and normalized not in _GENERIC_FACT_TERMS
                and normalized not in adjective_roots
                and not term.endswith(("한", "로운"))
                and term not in values
            ):
                values.append(term)
    return values


def _fact_variants(fact: str) -> list[str]:
    values: list[str] = []
    possessive_flavor = re.fullmatch(r"(.+?)의\s+(.+?)한\s+풍미", fact)
    if possessive_flavor:
        ingredient, modifier = possessive_flavor.groups()
        return [f"{ingredient}의 {modifier}함", f"{modifier}한 {ingredient} 풍미"]
    combination = re.fullmatch(r"(.+?)(와|과)(.+?)의\s*조합", fact)
    if combination:
        left, connector, right = combination.groups()
        values.append(f"{left}{connector} {right}")
        values.append(f"{right}{_and_particle(right)} {left}의 조화")
    adjective = re.fullmatch(r"(.+?)한\s+(.+?)\s+(풍미|식감)", fact)
    if adjective:
        modifier, subject, _ = adjective.groups()
        values.append(f"{modifier}함이 돋보이는 {subject}")
        values.append(f"{subject}{_object_particle(subject)} 중심으로")
    simple_adjective = re.fullmatch(r"(.+?)한\s+(풍미|식감)", fact)
    if simple_adjective:
        values.append(f"{simple_adjective.group(1)}함을 중심으로")
    sensory_subject = re.fullmatch(r"(.+?)\s+(풍미|향)", fact)
    if sensory_subject:
        subject = sensory_subject.group(1).removesuffix("의").strip()
        values.append(f"{subject}{_object_particle(subject)} 중심으로")
        values.append(f"{subject}{_subject_particle(subject)} 돋보이는 메뉴")
    if re.search(r"빵\s*결", fact):
        values.append("빵 결을 중심으로")
        values.append("결이 돋보이는 빵")
    paired_facts = re.fullmatch(r"(.+?)(와|과)(.+)", fact)
    if paired_facts and not fact.endswith("의 조합"):
        values.extend([paired_facts.group(1), paired_facts.group(3)])
    return values


def _meaningful_visual_titles(ad_input: dict[str, Any], limit: int) -> list[str]:
    values: list[str] = []

    def add(value: str) -> None:
        fitted = previous._fit_words(value, limit)
        if (
            len(fitted) >= 4
            and " " in fitted
            and not _is_low_ad_value_visual_title(fitted, ad_input)
            and _normalize_compact(fitted) not in {_normalize_compact(item) for item in values}
        ):
            values.append(fitted)

    for value in previous._visual_fallback_titles(ad_input, limit):
        if (
            _meaningful_visual_score(value, ad_input)
            + _product_entity_score(value, ad_input)
        ) >= 2:
            add(value)

    fact_terms = _specific_fact_terms(ad_input)
    visible_terms: list[str] = []
    visible_values: list[str] = []
    containers: list[str] = []
    product_types: list[str] = []
    arrangements: list[str] = []
    for fact in _visual_facts(ad_input):
        category = str(fact.get("category") or "")
        value = str(fact.get("value") or "")
        if category == "product_type" and 2 <= len(value) <= 10:
            product_types.append(value)
        if category == "container" and 2 <= len(value) <= 10:
            containers.append(value)
        if category == "arrangement":
            arrangements.append(value)
        if category == "visible_component":
            if 2 <= len(value) <= 10 and value not in visible_values:
                visible_values.append(value)
            for term in _terms(value):
                normalized = _normalize_compact(term)
                if (
                    2 <= len(term) <= 8
                    and normalized not in _GENERIC_COMPONENT_TERMS
                    and not term.endswith(("진", "린", "힌"))
                    and term not in visible_terms
                ):
                    visible_terms.append(term)
    if len(visible_values) >= 2:
        add(
            f"{visible_values[0]}{_and_particle(visible_values[0])} "
            f"{visible_values[1]}"
        )
    overlapping_entities = [
        entity
        for entity in fact_terms
        if any(
            _normalize_compact(entity) in _normalize_compact(value)
            for value in visible_terms
        )
    ]
    for entity in overlapping_entities:
        related_container = next(
            (
                container
                for container in containers
                if any(
                    _normalize_compact(container) in _normalize_compact(arrangement)
                    and _normalize_compact(entity) in _normalize_compact(arrangement)
                    for arrangement in arrangements
                )
            ),
            None,
        )
        if related_container:
            add(f"{related_container} 속 {entity}")
        elif product_types:
            add(f"{entity}{_subject_particle(entity)} 보이는 {product_types[0]}")
    if not overlapping_entities:
        for entity in fact_terms:
            if visible_terms:
                component = visible_terms[0]
                add(f"{component}{_subject_particle(component)} 보이는 {entity}")
                break
    return values


def _fallback_title_candidates(ad_input: dict[str, Any]) -> list[str]:
    profile = analyze_copy_input(ad_input)
    values: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        fitted = previous._fit_words(value, profile.limits.headline)
        if len(fitted) < 4 or previous._REPEATED_WORD_PATTERN.search(fitted):
            return
        if _normalize_compact(fitted) not in {_normalize_compact(item) for item in values}:
            values.append(fitted)

    for phrase in _campaign_phrases(profile.copy_request):
        add(phrase)
    for fact in profile.verified_facts:
        add(fact)
    for fact in profile.verified_facts:
        for variant in _fact_variants(fact):
            add(variant)
    for value in _meaningful_visual_titles(ad_input, profile.limits.headline):
        add(value)
    for fact in profile.verified_facts:
        adjective_flavor = re.fullmatch(r"(.+?)한\s+풍미", fact)
        if adjective_flavor:
            add(f"{adjective_flavor.group(1)}함을 중심으로")
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
    return previous._safe_fallback_options(ad_input)


def _resolve_partial_fallback(
    ad_input: dict[str, Any],
    copies: list[AdCopy] | None,
    issues: list[CopyValidationIssue],
) -> tuple[list[AdCopy], tuple[int, ...]]:
    fallback = _safe_fallback_options(ad_input)
    if copies is None:
        return fallback, (1, 2)
    invalid_indices = {
        issue.option_index for issue in issues if issue.option_index is not None
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
    """상품 장점 우선 문구를 만들고 실패 후보만 안전하게 교체한다."""

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
            issue.option_index for issue in issues if issue.option_index is not None
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
                output_text, ad_input, expected_count=option_count
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


VISUAL_FACT_SYSTEM_INSTRUCTIONS = visual_base.VISUAL_FACT_SYSTEM_INSTRUCTIONS.replace(
    "6. 하나의 observation에는 한 가지 짧은 사실만 넣으세요.",
    "6. 하나의 observation에는 한 가지 짧은 사실만 넣으세요.\n7. category와 confidence 값 외의 모든 value 및 excluded_inferences는 반드시 자연스러운 한국어로만 작성하세요. 영문 단어와 영문 문장을 사용하지 마세요.",
)


def _filter_korean_visual_result(
    result: visual_base.VisualFactExtractionResult,
) -> visual_base.VisualFactExtractionResult:
    accepted: list[dict[str, str]] = []
    rejected = [dict(item) for item in result.rejected_observations]
    for fact in result.verified_facts:
        value = str(fact.get("value") or "")
        if _LATIN_TEXT_PATTERN.search(value):
            rejected.append({**dict(fact), "reason": "한국어가 아닌 시각 사실"})
        else:
            accepted.append({**dict(fact), "id": f"v{len(accepted) + 1}"})
    excluded = tuple(
        value
        for value in result.excluded_inferences
        if not _LATIN_TEXT_PATTERN.search(str(value))
    )
    return visual_base.VisualFactExtractionResult(
        verified_facts=tuple(accepted),
        rejected_observations=tuple(rejected),
        excluded_inferences=excluded,
        status="extracted" if accepted else "empty",
        raw_output=result.raw_output,
        notice=None if accepted else "한국어 고신뢰 시각 정보를 찾지 못했습니다.",
        prompt_version=VISUAL_PROMPT_VERSION,
    )


def extract_verified_visual_facts(
    *,
    menu_name: str,
    model_id: str,
    api_key: str,
    image_bytes: bytes,
    image_mime_type: str,
    client: Any | None = None,
) -> visual_base.VisualFactExtractionResult:
    """한국어 시각 사실을 요청하고 영문이 나오면 한 번만 재추출한다."""

    first_result: visual_base.VisualFactExtractionResult | None = None
    last_error: Exception | None = None
    for attempt in range(1, 3):
        prompt = visual_base.build_visual_fact_prompt(menu_name)
        if attempt == 2:
            prompt += (
                "\n이전 응답에 영문 시각 사실이 포함되었습니다. 같은 이미지를 다시 관찰해 "
                "value와 excluded_inferences를 모두 한국어로만 반환하세요. 영문을 번역해 추측하지 말고 이미지에서 직접 확인하세요."
            )
        try:
            output_text = _generate_with_gemini(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                system_instructions=VISUAL_FACT_SYSTEM_INSTRUCTIONS,
                response_schema=visual_base.VISUAL_FACT_RESPONSE_SCHEMA,
            )
            parsed = visual_base.parse_visual_fact_response(output_text)
            parsed = visual_base.VisualFactExtractionResult(
                verified_facts=parsed.verified_facts,
                rejected_observations=parsed.rejected_observations,
                excluded_inferences=parsed.excluded_inferences,
                status=parsed.status,
                raw_output=parsed.raw_output,
                notice=parsed.notice,
                prompt_version=VISUAL_PROMPT_VERSION,
            )
            filtered = _filter_korean_visual_result(parsed)
            if first_result is None:
                first_result = filtered
            has_latin = any(
                _LATIN_TEXT_PATTERN.search(str(fact.get("value") or ""))
                for fact in parsed.verified_facts
            )
            if not has_latin:
                return filtered
        except Exception as exc:
            last_error = exc
    if first_result is not None:
        return first_result
    return visual_base.VisualFactExtractionResult(
        verified_facts=(),
        rejected_observations=(),
        excluded_inferences=(),
        status="omitted",
        notice=f"시각 정보 추출을 생략했습니다: {type(last_error).__name__}: {last_error}",
        prompt_version=VISUAL_PROMPT_VERSION,
    )


attach_visual_context = visual_base.attach_visual_context

