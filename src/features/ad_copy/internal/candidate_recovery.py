"""하드 안전성 검증과 광고성 순위를 분리한 문구 생성 v1.10.4."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    _generate_with_gemini,
)
from src.features.ad_copy.internal.base_rules import (
    _normalize_compact,
    title_lexical_similarity,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.base_workflow as base
import src.features.ad_copy.internal.structured_workflow as structured
import src.features.ad_copy.internal.ad_value_rules as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.4-hard-safety-ranked-four"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = 4
FINAL_OPTION_COUNT = 2


BANNER_COPY_SYSTEM_INSTRUCTIONS = previous.BANNER_COPY_SYSTEM_INSTRUCTIONS.replace(
    "구조화된 입력만 근거로 제목 또는 제목+보조 문구 선택지 2개를 작성하세요.",
    "구조화된 입력만 근거로 제목 또는 제목+보조 문구 내부 후보 4개를 작성하세요.",
).replace(
    "최초 생성은 선택지 2개를 반환합니다.",
    "최초 생성과 재시도는 내부 후보 4개를 반환합니다.",
) + """

[v1.10.4 후보 생성 규칙]

1. 서로 다른 관점의 내부 후보를 정확히 4개 작성하세요. 앱은 안전성 검증과 광고성 점수로 상위 2개만 선택합니다.
2. 네 후보의 제목을 어순이나 종결 표현만 바꾸지 말고, 요청 콘셉트·상품 장점·의미 있는 시각 구성 중 강조점을 달리하세요.
3. 제목과 보조 문구에 같은 핵심 상품명이 등장하는 것은 허용합니다. 보조 문구가 제목과 완전히 같거나 아무 정보도 더하지 않는 경우만 피하세요.
4. 내부 순위나 점수는 출력하지 말고 기존 JSON 구조의 options 배열에 후보 4개만 반환하세요.
"""


# 아래 항목은 사실 안전성 위반이 아니라 후보 간 비교에 사용하는 품질 신호다.
QUALITY_ONLY_CODES = frozenset({
    "LOW_AD_VALUE_VISUAL_TITLE",
    "LOW_CANDIDATE_DIVERSITY",
    "HEADLINE_SUBCOPY_REPETITION",
    "REPEATED_WORD",
    "REPEATED_COLOR",
})


@dataclass(frozen=True)
class RankedCandidate:
    """하드 검증을 통과한 내부 후보와 광고성 평가 근거."""

    copy: AdCopy
    candidate_index: int
    attempt: int
    score: int
    flags: tuple[str, ...]
    score_breakdown: dict[str, int]


def _deduplicate_issues(
    issues: list[CopyValidationIssue],
) -> list[CopyValidationIssue]:
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
    """최종 노출을 막아야 하는 하드 안전성·형식 문제만 반환한다."""

    return _deduplicate_issues([
        issue
        for issue in previous.collect_banner_copy_issues(copies, ad_input)
        if issue.code not in QUALITY_ONLY_CODES
    ])


def _candidate_hard_issues(
    copy: AdCopy,
    ad_input: dict[str, Any],
    *,
    candidate_index: int,
) -> list[CopyValidationIssue]:
    # 기존 검증기는 최종 2개를 전제로 한다. 같은 후보를 두 자리에 놓고 첫 번째
    # 후보의 로컬 문제만 가져오면 검증 규칙을 복제하지 않고 4개를 독립 평가할 수 있다.
    issues = previous.collect_banner_copy_issues([copy, dict(copy)], ad_input)
    return [
        CopyValidationIssue(
            issue.code,
            issue.message.replace("선택지 1", f"내부 후보 {candidate_index}"),
            candidate_index,
            issue.field,
        )
        for issue in issues
        if issue.option_index == 1 and issue.code not in QUALITY_ONLY_CODES
    ]


def _has_excessive_headline_subcopy_repetition(copy: AdCopy) -> bool:
    """핵심 명사 공유는 허용하고 사실상 같은 문장인 경우만 반복으로 본다."""

    headline = copy["headline"].strip()
    body = copy["body"].strip()
    if not headline or not body:
        return False
    if _normalize_compact(headline) == _normalize_compact(body):
        return True
    return title_lexical_similarity(headline, body) >= 0.95


def _candidate_quality_flags(copy: AdCopy, ad_input: dict[str, Any]) -> tuple[str, ...]:
    flags: list[str] = []
    if previous._is_low_ad_value_visual_title(copy["headline"], ad_input):
        flags.append("low_ad_value_visual_title")
    if _has_excessive_headline_subcopy_repetition(copy):
        flags.append("excessive_headline_subcopy_repetition")
    if structured._REPEATED_WORD_PATTERN.search(copy["headline"]):
        flags.append("repeated_word")
    if structured._COLOR_ECHO_PATTERN.search(copy["headline"]):
        flags.append("repeated_color")
    return tuple(flags)


def score_ad_candidate(
    copy: AdCopy,
    ad_input: dict[str, Any],
) -> tuple[int, tuple[str, ...], dict[str, int]]:
    """안전성 합격 후보를 광고 메시지로서 비교하는 결정론적 점수."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    combined = f"{copy['headline']} {copy['body']}".strip()
    compact = _normalize_compact(combined)
    breakdown: dict[str, int] = {}

    campaign_matches = sum(
        _normalize_compact(phrase) in compact
        for phrase in previous._campaign_phrases(profile.copy_request)
    )
    if campaign_matches:
        breakdown["requested_concept"] = min(24, campaign_matches * 12)
    if previous._strong_product_or_campaign_grounding(copy["headline"], ad_input):
        breakdown["headline_grounding"] = 24
    if copy["body"] and previous._strong_product_or_campaign_grounding(copy["body"], ad_input):
        breakdown["subcopy_grounding"] = 10

    meaningful_visual = previous._meaningful_visual_score(copy["headline"], ad_input)
    if meaningful_visual:
        breakdown["meaningful_visual"] = min(12, meaningful_visual * 6)
    if 8 <= len(copy["headline"]) <= 15:
        breakdown["headline_length"] = 8
    elif 4 <= len(copy["headline"]) <= profile.limits.headline:
        breakdown["headline_length"] = 4

    flags = _candidate_quality_flags(copy, ad_input)
    penalties = {
        "low_ad_value_visual_title": -36,
        "excessive_headline_subcopy_repetition": -14,
        "repeated_word": -8,
        "repeated_color": -8,
    }
    for flag in flags:
        breakdown[flag] = penalties[flag]
    return sum(breakdown.values()), flags, breakdown


def _rank_candidates(
    candidates: list[AdCopy],
    ad_input: dict[str, Any],
    *,
    attempt: int,
) -> tuple[list[RankedCandidate], list[dict[str, Any]]]:
    valid: list[RankedCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for index, copy in enumerate(candidates, start=1):
        hard_issues = _candidate_hard_issues(
            copy,
            ad_input,
            candidate_index=index,
        )
        score, flags, breakdown = score_ad_candidate(copy, ad_input)
        diagnostics.append({
            "candidate_index": index,
            "headline": copy["headline"],
            "subcopy": copy["body"],
            "hard_issues": [issue.message for issue in hard_issues],
            "ad_score": score,
            "quality_flags": list(flags),
            "score_breakdown": breakdown,
        })
        if not hard_issues:
            valid.append(RankedCandidate(
                copy=dict(copy),
                candidate_index=index,
                attempt=attempt,
                score=score,
                flags=flags,
                score_breakdown=breakdown,
            ))
    return valid, diagnostics


def select_top_two_candidates(
    candidates: list[RankedCandidate],
    ad_input: dict[str, Any],
) -> tuple[list[AdCopy], list[dict[str, Any]]]:
    """개별 광고성 점수와 두 제목의 관점 차이를 함께 고려한다."""

    best: tuple[int, RankedCandidate, RankedCandidate, float, int] | None = None
    for left, right in combinations(candidates, 2):
        if _normalize_compact(left.copy["headline"]) == _normalize_compact(right.copy["headline"]):
            continue
        similarity = title_lexical_similarity(
            left.copy["headline"],
            right.copy["headline"],
        )
        diversity_adjustment = 8
        if similarity >= 0.85:
            diversity_adjustment = -24
        elif similarity >= 0.5:
            diversity_adjustment = -10
        elif previous._has_visual_angle(left.copy["headline"], ad_input) != previous._has_visual_angle(
            right.copy["headline"], ad_input
        ):
            diversity_adjustment = 12
        pair_score = left.score + right.score + diversity_adjustment
        candidate = (pair_score, left, right, similarity, diversity_adjustment)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return [], []
    _, left, right, similarity, diversity_adjustment = best
    selected = sorted((left, right), key=lambda item: (-item.score, item.attempt, item.candidate_index))
    metadata = [
        {
            "attempt": item.attempt,
            "candidate_index": item.candidate_index,
            "headline": item.copy["headline"],
            "ad_score": item.score,
            "quality_flags": list(item.flags),
        }
        for item in selected
    ]
    for item in metadata:
        item["pair_similarity"] = round(similarity, 3)
        item["pair_diversity_adjustment"] = diversity_adjustment
    return [dict(item.copy) for item in selected], metadata


def _prompt_payload(ad_input: dict[str, Any]) -> dict[str, Any]:
    profile = analyze_copy_input(ad_input, image_provided=False)
    payload = previous._prompt_payload(profile, ad_input)
    payload["candidate_selection"] = {
        "internal_candidate_count": INTERNAL_CANDIDATE_COUNT,
        "visible_option_count": FINAL_OPTION_COUNT,
        "hard_validation": "근거·사실·CTA·길이·문법·출력 형식",
        "ranking": "요청 콘셉트, 상품 장점, 의미 있는 시각 정보, 관점 다양성",
        "headline_subcopy_repetition": (
            "같은 핵심 상품명 공유 허용. 완전히 같거나 새 정보를 더하지 않는 반복만 피함"
        ),
    }
    payload["output_limits"]["initial_option_count"] = INTERNAL_CANDIDATE_COUNT
    return payload


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "서로 다른 광고 관점의 내부 후보를 정확히 4개 작성하세요. "
        "앱이 하드 안전성 검증 후 광고성 점수로 상위 2개를 선택합니다.\n"
        + json.dumps(_prompt_payload(ad_input), ensure_ascii=False, indent=2)
    )


def build_candidate_retry_prompt(
    ad_input: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> str:
    context = {
        **_prompt_payload(ad_input),
        "retry": True,
        "previous_candidates": diagnostics,
        "retry_priority": [
            "hard_issues가 있었던 표현을 반복하지 않음",
            "안전했던 후보와도 다른 광고 관점 사용",
            "요청 콘셉트와 verified_product_facts 우선",
            "일반적인 사진 묘사보다 고객에게 의미 있는 장점 우선",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "이전 내부 후보를 검토하고 하드 안전성 문제를 모두 제거한 새로운 내부 후보 4개를 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _resolve_with_fallback(
    ad_input: dict[str, Any],
    ranked: list[RankedCandidate],
) -> tuple[list[AdCopy], tuple[int, ...], list[dict[str, Any]]]:
    fallback = previous._safe_fallback_options(ad_input)
    if not ranked:
        return fallback, (1, 2), []

    best = max(ranked, key=lambda item: (item.score, -item.attempt, -item.candidate_index))
    replacement = next(
        (
            copy
            for copy in fallback
            if _normalize_compact(copy["headline"])
            != _normalize_compact(best.copy["headline"])
        ),
        fallback[0],
    )
    resolved = [dict(best.copy), dict(replacement)]
    if collect_banner_copy_issues(resolved, ad_input):
        return fallback, (1, 2), []
    return resolved, (2,), [{
        "attempt": best.attempt,
        "candidate_index": best.candidate_index,
        "headline": best.copy["headline"],
        "ad_score": best.score,
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
    """내부 후보 4개를 안전성 검증한 뒤 광고성이 높은 2개를 선택한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider != "gemini":
        raise CopyGenerationError("배너 문구 생성은 Gemini 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    ranked_pool: list[RankedCandidate] = []
    history: list[dict[str, Any]] = []
    first_issues: tuple[str, ...] = ()
    previous_diagnostics: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        prompt = (
            build_banner_copy_prompt(ad_input)
            if attempt == 1
            else build_candidate_retry_prompt(ad_input, previous_diagnostics)
        )
        output_text = ""
        try:
            output_text = _generate_with_gemini(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                system_instructions=BANNER_COPY_SYSTEM_INSTRUCTIONS,
                response_schema=base._response_schema(profile, INTERNAL_CANDIDATE_COUNT),
            )
            returned = base.parse_banner_copy_options(
                output_text,
                ad_input,
                expected_count=INTERNAL_CANDIDATE_COUNT,
            )
            ranked, diagnostics = _rank_candidates(
                returned,
                ad_input,
                attempt=attempt,
            )
            ranked_pool.extend(ranked)
            previous_diagnostics = diagnostics
            selected, selection = select_top_two_candidates(ranked_pool, ad_input)
            hard_messages = [
                message
                for item in diagnostics
                for message in item["hard_issues"]
            ]
            history.append({
                "attempt": attempt,
                "repair": attempt > 1,
                "internal_candidate_count": len(returned),
                "hard_valid_candidate_count": len(ranked),
                "raw_image_sent": False,
                "raw_output": output_text,
                "candidates": diagnostics,
                "selected": selection,
                "issues": hard_messages,
            })
            if not first_issues and hard_messages:
                first_issues = tuple(hard_messages)
            if len(selected) == FINAL_OPTION_COUNT:
                final_issues = collect_banner_copy_issues(selected, ad_input)
                if not final_issues:
                    return BannerCopyGenerationResult(
                        copies=selected,
                        generation_mode=profile.generation_mode,
                        status="generated" if attempt == 1 else "retried",
                        attempts=attempt,
                        notice=(
                            None
                            if attempt == 1
                            else "하드 검증을 통과한 내부 후보 중 광고성 상위 2개를 선택했습니다."
                        ),
                        validation_issues=first_issues,
                        attempt_history=tuple(history),
                        prompt_version=COPY_PROMPT_VERSION,
                    )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            history.append({
                "attempt": attempt,
                "repair": attempt > 1,
                "internal_candidate_count": 0,
                "hard_valid_candidate_count": 0,
                "raw_image_sent": False,
                "raw_output": output_text,
                "candidates": [],
                "selected": [],
                "issues": [error_message],
            })
            if not first_issues:
                first_issues = (error_message,)

    resolved, fallback_indices, selection = _resolve_with_fallback(ad_input, ranked_pool)
    if history:
        history[-1]["selected"] = selection
    return BannerCopyGenerationResult(
        copies=resolved,
        generation_mode=profile.generation_mode,
        status="fallback",
        attempts=max_attempts,
        notice=(
            "하드 검증을 통과한 내부 후보가 부족해 확인된 입력 기반 fallback으로 보완했습니다. "
            f"교체 선택지: {', '.join(map(str, fallback_indices))}"
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=COPY_PROMPT_VERSION,
    )


extract_verified_visual_facts = previous.extract_verified_visual_facts
attach_visual_context = previous.attach_visual_context

