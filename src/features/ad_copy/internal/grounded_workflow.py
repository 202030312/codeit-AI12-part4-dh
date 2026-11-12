"""제목 최소 광고성 점수와 의미 반복 감점을 적용한 문구 생성 v1.10.5."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
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
    _terms,
    title_lexical_similarity,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    analyze_copy_input,
)
import src.features.ad_copy.internal.base_workflow as base
import src.features.ad_copy.internal.structured_workflow as structured
import src.features.ad_copy.internal.ad_value_rules as ad_value_base
import src.features.ad_copy.internal.candidate_recovery as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.5-title-score-floor-semantic-repeat"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = previous.INTERNAL_CANDIDATE_COUNT
FINAL_OPTION_COUNT = previous.FINAL_OPTION_COUNT
MIN_HEADLINE_AD_SCORE = 18
MIN_AD_SCORE = 20


BANNER_COPY_SYSTEM_INSTRUCTIONS = previous.BANNER_COPY_SYSTEM_INSTRUCTIONS + """

[v1.10.5 선택 기준 대응]

1. 내부 후보 4개 중 최소 2개는 제목 자체만으로 요청 콘셉트 또는 verified_product_facts의 상품 장점이 드러나야 합니다.
2. 보조 문구가 상품 특징을 잘 설명하더라도 제목이 컵·접시·테이블·색상·단순 형태 설명에 머물면 선택되지 않습니다.
3. 제목과 보조 문구는 같은 상품명을 공유할 수 있지만, '조화↔조합', '풍미↔풍미가 돋보여요'처럼 새 정보를 더하지 않는 의미 반복은 피하세요.
4. 시각 사실만 사용하는 후보는 최대 1개로 제한하고 수량·제품 구성·구별되는 토핑처럼 고객에게 의미 있는 경우에만 작성하세요.
"""


_GENERIC_ADDITIVE_TERM_ROOTS = (
    "돋보", "느껴", "보여", "담", "중심", "그대로", "특징", "매력",
    "표현", "강조", "메뉴", "상품", "음료", "디저트", "베이커리",
)


@dataclass(frozen=True)
class RankedCandidate:
    copy: AdCopy
    candidate_index: int
    attempt: int
    score: int
    headline_score: int
    subcopy_score: int
    flags: tuple[str, ...]
    score_breakdown: dict[str, int]

    @property
    def qualifies(self) -> bool:
        return (
            self.headline_score >= MIN_HEADLINE_AD_SCORE
            and self.score >= MIN_AD_SCORE
        )


def _is_generic_additive_term(term: str) -> bool:
    normalized = _normalize_compact(term)
    return any(root in normalized for root in _GENERIC_ADDITIVE_TERM_ROOTS)


def has_semantic_headline_subcopy_repetition(copy: AdCopy) -> bool:
    """표현만 바뀌고 상품 정보가 늘지 않은 제목·보조 문구를 찾는다."""

    headline = copy["headline"].strip()
    body = copy["body"].strip()
    if not headline or not body:
        return False
    if _normalize_compact(headline) == _normalize_compact(body):
        return True

    headline_terms = set(_terms(headline))
    body_terms = set(_terms(body))
    if not headline_terms or not body_terms:
        return False
    if headline_terms == body_terms:
        return True
    if body_terms <= headline_terms:
        return True

    shared = headline_terms & body_terms
    added = body_terms - headline_terms
    coverage = len(shared) / len(headline_terms)
    return bool(
        coverage >= 0.75
        and added
        and all(_is_generic_additive_term(term) for term in added)
    )


def _quality_flags(copy: AdCopy, ad_input: dict[str, Any]) -> tuple[str, ...]:
    flags = list(previous._candidate_quality_flags(copy, ad_input))
    if has_semantic_headline_subcopy_repetition(copy):
        flags = [
            flag
            for flag in flags
            if flag != "excessive_headline_subcopy_repetition"
        ]
        flags.append("semantic_headline_subcopy_repetition")
    return tuple(dict.fromkeys(flags))


def score_ad_candidate(
    copy: AdCopy,
    ad_input: dict[str, Any],
) -> tuple[int, int, int, tuple[str, ...], dict[str, int]]:
    """제목 광고성과 보조 문구 충실도를 분리해 계산한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    headline = copy["headline"]
    body = copy["body"]
    headline_compact = _normalize_compact(headline)
    breakdown: dict[str, int] = {}

    campaign_matches = sum(
        _normalize_compact(phrase) in headline_compact
        for phrase in ad_value_base._campaign_phrases(profile.copy_request)
    )
    if campaign_matches:
        breakdown["headline_requested_concept"] = min(24, campaign_matches * 12)
    if ad_value_base._strong_product_or_campaign_grounding(headline, ad_input):
        breakdown["headline_grounding"] = 24

    meaningful_visual = ad_value_base._meaningful_visual_score(headline, ad_input)
    if meaningful_visual:
        breakdown["headline_meaningful_visual"] = min(12, meaningful_visual * 6)
    if 8 <= len(headline) <= 15:
        breakdown["headline_length"] = 8
    elif 4 <= len(headline) <= profile.limits.headline:
        breakdown["headline_length"] = 4

    flags = _quality_flags(copy, ad_input)
    headline_penalties = {
        "low_ad_value_visual_title": -36,
        "repeated_word": -8,
        "repeated_color": -8,
    }
    for flag, penalty in headline_penalties.items():
        if flag in flags:
            breakdown[flag] = penalty

    headline_score = sum(
        value
        for key, value in breakdown.items()
        if key.startswith("headline_") or key in headline_penalties
    )

    subcopy_score = 0
    if body and ad_value_base._strong_product_or_campaign_grounding(body, ad_input):
        subcopy_score = 10
        breakdown["subcopy_grounding"] = subcopy_score

    if "semantic_headline_subcopy_repetition" in flags:
        breakdown["semantic_headline_subcopy_repetition"] = -32
    elif "excessive_headline_subcopy_repetition" in flags:
        breakdown["excessive_headline_subcopy_repetition"] = -18

    score = sum(breakdown.values())
    return score, headline_score, subcopy_score, flags, breakdown


def _rank_candidates(
    candidates: list[AdCopy],
    ad_input: dict[str, Any],
    *,
    attempt: int,
) -> tuple[list[RankedCandidate], list[dict[str, Any]]]:
    valid: list[RankedCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for index, copy in enumerate(candidates, start=1):
        hard_issues = previous._candidate_hard_issues(
            copy,
            ad_input,
            candidate_index=index,
        )
        score, headline_score, subcopy_score, flags, breakdown = score_ad_candidate(
            copy,
            ad_input,
        )
        ranking_issues = []
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
            "hard_issues": [issue.message for issue in hard_issues],
            "ranking_issues": ranking_issues,
            "ad_score": score,
            "headline_ad_score": headline_score,
            "subcopy_score": subcopy_score,
            "quality_flags": list(flags),
            "score_breakdown": breakdown,
        })
        if not hard_issues:
            valid.append(RankedCandidate(
                copy=dict(copy),
                candidate_index=index,
                attempt=attempt,
                score=score,
                headline_score=headline_score,
                subcopy_score=subcopy_score,
                flags=flags,
                score_breakdown=breakdown,
            ))
    return valid, diagnostics


def select_top_two_candidates(
    candidates: list[RankedCandidate],
    ad_input: dict[str, Any],
) -> tuple[list[AdCopy], list[dict[str, Any]]]:
    qualified = [candidate for candidate in candidates if candidate.qualifies]
    best: tuple[int, RankedCandidate, RankedCandidate, float, int] | None = None
    for left, right in combinations(qualified, 2):
        if _normalize_compact(left.copy["headline"]) == _normalize_compact(right.copy["headline"]):
            continue
        similarity = title_lexical_similarity(
            left.copy["headline"], right.copy["headline"]
        )
        diversity_adjustment = 8
        if similarity >= 0.85:
            diversity_adjustment = -24
        elif similarity >= 0.5:
            diversity_adjustment = -10
        elif ad_value_base._has_visual_angle(
            left.copy["headline"], ad_input
        ) != ad_value_base._has_visual_angle(right.copy["headline"], ad_input):
            diversity_adjustment = 12
        pair_score = left.score + right.score + diversity_adjustment
        candidate = (pair_score, left, right, similarity, diversity_adjustment)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return [], []

    _, left, right, similarity, diversity_adjustment = best
    selected = sorted(
        (left, right),
        key=lambda item: (-item.score, -item.headline_score, item.attempt, item.candidate_index),
    )
    metadata = [{
        "attempt": item.attempt,
        "candidate_index": item.candidate_index,
        "headline": item.copy["headline"],
        "ad_score": item.score,
        "headline_ad_score": item.headline_score,
        "subcopy_score": item.subcopy_score,
        "quality_flags": list(item.flags),
        "pair_similarity": round(similarity, 3),
        "pair_diversity_adjustment": diversity_adjustment,
    } for item in selected]
    return [dict(item.copy) for item in selected], metadata


def _prompt_payload(ad_input: dict[str, Any]) -> dict[str, Any]:
    payload = previous._prompt_payload(ad_input)
    payload["candidate_selection"].update({
        "minimum_headline_ad_score": MIN_HEADLINE_AD_SCORE,
        "minimum_total_ad_score": MIN_AD_SCORE,
        "minimum_grounded_or_meaningful_candidates": 2,
        "headline_scored_independently": True,
        "subcopy_cannot_rescue_weak_headline": True,
        "semantic_repetition_penalty": (
            "조화↔조합, 풍미↔풍미가 돋보여요처럼 새 사실 없는 반복 감점"
        ),
    })
    return payload


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "내부 후보를 정확히 4개 작성하되, 그중 최소 2개는 제목 자체로 요청 콘셉트나 상품 장점이 명확해야 합니다. "
        "보조 문구로 약한 제목을 보완하려 하지 마세요.\n"
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
            f"headline_ad_score가 {MIN_HEADLINE_AD_SCORE}점 미만이었던 제목을 완전히 다른 상품 장점·요청 콘셉트로 교체",
            "hard_issues가 있었던 표현을 반복하지 않음",
            "제목 자체에 요청 콘셉트 또는 verified_product_facts를 구체적으로 반영",
            "제목과 보조 문구에서 새 정보 없는 의미 반복 제거",
            "순수 시각 묘사는 최대 한 개만 작성",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "이전 후보의 하드 문제와 제목 점수 미달을 보완한 새로운 내부 후보 4개를 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _resolve_with_fallback(
    ad_input: dict[str, Any],
    ranked: list[RankedCandidate],
) -> tuple[list[AdCopy], tuple[int, ...], list[dict[str, Any]]]:
    fallback = ad_value_base._safe_fallback_options(ad_input)
    qualified = [candidate for candidate in ranked if candidate.qualifies]
    if not qualified:
        return fallback, (1, 2), []

    best = max(
        qualified,
        key=lambda item: (item.score, item.headline_score, -item.attempt, -item.candidate_index),
    )
    replacement = next(
        (
            copy for copy in fallback
            if _normalize_compact(copy["headline"])
            != _normalize_compact(best.copy["headline"])
        ),
        fallback[0],
    )
    resolved = [dict(best.copy), dict(replacement)]
    if previous.collect_banner_copy_issues(resolved, ad_input):
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
    """제목 점수 기준을 넘는 안전 후보 두 개만 선택한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider not in {"gemini", "openai"}:
        raise CopyGenerationError("배너 문구 생성은 Gemini 또는 OpenAI 모델만 지원합니다.")
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
        usage: dict[str, int] = {}
        try:
            response_schema = base._response_schema(
                profile,
                INTERNAL_CANDIDATE_COUNT,
            )
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
                ad_input,
                expected_count=INTERNAL_CANDIDATE_COUNT,
            )
            ranked, diagnostics = _rank_candidates(returned, ad_input, attempt=attempt)
            ranked_pool.extend(ranked)
            previous_diagnostics = diagnostics
            selected, selection = select_top_two_candidates(ranked_pool, ad_input)
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
            })
            if not first_issues and hard_messages:
                first_issues = tuple(hard_messages)
            if len(selected) == FINAL_OPTION_COUNT:
                final_issues = previous.collect_banner_copy_issues(selected, ad_input)
                if not final_issues:
                    return BannerCopyGenerationResult(
                        copies=selected,
                        generation_mode=profile.generation_mode,
                        status="generated" if attempt == 1 else "retried",
                        attempts=attempt,
                        notice=(
                            None if attempt == 1 else
                            "제목 최소 점수를 통과한 내부 후보 중 광고성 상위 2개를 선택했습니다."
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
                "qualified_candidate_count": 0,
                "raw_image_sent": False,
                "raw_output": output_text,
                "candidates": [],
                "selected": [],
                "issues": [error_message],
                "usage": usage,
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
            "제목 최소 점수를 통과한 내부 후보가 부족해 확인된 입력 기반 fallback으로 보완했습니다. "
            f"교체 선택지: {', '.join(map(str, fallback_indices))}"
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=COPY_PROMPT_VERSION,
    )


collect_banner_copy_issues = previous.collect_banner_copy_issues
attach_visual_context = previous.attach_visual_context


def extract_verified_visual_facts(**kwargs: Any) -> Any:
    """v1.10.3 한국어 재추출 규칙을 timeout 가능한 일반 API로 실행한다."""

    ad_value_base._generate_with_gemini = _generate_with_gemini
    return ad_value_base.extract_verified_visual_facts(**kwargs)

