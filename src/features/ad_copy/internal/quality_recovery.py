"""확인된 상품 특징을 설명문으로 전달하는 보조 문구 워크플로 v1.10.6."""

from __future__ import annotations

from itertools import combinations
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
from src.features.ad_copy.internal.base_rules import _normalize_compact, title_lexical_similarity
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.base_workflow as base
import src.features.ad_copy.internal.ad_value_rules as fallback_base
import src.features.ad_copy.internal.grounded_workflow as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.6-descriptive-subcopy"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = previous.INTERNAL_CANDIDATE_COUNT
FINAL_OPTION_COUNT = previous.FINAL_OPTION_COUNT
MIN_HEADLINE_AD_SCORE = previous.MIN_HEADLINE_AD_SCORE
MIN_AD_SCORE = previous.MIN_AD_SCORE
SUBCOPY_RECOMMENDED_MIN = 16
SUBCOPY_RECOMMENDED_MAX = 28
SUBCOPY_HARD_MAX = 34


BANNER_COPY_SYSTEM_INSTRUCTIONS = previous.BANNER_COPY_SYSTEM_INSTRUCTIONS + """

[v1.10.6 보조 문구 역할 분리]

headline_and_subcopy 모드에서는 다음 규칙을 모두 지키세요.

1. 제목은 고객의 관심을 끄는 요청 콘셉트·핵심 매력·의미 있는 시각 관점 중 하나를 전달합니다.
2. 보조 문구는 제목을 다시 말하는 짧은 라벨이 아니라 verified_product_facts의 특징을 소비자가 이해할 수 있게 설명하는 완결된 한 문장입니다.
3. 보조 문구에는 verified_product_facts의 원문 핵심 표현을 최소 한 개 그대로 포함하세요. 입력에 없는 맛·향·식감·강도·재료는 추가하지 마세요.
4. 보조 문구는 공백과 문장부호를 포함해 16~28자를 권장하고 34자를 절대 넘지 마세요.
5. '초콜릿 풍미', '초콜릿 풍미의 밀크셰이크'처럼 명사구로 끝내지 말고 '초콜릿 풍미를 느낄 수 있어요.'처럼 설명을 완결하세요.
6. '느낄 수 있어요', '특징이에요', '어우러져요' 같은 비명령형 설명은 허용합니다. '즐겨보세요', '만나보세요', '선택하세요' 같은 직접 명령형 CTA는 사용하지 마세요.
7. 확인 사실이 하나뿐이면 제목은 요청 콘셉트나 의미 있는 시각 관점을 사용하고, 보조 문구는 그 확인 사실을 설명하세요. 새로운 특징을 만들어 길이를 채우지 마세요.
"""


_DESCRIPTIVE_END_PATTERN = re.compile(
    r"(?:예요|이에요|해요|돼요|있어요|없어요|느껴져요|보여요|돋보여요|"
    r"어우러져요|입니다|합니다|됩니다)\.?$"
)


def is_descriptive_subcopy(text: str) -> bool:
    """짧은 라벨이 아니라 자연스럽게 끝나는 설명문인지 확인한다."""

    return bool(text.strip() and _DESCRIPTIVE_END_PATTERN.search(text.strip()))


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
    """기존 사실 검증에 설명형 보조 문구 형식 검증을 더한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    issues = list(previous.collect_banner_copy_issues(copies, ad_input))
    if profile.generation_mode != "headline_and_subcopy":
        return _deduplicate_issues(issues)

    for index, copy in enumerate(copies, start=1):
        body = copy["body"].strip()
        if not body:
            continue
        if len(body) > min(profile.limits.subcopy, SUBCOPY_HARD_MAX):
            issues.append(CopyValidationIssue(
                "SUBCOPY_TOO_LONG_V1106",
                f"선택지 {index} 보조 문구가 v1.10.6 최대 {SUBCOPY_HARD_MAX}자를 넘습니다.",
                index,
                "subcopy",
            ))
        if not is_descriptive_subcopy(body):
            issues.append(CopyValidationIssue(
                "SUBCOPY_NOT_DESCRIPTIVE",
                f"선택지 {index} 보조 문구가 상품 특징을 설명하는 완결된 문장이 아닙니다.",
                index,
                "subcopy",
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
    """v1.10.5 점수에 설명문 완결성과 권장 길이를 보조 문구 점수로 더한다."""

    _, headline_score, subcopy_score, flags, base_breakdown = (
        previous.score_ad_candidate(copy, ad_input)
    )
    breakdown = dict(base_breakdown)
    profile = analyze_copy_input(ad_input, image_provided=False)
    body = copy["body"].strip()
    if profile.generation_mode == "headline_and_subcopy" and body:
        if is_descriptive_subcopy(body):
            breakdown["subcopy_descriptive_sentence"] = 12
            subcopy_score += 12
        if SUBCOPY_RECOMMENDED_MIN <= len(body) <= SUBCOPY_RECOMMENDED_MAX:
            breakdown["subcopy_recommended_length"] = 8
            subcopy_score += 8
        elif len(body) <= SUBCOPY_HARD_MAX:
            breakdown["subcopy_acceptable_length"] = 3
            subcopy_score += 3
    return sum(breakdown.values()), headline_score, subcopy_score, flags, breakdown


def _rank_candidates(
    candidates: list[AdCopy],
    ad_input: dict[str, Any],
    *,
    attempt: int,
) -> tuple[list[previous.RankedCandidate], list[dict[str, Any]]]:
    valid: list[previous.RankedCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for index, copy in enumerate(candidates, start=1):
        hard_issues = _candidate_hard_issues(
            copy,
            ad_input,
            candidate_index=index,
        )
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
            "hard_issues": [issue.message for issue in hard_issues],
            "ranking_issues": ranking_issues,
            "ad_score": score,
            "headline_ad_score": headline_score,
            "subcopy_score": subcopy_score,
            "quality_flags": list(flags),
            "score_breakdown": breakdown,
        })
        if not hard_issues:
            valid.append(previous.RankedCandidate(
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


def _prompt_payload(ad_input: dict[str, Any]) -> dict[str, Any]:
    payload = previous._prompt_payload(ad_input)
    profile = analyze_copy_input(ad_input, image_provided=False)
    payload["subcopy_policy"] = {
        "role": "제목에 없는 확인된 상품 특징을 설명하는 완결된 한 문장",
        "verified_fact_required": True,
        "recommended_length": f"{SUBCOPY_RECOMMENDED_MIN}~{SUBCOPY_RECOMMENDED_MAX}자",
        "hard_max_length": min(profile.limits.subcopy, SUBCOPY_HARD_MAX),
        "bare_noun_phrase": "금지",
        "non_imperative_experience": "느낄 수 있어요·특징이에요 같은 설명 허용",
        "imperative_cta": "즐겨보세요·만나보세요·선택하세요 금지",
        "invented_detail": "길이를 채우기 위한 새 맛·향·식감·강도·재료 추가 금지",
    }
    if profile.generation_mode == "headline_and_subcopy":
        payload["output_limits"]["subcopy_max_length"] = min(
            profile.limits.subcopy,
            SUBCOPY_HARD_MAX,
        )
        payload["output_limits"]["subcopy_recommended_length"] = (
            f"{SUBCOPY_RECOMMENDED_MIN}~{SUBCOPY_RECOMMENDED_MAX}"
        )
    return payload


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "내부 후보를 정확히 4개 작성하세요. 제목은 광고 메시지, 보조 문구는 확인된 상품 특징을 설명하는 완결된 문장으로 역할을 분리하세요. "
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
            "hard_issues가 있었던 표현을 반복하지 않음",
            "보조 문구를 명사구가 아닌 완결된 설명문으로 작성",
            "보조 문구에 verified_product_facts 핵심 표현을 그대로 포함",
            f"보조 문구 {SUBCOPY_RECOMMENDED_MIN}~{SUBCOPY_RECOMMENDED_MAX}자 권장·{SUBCOPY_HARD_MAX}자 이하",
            "제목과 보조 문구에서 새 정보 없는 의미 반복 제거",
            f"headline_ad_score가 {MIN_HEADLINE_AD_SCORE}점 미만인 제목을 다른 상품 장점·요청 콘셉트로 교체",
        ],
    }
    return (
        f"문구 프롬프트 버전: {COPY_PROMPT_VERSION}\n"
        "이전 후보의 문제를 보완한 새로운 내부 후보 4개를 반환하세요.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _descriptive_fallback_subcopy(ad_input: dict[str, Any]) -> str:
    profile = analyze_copy_input(ad_input, image_provided=False)
    fact = profile.verified_facts[0].strip()
    if re.search(r"(?:조합|구성|재료)$", fact):
        value = f"{fact}{fallback_base._subject_particle(fact)} 특징이에요."
    elif re.search(r"(?:맛|향|풍미|식감|질감)$", fact):
        value = f"{fact}{fallback_base._object_particle(fact)} 느낄 수 있어요."
    else:
        value = f"{fact}{fallback_base._subject_particle(fact)} 특징이에요."
    limit = min(profile.limits.subcopy, SUBCOPY_HARD_MAX)
    if len(value) <= limit:
        return value
    shorter = f"{fact}{fallback_base._subject_particle(fact)} 특징이에요."
    return shorter if len(shorter) <= limit else fact[:limit].rstrip()


def _safe_fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    profile = analyze_copy_input(ad_input, image_provided=False)
    fallback = fallback_base._safe_fallback_options(ad_input)
    if profile.generation_mode != "headline_and_subcopy":
        return fallback
    body = _descriptive_fallback_subcopy(ad_input)
    copies = [{**copy, "body": body} for copy in fallback]
    if not collect_banner_copy_issues(copies, ad_input):
        return copies
    return fallback


def _resolve_with_fallback(
    ad_input: dict[str, Any],
    ranked: list[previous.RankedCandidate],
) -> tuple[list[AdCopy], tuple[int, ...], list[dict[str, Any]]]:
    fallback = _safe_fallback_options(ad_input)
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
    if collect_banner_copy_issues(resolved, ad_input):
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


def _response_schema(profile: Any) -> dict[str, Any]:
    schema = base._response_schema(profile, INTERNAL_CANDIDATE_COUNT)
    if profile.generation_mode == "headline_and_subcopy":
        subcopy = schema["properties"]["options"]["items"]["properties"]["subcopy"]
        subcopy["maxLength"] = min(profile.limits.subcopy, SUBCOPY_HARD_MAX)
    return schema


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    """공통 v1.10.6 기준으로 내부 후보 4개 중 최종 2개를 선택한다."""

    profile = analyze_copy_input(ad_input, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider not in {"gemini", "openai"}:
        raise CopyGenerationError("배너 문구 생성은 Gemini 또는 OpenAI 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    ranked_pool: list[previous.RankedCandidate] = []
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
            response_schema = _response_schema(profile)
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
            selected, selection = previous.select_top_two_candidates(ranked_pool, ad_input)
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
                final_issues = collect_banner_copy_issues(selected, ad_input)
                if not final_issues:
                    return BannerCopyGenerationResult(
                        copies=selected,
                        generation_mode=profile.generation_mode,
                        status="generated" if attempt == 1 else "retried",
                        attempts=attempt,
                        notice=(
                            None if attempt == 1 else
                            "설명형 보조 문구와 제목 최소 점수를 통과한 후보 중 상위 2개를 선택했습니다."
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
            "설명형 보조 문구와 제목 최소 점수를 통과한 내부 후보가 부족해 확인된 입력 기반 fallback으로 보완했습니다. "
            f"교체 선택지: {', '.join(map(str, fallback_indices))}"
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=COPY_PROMPT_VERSION,
    )


attach_visual_context = previous.attach_visual_context


def extract_verified_visual_facts(**kwargs: Any) -> Any:
    fallback_base._generate_with_gemini = _generate_with_gemini
    return fallback_base.extract_verified_visual_facts(**kwargs)
