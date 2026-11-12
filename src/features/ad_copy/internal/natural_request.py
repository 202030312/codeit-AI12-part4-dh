"""프로모션을 기본적으로 배지 전용 정보로 분리하는 문구 파이프라인 v1.10.15."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, permutations
import re
from typing import Any

from src.features.ad_copy.generator import CopyGenerationError, MODEL_CONFIGS
from src.features.ad_copy.promotion import (
    BADGE_ONLY,
    HEADLINE_ALLOWED,
    copy_input_for_promotion_policy,
    headline_is_exact_promotion,
    headline_mentions_promotion,
    normalize_promotion_copy_mode,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.product_rules as previous


COPY_PROMPT_VERSION = "gemini_copy_v1.10.15-promotion-separation"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION


def _provider_copy_prompt_version(provider: str) -> str:
    return f"{provider}_copy_v1.10.15-promotion-separation"


def _append_notice(current: str | None, extra: str | None) -> str | None:
    values = [value for value in (current, extra) if value]
    return " ".join(values) or None


def _with_current_version(
    result: BannerCopyGenerationResult,
    *,
    provider: str,
    notice: str | None = None,
) -> BannerCopyGenerationResult:
    return replace(
        result,
        notice=_append_notice(result.notice, notice),
        prompt_version=_provider_copy_prompt_version(provider),
    )


def _badge_only_input(ad_input: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(ad_input)
    prepared["campaign"] = dict(ad_input.get("campaign", {}))
    prepared["campaign"]["promotion_copy_mode"] = BADGE_ONLY
    return copy_input_for_promotion_policy(prepared)


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    """일반 모드에서는 행사 원문과 행사 강조 요청을 모델 프롬프트에서 제외한다."""

    return previous.build_banner_copy_prompt(
        copy_input_for_promotion_policy(ad_input)
    ).replace(previous.COPY_PROMPT_VERSION, COPY_PROMPT_VERSION)


def collect_banner_copy_issues(
    copies: list[dict[str, Any]],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    campaign = ad_input.get("campaign", {})
    mode = normalize_promotion_copy_mode(campaign.get("promotion_copy_mode"))
    promotion = str(campaign.get("promotion") or "").strip()
    if mode == BADGE_ONLY or not any(
        headline_is_exact_promotion(str(copy.get("headline") or ""), promotion)
        for copy in copies
    ):
        return previous.collect_banner_copy_issues(
            copies,
            _badge_only_input(ad_input),
        )

    issues = previous.collect_banner_copy_issues(copies, ad_input)
    allowed_grounded_codes = {
        "UNVERIFIED_HEADLINE_EXPRESSION",
        "DISALLOWED_HEADLINE_CONTENT",
        "HEADLINE_NOT_GROUNDED",
    }
    return [
        issue
        for issue in issues
        if not (
            issue.code in allowed_grounded_codes
            and issue.option_index is not None
            and 1 <= issue.option_index <= len(copies)
            and headline_is_exact_promotion(
                str(copies[issue.option_index - 1].get("headline") or ""),
                promotion,
            )
        )
    ]


def _generate_badge_only(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None,
    max_attempts: int,
    notice: str | None = None,
) -> BannerCopyGenerationResult:
    prepared = _badge_only_input(ad_input)
    try:
        result = previous.generate_safe_banner_copy_options(
            prepared,
            model_id=model_id,
            api_key=api_key,
            client=client,
            max_attempts=max_attempts,
        )
    except CopyGenerationError:
        result = _compact_compound_fallback(prepared, max_attempts=max_attempts)
    provider = MODEL_CONFIGS[model_id].provider
    return _with_current_version(result, provider=provider, notice=notice)


def _compact_compound_fallback(
    ad_input: dict[str, Any],
    *,
    max_attempts: int,
) -> BannerCopyGenerationResult:
    """긴 복합 상품 특징에서 의미를 보존한 짧은 제목 두 안을 만든다."""

    effective, _, switch_reason, _ = previous._prepare_effective_input(ad_input)
    profile = analyze_copy_input(effective, image_provided=False)
    primary, secondary, _ = previous.assign_message_roles(effective)
    if (
        profile.generation_mode != "headline_and_subcopy"
        or primary is None
        or secondary is None
        or primary.get("source") != "verified_product_facts"
    ):
        raise CopyGenerationError("안전한 복합 상품 특징 fallback을 만들 수 없습니다.")

    raw_value = str(primary.get("value") or "")
    candidates = list(previous._message_title_candidates(primary, effective))
    compact = re.sub(
        r"\s*(?:윗면|표면|부분)(?=\s*(?:과|와)\s*)",
        "",
        raw_value,
        count=1,
    )
    variants = [
        compact,
        re.sub(r"\s*(?:과|와)\s*", ", ", compact, count=1),
    ]
    for value in variants:
        cleaned = " ".join(value.strip(" ,.! ").split())
        if (
            4 <= len(cleaned) <= profile.limits.headline
            and previous._message_is_reflected(cleaned, primary, effective)
            and not previous.is_truncated_headline(cleaned, effective)
            and cleaned not in candidates
        ):
            candidates.append(cleaned)

    bodies = previous._message_fallback_subcopies(secondary, effective)
    hashtags = previous.ad_value_base.build_deterministic_hashtags(effective)
    for first, second in combinations(candidates, 2):
        for first_body, second_body in permutations(bodies, 2):
            copies = [
                {"headline": first, "body": first_body, "hashtags": hashtags.copy()},
                {"headline": second, "body": second_body, "hashtags": hashtags.copy()},
            ]
            if not previous._subcopy_pair_is_diverse(copies[0], copies[1], effective):
                continue
            if not previous.collect_banner_copy_issues(copies, effective):
                fallback_notice = (
                    "긴 복합 상품 특징을 사실 손실 없이 압축한 fallback 두 안을 사용했습니다."
                )
                return BannerCopyGenerationResult(
                    copies=copies,
                    generation_mode=profile.generation_mode,
                    status="fallback",
                    attempts=max_attempts,
                    notice=_append_notice(switch_reason, fallback_notice),
                    prompt_version=COPY_PROMPT_VERSION,
                )
    raise CopyGenerationError("안전한 복합 상품 특징 fallback 두 개를 만들 수 없습니다.")


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    """일반 모드는 배지 전용, 프로모션 중심 모드는 제목 사용을 조건부 허용한다."""

    config = MODEL_CONFIGS.get(model_id)
    if config is None:
        raise CopyGenerationError(f"지원하지 않는 모델입니다: {model_id}")
    campaign = ad_input.get("campaign", {})
    mode = normalize_promotion_copy_mode(campaign.get("promotion_copy_mode"))
    promotion = str(campaign.get("promotion") or "").strip()
    if mode == BADGE_ONLY or not promotion:
        return _generate_badge_only(
            ad_input,
            model_id=model_id,
            api_key=api_key,
            client=client,
            max_attempts=max_attempts,
        )

    headline_limit = analyze_copy_input(ad_input, image_provided=False).limits.headline
    if len(promotion) > headline_limit:
        return _generate_badge_only(
            ad_input,
            model_id=model_id,
            api_key=api_key,
            client=client,
            max_attempts=max_attempts,
            notice=(
                "프로모션 원문이 제목 글자 수를 넘어 조건을 줄이지 않고 "
                "가로형 배지에만 표시합니다."
            ),
        )

    try:
        result = previous.generate_safe_banner_copy_options(
            ad_input,
            model_id=model_id,
            api_key=api_key,
            client=client,
            max_attempts=max_attempts,
        )
    except CopyGenerationError:
        return _generate_badge_only(
            ad_input,
            model_id=model_id,
            api_key=api_key,
            client=client,
            max_attempts=max_attempts,
            notice=(
                "프로모션 제목 후보가 최종 검증을 통과하지 못해 원문 배지 전용으로 "
                "안전하게 전환했습니다."
            ),
        )

    invalid_rewrite = any(
        headline_mentions_promotion(str(copy.get("headline") or ""))
        and not headline_is_exact_promotion(
            str(copy.get("headline") or ""),
            promotion,
        )
        for copy in result.copies
    )
    if invalid_rewrite:
        return _generate_badge_only(
            ad_input,
            model_id=model_id,
            api_key=api_key,
            client=client,
            max_attempts=max_attempts,
            notice=(
                "프로모션 조건을 바꾼 제목 후보를 사용하지 않고 원문 배지 전용으로 "
                "안전하게 전환했습니다."
            ),
        )
    return _with_current_version(result, provider=config.provider)


attach_visual_context = previous.attach_visual_context
extract_verified_visual_facts = previous.extract_verified_visual_facts
