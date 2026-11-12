"""프로모션 문구의 생성·표시 책임을 분리하는 공통 정책."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Literal


PromotionCopyMode = Literal["badge_only", "headline_allowed"]

BADGE_ONLY: PromotionCopyMode = "badge_only"
HEADLINE_ALLOWED: PromotionCopyMode = "headline_allowed"
VALID_PROMOTION_COPY_MODES = {BADGE_ONLY, HEADLINE_ALLOWED}

_PROMOTION_TERM_PATTERN = re.compile(
    r"(?:프로모션|행사|할인|증정|무료|혜택|쿠폰|"
    r"\d[\d,]*\s*(?:%|원)\s*(?:할인|혜택)?|\d+\s*\+\s*\d+)",
    re.IGNORECASE,
)


def normalize_promotion_copy_mode(value: Any) -> PromotionCopyMode:
    """누락된 값은 안전한 배지 전용 모드로 처리한다."""

    normalized = str(value or BADGE_ONLY).strip().casefold()
    if normalized not in VALID_PROMOTION_COPY_MODES:
        return BADGE_ONLY
    return normalized  # type: ignore[return-value]


def _compact(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _without_promotion_request(value: str | None, promotion: str | None) -> str | None:
    """일반 모드의 자유 요청에서 행사 강조 절만 제외한다."""

    if not value:
        return None
    cleaned = value
    if promotion:
        cleaned = cleaned.replace(promotion, " ")
    clauses = re.split(r"(?<=[.!?。])\s+|\s*[,;]\s*", cleaned)
    kept = [clause.strip() for clause in clauses if clause.strip() and not _PROMOTION_TERM_PATTERN.search(clause)]
    normalized = " ".join(kept).strip()
    return normalized or None


def copy_input_for_promotion_policy(ad_input: dict[str, Any]) -> dict[str, Any]:
    """문구 생성기가 배지 전용 프로모션을 제목 근거로 사용하지 않게 한다."""

    prepared = deepcopy(ad_input)
    campaign = prepared.setdefault("campaign", {})
    mode = normalize_promotion_copy_mode(campaign.get("promotion_copy_mode"))
    campaign["promotion_copy_mode"] = mode
    if mode == BADGE_ONLY:
        promotion = str(campaign.get("promotion") or "").strip() or None
        campaign["promotion"] = None
        campaign["copy_request"] = _without_promotion_request(
            campaign.get("copy_request"),
            promotion,
        )
    return prepared


def headline_is_exact_promotion(headline: str, promotion: str | None) -> bool:
    """프로모션 조건을 바꾸지 않고 제목 전체로 사용했는지 확인한다."""

    if not promotion:
        return False
    return bool(_compact(headline) and _compact(headline) == _compact(promotion))


def headline_mentions_promotion(headline: str) -> bool:
    return bool(_PROMOTION_TERM_PATTERN.search(headline))


def promotion_badge_text(
    promotion: str | None,
    *,
    headline: str,
    promotion_copy_mode: Any = BADGE_ONLY,
) -> str | None:
    """제목에 원문 프로모션을 쓴 경우에만 중복 배지를 숨긴다."""

    normalized = str(promotion or "").strip()
    if not normalized:
        return None
    mode = normalize_promotion_copy_mode(promotion_copy_mode)
    if mode == HEADLINE_ALLOWED and headline_is_exact_promotion(headline, normalized):
        return None
    return normalized
