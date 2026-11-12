from __future__ import annotations

from src.features.ad_copy.promotion import (
    BADGE_ONLY,
    HEADLINE_ALLOWED,
    copy_input_for_promotion_policy,
    headline_is_exact_promotion,
    promotion_badge_text,
)


def _input(mode: str) -> dict:
    return {
        "product": {"name": "아이스 아메리카노", "features": "깔끔한 커피 풍미"},
        "campaign": {
            "promotion": "평일 오전 테이크아웃 고객 500원 할인",
            "promotion_copy_mode": mode,
            "copy_request": "짧고 친근하게, 평일 오전 테이크아웃 고객 500원 할인 강조",
        },
    }


def test_badge_only_removes_promotion_from_copy_input_without_changing_original() -> None:
    ad_input = _input(BADGE_ONLY)

    prepared = copy_input_for_promotion_policy(ad_input)

    assert prepared["campaign"]["promotion"] is None
    assert "할인" not in str(prepared["campaign"]["copy_request"])
    assert ad_input["campaign"]["promotion"] == "평일 오전 테이크아웃 고객 500원 할인"


def test_headline_allowed_keeps_exact_promotion_available() -> None:
    ad_input = _input(HEADLINE_ALLOWED)

    prepared = copy_input_for_promotion_policy(ad_input)

    assert prepared["campaign"]["promotion"] == ad_input["campaign"]["promotion"]


def test_badge_is_hidden_only_for_an_exact_promotion_headline() -> None:
    promotion = "오후 5시 이후 10% 할인"

    assert headline_is_exact_promotion("오후 5시 이후 10% 할인", promotion)
    assert promotion_badge_text(
        promotion,
        headline=promotion,
        promotion_copy_mode=HEADLINE_ALLOWED,
    ) is None
    assert promotion_badge_text(
        promotion,
        headline="오후 5시 10% 할인",
        promotion_copy_mode=HEADLINE_ALLOWED,
    ) == promotion
    assert promotion_badge_text(
        promotion,
        headline=promotion,
        promotion_copy_mode=BADGE_ONLY,
    ) == promotion
