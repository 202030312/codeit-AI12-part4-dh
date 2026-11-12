from __future__ import annotations

from PIL import Image

from src.shared.combined_banner_renderer import render_combined_banner


def test_vertical_layout_preserves_exact_promotion_badge() -> None:
    result = render_combined_banner(
        Image.new("RGB", (1080, 1080), (245, 245, 245)),
        template_id="promo",
        text_safe_area=(50, 80, 450, 1000),
        copy={"headline": "깔끔한 커피 풍미", "body": "", "hashtags": []},
        store_name="오늘의 작은 카페",
        product_name="아이스 아메리카노",
        base_price="4,500원",
        sale_price=None,
        promotion="평일 오전 테이크아웃 고객 500원 할인",
        promotion_copy_mode="badge_only",
    )

    assert result.image.size == (1080, 1080)
    assert result.layout == "vertical"
    assert result.promotion_badge_text == "평일 오전 테이크아웃 고객 500원 할인"
    assert all(check["passed"] for check in result.contrast_checks)
    roles = [check["role"] for check in result.contrast_checks]
    assert roles.index("store_name") < roles.index("headline") < roles.index("product_name")
    boxes_by_role = {
        check["role"]: check["box"] for check in result.contrast_checks
    }
    headline_box = boxes_by_role["headline"]
    badge_box = boxes_by_role["promotion_badge"]
    assert headline_box[3] - headline_box[1] < 60
    assert badge_box[3] - badge_box[1] >= 85
    assert all(
        50 <= box[0] and 80 <= box[1] and box[2] <= 450 and box[3] <= 1000
        for box in result.text_boxes
    )


def test_horizontal_layout_keeps_base_and_sale_price_in_safe_area() -> None:
    result = render_combined_banner(
        Image.new("RGB", (1080, 1080), (245, 235, 207)),
        template_id="natural",
        text_safe_area=(70, 750, 1010, 1030),
        copy={
            "headline": "카다멈 향이 전하는 매력",
            "body": "자연스러운 분위기의 번",
            "hashtags": [],
        },
        store_name="매일 굽는 작은 베이커리",
        product_name="카다멈 번",
        base_price="5,000원",
        sale_price="4,500원",
        promotion=None,
    )

    assert result.layout == "horizontal"
    assert all(
        70 <= box[0] and 750 <= box[1] and box[2] <= 1010 and box[3] <= 1030
        for box in result.text_boxes
    )
    assert all(check["passed"] for check in result.contrast_checks)
    roles = [check["role"] for check in result.contrast_checks]
    assert (
        roles.index("store_name")
        < roles.index("headline")
        < roles.index("product_name")
        < roles.index("body")
    )
    boxes_by_role = {
        check["role"]: check["box"] for check in result.contrast_checks
    }
    headline_box = boxes_by_role["headline"]
    assert headline_box[3] - headline_box[1] < 55
