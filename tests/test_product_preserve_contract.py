import json

import pytest

from src.features.product_preserve.contract import (
    FEATURE3_OUTPUT_SIZE,
    FEATURE3_VERSION,
    build_image_request,
    build_feature3_request,
    recommended_template_ids,
)


def valid_image_request(**overrides: object) -> dict:
    values = {
        "background_mood": "warm_soft",
        "color_tone": "white_cream",
        "product_position": "center",
        "product_size": "default",
        "margin": "default",
        "include_shadow": True,
    }
    values.update(overrides)
    return build_image_request(**values)  # type: ignore[arg-type]


def test_feature3_scope_is_fixed_to_square_textless_output() -> None:
    request = build_feature3_request(
        image_filename=" strawberry_latte.png ",
        image_content_type=" image/png ",
        menu_category="iced_drink",
        image_purpose="image_only",
        text_space="none",
        image_request=valid_image_request(),
    )

    assert FEATURE3_VERSION == "image_ad_v0.4.1"
    assert FEATURE3_OUTPUT_SIZE == (1080, 1080)
    assert request == {
        "schema_version": "1.0",
        "feature_version": "image_ad_v0.4.1",
        "request_source": "direct_request",
        "menu_category": "iced_drink",
        "image_purpose": "image_only",
        "text_space": "none",
        "image": {
            "file_name": "strawberry_latte.png",
            "content_type": "image/png",
        },
        "image_request": {
            "background_mood": "warm_soft",
            "color_tone": "white_cream",
            "product_position": "center",
            "product_size": "default",
            "margin": "default",
            "include_shadow": True,
        },
        "style_template_id": None,
        "output": {
            "width": 1080,
            "height": 1080,
            "format": "PNG",
            "include_text": False,
        },
    }
    assert json.loads(json.dumps(request, ensure_ascii=False)) == request


@pytest.mark.parametrize(
    ("image_request", "style_template_id"),
    [
        (valid_image_request(), None),
        (None, "minimal_cream"),
        (valid_image_request(product_position="right"), "minimal_cream"),
    ],
)
def test_accepts_request_template_or_both(
    image_request: dict | None,
    style_template_id: str | None,
) -> None:
    result = build_feature3_request(
        image_filename="product.jpg",
        image_content_type="image/jpeg",
        menu_category="dessert",
        image_purpose="copy_later",
        text_space="auto",
        image_request=image_request,
        style_template_id=style_template_id,
    )

    assert result["image_request"] == image_request
    assert result["style_template_id"] == style_template_id
    expected_source = (
        "direct_request_and_style_template"
        if image_request is not None and style_template_id is not None
        else "direct_request"
        if image_request is not None
        else "style_template"
    )
    assert result["request_source"] == expected_source


def test_rejects_request_without_product_image() -> None:
    with pytest.raises(ValueError, match="상품 사진"):
        build_feature3_request(
            image_filename=" ",
            image_content_type=None,
            menu_category="dessert",
            image_purpose="image_only",
            text_space="none",
            image_request=valid_image_request(),
        )


def test_rejects_request_without_instructions_or_template() -> None:
    with pytest.raises(ValueError, match="요청 사항 또는 스타일 템플릿"):
        build_feature3_request(
            image_filename="product.png",
            image_content_type="image/png",
            menu_category="dessert",
            image_purpose="image_only",
            text_space="none",
            image_request=None,
            style_template_id=None,
        )


def test_contract_does_not_accept_copy_or_product_description_fields() -> None:
    with pytest.raises(TypeError):
        build_feature3_request(
            image_filename="product.png",
            image_content_type="image/png",
            menu_category="dessert",
            image_purpose="image_only",
            text_space="none",
            image_request=valid_image_request(),
            product_features="진한 원두 향",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("background_mood", "retro"),
        ("color_tone", "rainbow"),
        ("product_position", "top"),
        ("product_size", "extra_large"),
        ("margin", "none"),
        ("include_shadow", "yes"),
    ],
)
def test_rejects_unsupported_request_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError):
        valid_image_request(**{field_name: invalid_value})


def test_rejects_request_fields_outside_the_fixed_scope() -> None:
    request = valid_image_request()
    request["headline"] = "오늘의 추천"

    with pytest.raises(ValueError, match="요청 사항에는"):
        build_feature3_request(
            image_filename="product.png",
            image_content_type="image/png",
            menu_category="dessert",
            image_purpose="image_only",
            text_space="none",
            image_request=request,
        )


def test_recommends_three_templates_for_each_menu_category() -> None:
    assert recommended_template_ids("bread") == ("natural", "cream", "promo")
    assert recommended_template_ids("hot_drink") == ("cream", "night", "natural")
    assert recommended_template_ids("iced_drink") == ("clean", "berry", "promo")
    assert recommended_template_ids("dessert") == ("berry", "night", "cream")
    assert recommended_template_ids("set_gift") == ("night", "cream", "promo")


def test_rejects_copy_later_without_text_space() -> None:
    with pytest.raises(ValueError, match="문구 여백"):
        build_feature3_request(
            image_filename="product.png",
            image_content_type="image/png",
            menu_category="dessert",
            image_purpose="copy_later",
            text_space="none",
            style_template_id="berry",
        )
