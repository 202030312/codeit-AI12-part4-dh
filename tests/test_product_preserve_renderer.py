from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from src.features.product_preserve.contract import build_image_request
from src.features.product_preserve.renderer import (
    FEATURE3_TEMPLATES,
    composite_feature3_background,
    feature3_image_to_png_bytes,
    render_feature3_image,
)


def product_image() -> Image.Image:
    image = Image.new("RGBA", (500, 700), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((100, 80, 400, 650), radius=80, fill=(220, 35, 70, 255))
    return image


def image_request(**overrides: object) -> dict:
    values = {
        "background_mood": "bright_clean",
        "color_tone": "white_cream",
        "product_position": "center",
        "product_size": "default",
        "margin": "default",
        "include_shadow": False,
    }
    values.update(overrides)
    return build_image_request(**values)  # type: ignore[arg-type]


def red_product_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    red_mask = Image.new("1", image.size)
    source = image.convert("RGB")
    red_mask.putdata(
        [
            red > 190 and green < 80 and blue < 100
            for red, green, blue in source.get_flattened_data()
        ]
    )
    bounds = red_mask.getbbox()
    assert bounds is not None
    return bounds


def test_defines_six_unique_textless_templates() -> None:
    assert len(FEATURE3_TEMPLATES) == 6
    assert len({item.template_id for item in FEATURE3_TEMPLATES}) == 6


@pytest.mark.parametrize("template", FEATURE3_TEMPLATES)
def test_renders_every_template_at_fixed_output_size(template) -> None:
    result = render_feature3_image(product_image(), template_id=template.template_id)
    png_bytes = feature3_image_to_png_bytes(result)

    assert result.size == (1080, 1080)
    assert result.mode == "RGB"
    with Image.open(BytesIO(png_bytes)) as decoded:
        assert decoded.size == (1080, 1080)
        assert decoded.format == "PNG"


def test_image_only_is_larger_and_copy_later_reserves_requested_side() -> None:
    image_only = render_feature3_image(
        product_image(),
        template_id="clean",
        image_purpose="image_only",
        text_space="none",
    )
    copy_later = render_feature3_image(
        product_image(),
        template_id="clean",
        image_purpose="copy_later",
        text_space="left",
    )
    image_only_bounds = red_product_bounds(image_only)
    copy_later_bounds = red_product_bounds(copy_later)

    assert image_only_bounds[3] - image_only_bounds[1] > copy_later_bounds[3] - copy_later_bounds[1]
    assert copy_later_bounds[0] >= 500


def test_natural_textless_template_does_not_draw_large_blank_card() -> None:
    image_only = render_feature3_image(
        product_image(),
        template_id="natural",
        image_purpose="image_only",
        text_space="none",
    )
    copy_later = render_feature3_image(
        product_image(),
        template_id="natural",
        image_purpose="copy_later",
        text_space="right",
    )

    assert image_only.getpixel((100, 100))[0] < 250
    assert copy_later.getpixel((100, 100))[0] < 250


def test_request_position_and_size_are_applied_without_cropping() -> None:
    small_left = render_feature3_image(
        product_image(),
        image_request=image_request(product_position="left", product_size="small"),
    )
    large_right = render_feature3_image(
        product_image(),
        image_request=image_request(product_position="right", product_size="large"),
    )
    left_bounds = red_product_bounds(small_left)
    right_bounds = red_product_bounds(large_right)

    assert left_bounds[0] < 300
    assert right_bounds[2] > 780
    assert right_bounds[2] - right_bounds[0] > left_bounds[2] - left_bounds[0]
    assert min(*left_bounds) >= 0
    assert right_bounds[2] <= 1080
    assert right_bounds[3] <= 1080


def test_request_overrides_template_defaults() -> None:
    template_only = render_feature3_image(product_image(), template_id="pastel_pink")
    overridden = render_feature3_image(
        product_image(),
        template_id="pastel_pink",
        image_request=image_request(product_position="right", color_tone="dark_gold"),
    )

    assert red_product_bounds(template_only) != red_product_bounds(overridden)
    assert template_only.getpixel((0, 0)) != overridden.getpixel((0, 0))


def test_rejects_unknown_template() -> None:
    with pytest.raises(ValueError, match="지원하지 않는"):
        render_feature3_image(product_image(), template_id="unknown")


def test_composites_product_on_generated_background() -> None:
    background = Image.new("RGB", (1024, 1024), (20, 40, 60))

    result = composite_feature3_background(
        background,
        product_image(),
        product_position="right",
        product_size="large",
        margin="narrow",
        include_shadow=True,
    )

    assert result.size == (1080, 1080)
    assert result.mode == "RGB"
    bounds = red_product_bounds(result)
    assert bounds[2] > 780
    assert bounds[2] <= 1080
