from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
import pytest

from src.shared.banner_generator import (
    BANNER_DESIGN_VERSION,
    BANNER_DESIGN_RULES,
    BANNER_INFORMATION_PRIORITY,
    BANNER_TEMPLATES,
    CANVAS_SIZE,
    FONT_PATH,
    HEADLINE_MAX_FONT_SIZE,
    PRODUCT_NAME_MIN_FONT_SIZE,
    PRODUCT_IMAGE_PRESERVATION_RULES,
    TEXT_SAFE_AREA_RULES,
    _balanced_two_lines,
    _fit_single_line,
    _fit_text,
    _price_labels,
    banner_to_png_bytes,
    banner_product_box,
    banner_text_safe_areas,
    generate_banner,
    generate_textless_banner,
    resolve_auto_text_space,
    template_decoration_exclusion_boxes,
    wrap_text,
)


def test_banner_design_spec_uses_square_canvas_and_clear_information_priority() -> None:
    assert CANVAS_SIZE == (1080, 1080)
    assert BANNER_INFORMATION_PRIORITY == (
        "product_name",
        "headline",
        "product_image",
        "body_when_distinct_fact_exists",
        "price_or_promotion",
        "store_name",
    )
    assert BANNER_DESIGN_VERSION == "v0.9.1"
    assert any("상품 이미지" in rule for rule in BANNER_DESIGN_RULES)
    assert any("영어 문구" in rule for rule in BANNER_DESIGN_RULES)
    assert PRODUCT_NAME_MIN_FONT_SIZE > HEADLINE_MAX_FONT_SIZE
    assert any("RGB" in rule for rule in PRODUCT_IMAGE_PRESERVATION_RULES)
    assert any("안전 영역" in rule for rule in TEXT_SAFE_AREA_RULES)


def _boxes_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _sample_cutout() -> Image.Image:
    image = Image.new("RGBA", (360, 520), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 40, 290, 500), radius=80, fill=(235, 92, 120, 255))
    return image


def _sample_copy() -> dict[str, object]:
    return {
        "headline": "생딸기의 향, 한 잔에 담다",
        "body": "생딸기를 사용한 딸기 라떼를 감성적인 분위기와 함께 즐겨보세요.",
        "hashtags": ["#딸기라떼", "#생딸기"],
    }


@pytest.mark.parametrize("template_id", [item.template_id for item in BANNER_TEMPLATES])
@pytest.mark.parametrize(
    ("image_purpose", "text_space"),
    [("image_only", "none"), ("copy_later", "auto")],
)
def test_textless_banner_preserves_product_and_reserves_copy_area(
    template_id: str,
    image_purpose: str,
    text_space: str,
) -> None:
    product = _sample_cutout()
    result = generate_textless_banner(
        product,
        template_id=template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    product_box = banner_product_box(
        product,
        template_id=template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    safe_areas = banner_text_safe_areas(
        template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )

    assert result.size == CANVAS_SIZE
    assert bool(safe_areas) is (image_purpose == "copy_later")
    assert all(not _boxes_overlap(product_box, area) for area in safe_areas)

    source = product.crop(product.getchannel("A").getbbox()).resize(
        (product_box[2] - product_box[0], product_box[3] - product_box[1]),
        Image.Resampling.LANCZOS,
    )
    output = result.crop(product_box).convert("RGB")
    source_rgb = source.convert("RGB")
    for expected, alpha, actual in zip(
        source_rgb.get_flattened_data(),
        source.getchannel("A").get_flattened_data(),
        output.get_flattened_data(),
    ):
        if alpha == 255:
            assert actual == expected


@pytest.mark.parametrize(("template_id", "edge_x"), [("berry", 58), ("night", 60)])
def test_textless_background_has_no_visible_empty_card(
    template_id: str,
    edge_x: int,
) -> None:
    result = generate_textless_banner(_sample_cutout(), template_id=template_id)
    differences = []
    for y in range(180, 901, 20):
        left = result.getpixel((edge_x - 3, y))
        right = result.getpixel((edge_x + 3, y))
        differences.extend(abs(a - b) for a, b in zip(left, right))

    assert sum(differences) / len(differences) <= 3.0


@pytest.mark.parametrize("template_id", [item.template_id for item in BANNER_TEMPLATES])
def test_generate_banner_for_every_template(template_id: str) -> None:
    banner = generate_banner(
        _sample_cutout(),
        _sample_copy(),  # type: ignore[arg-type]
        store_name="오늘의 작은 카페",
        product_name="딸기 라떼",
        base_price="7,000원",
        sale_price="6,000원",
        promotion="이번 주말 10% 할인",
        template_id=template_id,
    )
    png_bytes = banner_to_png_bytes(banner)
    reopened = Image.open(BytesIO(png_bytes))

    assert banner.size == CANVAS_SIZE
    assert banner.mode == "RGB"
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert reopened.size == CANVAS_SIZE


@pytest.mark.parametrize("template_id", [item.template_id for item in BANNER_TEMPLATES])
def test_generate_banner_hides_empty_optional_body(template_id: str) -> None:
    copy = _sample_copy()
    copy["body"] = ""

    banner = generate_banner(
        _sample_cutout(),
        copy,  # type: ignore[arg-type]
        store_name="오늘의 작은 카페",
        product_name="딸기 라떼",
        base_price="7,000원",
        promotion=None,
        call_to_action=None,
        template_id=template_id,
    )

    assert banner.size == CANVAS_SIZE


def test_wrap_text_stays_within_pixel_width() -> None:
    font = ImageFont.truetype(str(FONT_PATH), size=24)
    lines = wrap_text("a long advertising headline", font, max_width=95)

    assert len(lines) > 1
    assert all(font.getlength(line) <= 95 for line in lines)


def test_wrap_text_keeps_words_together_when_each_word_fits() -> None:
    font = ImageFont.truetype(str(FONT_PATH), size=48)

    lines = wrap_text("매일 굽는 바게트", font, max_width=280)

    assert "바게트" in lines
    assert "바게" not in lines
    assert "트" not in lines


def test_headline_prefers_two_balanced_lines_when_it_can_shrink() -> None:
    _, lines = _fit_text(
        "우아한 겉바속촉, 까눌레",
        max_width=455,
        max_lines=3,
        max_size=76,
        min_size=42,
        weight=800,
        preferred_max_lines=2,
    )

    assert len(lines) <= 2


def test_balanced_two_lines_avoids_orphan_word() -> None:
    font = ImageFont.truetype(str(FONT_PATH), size=80)

    lines = _balanced_two_lines(
        "나를 깨우는 맑고 깊은 여운",
        font,
        max_width=870,
    )

    assert lines == ["나를 깨우는", "맑고 깊은 여운"]


def test_promotion_shrinks_before_using_ellipsis() -> None:
    _, label = _fit_single_line(
        "평일 오전 테이크아웃 고객 500원 할인",
        max_width=401,
        max_size=26,
        min_size=16,
        weight=650,
    )

    assert label == "평일 오전 테이크아웃 고객 500원 할인"


def test_generate_banner_rejects_unknown_template() -> None:
    with pytest.raises(ValueError, match="지원하지 않는"):
        generate_banner(
            _sample_cutout(),
            _sample_copy(),  # type: ignore[arg-type]
            store_name=None,
            product_name="딸기 라떼",
            base_price=None,
            sale_price=None,
            promotion=None,
            template_id="unknown",
        )


@pytest.mark.parametrize("template_id", [item.template_id for item in BANNER_TEMPLATES])
def test_generate_banner_handles_long_korean_text(template_id: str) -> None:
    long_copy = {
        "headline": "가을 오후를 위한 부드럽고 향긋한 생딸기 크림 라떼 한 잔",
        "body": (
            "생딸기와 부드러운 크림을 사용한 딸기 라떼를 따뜻하고 감성적인 "
            "분위기 속에서 여유롭게 즐겨보세요."
        ),
        "hashtags": ["#딸기라떼", "#생딸기"],
    }

    banner = generate_banner(
        _sample_cutout(),
        long_copy,  # type: ignore[arg-type]
        store_name="우리 동네에서 가장 따뜻한 오늘의 작은 카페",
        product_name="생딸기 수제 크림 스페셜 라떼",
        base_price="12,500원",
        sale_price="10,000원",
        promotion="이번 주말 방문 고객에게만 제공되는 특별 행사",
        template_id=template_id,
    )

    assert banner.size == CANVAS_SIZE


@pytest.mark.parametrize("product_size", [(900, 220), (220, 900), (80, 80)])
def test_generate_banner_handles_product_aspect_ratios(product_size: tuple[int, int]) -> None:
    product = Image.new("RGBA", product_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(product)
    draw.ellipse((0, 0, product_size[0] - 1, product_size[1] - 1), fill=(220, 90, 80, 255))

    banner = generate_banner(
        product,
        _sample_copy(),  # type: ignore[arg-type]
        store_name="오늘의 작은 카페",
        product_name="테스트 상품",
        base_price=None,
        sale_price=None,
        promotion=None,
        template_id="cream",
    )

    assert banner.size == CANVAS_SIZE


def test_price_labels_make_base_sale_and_promotion_relationship_explicit() -> None:
    assert _price_labels("4,500원", "4,000원", "평일 오전 500원 할인") == (
        "기본가 4,500원",
        "할인가 4,000원",
    )
    assert _price_labels("4,500원", None, "평일 오전 500원 할인") == (
        "가격 4,500원",
        None,
    )
    assert _price_labels("6,500원", None, None) == ("가격 6,500원", None)


def test_natural_template_uses_a_light_information_card() -> None:
    banner = generate_banner(
        _sample_cutout(),
        _sample_copy(),  # type: ignore[arg-type]
        store_name="매일 굽는 작은 베이커리",
        product_name="카다멈 번",
        base_price="4,500원",
        sale_price=None,
        promotion=None,
        call_to_action="매장에서 메뉴를 만나보세요.",
        template_id="natural",
    )

    red, green, blue = banner.getpixel((600, 1000))
    assert min(red, green, blue) > 180


def test_generate_banner_rejects_sale_price_without_base_price() -> None:
    with pytest.raises(ValueError, match="기본 가격"):
        generate_banner(
            _sample_cutout(),
            _sample_copy(),  # type: ignore[arg-type]
            store_name="오늘의 작은 카페",
            product_name="딸기 라떼",
            base_price=None,
            sale_price="5,000원",
            promotion=None,
            template_id="berry",
        )


def test_auto_text_space_uses_horizontal_layout_for_wide_cutout() -> None:
    wide = Image.new("RGBA", (620, 180), (210, 120, 70, 255))

    assert resolve_auto_text_space(
        wide,
        template_id="berry",
        image_purpose="copy_later",
        text_space="auto",
    ) == "top"


def test_auto_text_space_centers_portrait_cutout_below_top_copy() -> None:
    portrait = Image.new("RGBA", (220, 720), (210, 120, 70, 255))

    text_space = resolve_auto_text_space(
        portrait,
        template_id="clean",
        image_purpose="copy_later",
        text_space="auto",
    )
    product_box = banner_product_box(
        portrait,
        template_id="clean",
        image_purpose="copy_later",
        text_space="auto",
    )

    assert text_space == "top"
    assert abs((product_box[0] + product_box[2]) / 2 - CANVAS_SIZE[0] / 2) <= 1


def test_decoration_zones_never_overlap_text_safe_area() -> None:
    for template in BANNER_TEMPLATES:
        for text_space in ("left", "right", "top", "bottom"):
            safe_area = banner_text_safe_areas(
                template.template_id,
                image_purpose="copy_later",
                text_space=text_space,
            )[0]
            decorations = template_decoration_exclusion_boxes(
                template.template_id,
                image_purpose="copy_later",
                text_space=text_space,
            )
            assert all(not _boxes_overlap(safe_area, box) for box in decorations)
