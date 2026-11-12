"""기능 3의 문구 없는 템플릿 미리보기와 최종 이미지를 렌더링한다."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from src.shared.banner_generator import (
    BANNER_TEMPLATES,
    banner_product_box,
    generate_textless_banner,
)
from src.features.product_preserve.contract import FEATURE3_OUTPUT_SIZE


@dataclass(frozen=True)
class Feature3Template:
    template_id: str
    label: str
    description: str
    start_color: tuple[int, int, int]
    end_color: tuple[int, int, int]
    accent_color: tuple[int, int, int]
    background_mood: str
    color_tone: str
    product_position: str
    product_size: str
    margin: str
    include_shadow: bool


LEGACY_FEATURE3_TEMPLATES = (
    Feature3Template(
        "minimal_cream",
        "미니멀 크림",
        "따뜻한 크림색과 넉넉한 여백",
        (255, 251, 240),
        (239, 222, 198),
        (181, 139, 96),
        "warm_soft",
        "white_cream",
        "center",
        "default",
        "wide",
        True,
    ),
    Feature3Template(
        "clean_white",
        "클린 화이트",
        "깨끗한 흰색과 은은한 민트 포인트",
        (255, 255, 255),
        (225, 244, 239),
        (66, 151, 138),
        "bright_clean",
        "white_cream",
        "center",
        "large",
        "default",
        False,
    ),
    Feature3Template(
        "natural_beige",
        "내추럴 베이지",
        "베이지와 브라운의 편안한 분위기",
        (244, 231, 202),
        (211, 184, 143),
        (116, 102, 67),
        "warm_soft",
        "beige_brown",
        "right",
        "default",
        "default",
        True,
    ),
    Feature3Template(
        "pastel_pink",
        "파스텔 핑크",
        "부드러운 핑크와 라벤더 색감",
        (255, 232, 239),
        (224, 211, 245),
        (200, 104, 145),
        "warm_soft",
        "pink_lavender",
        "left",
        "default",
        "default",
        True,
    ),
    Feature3Template(
        "fresh_mint",
        "프레시 민트",
        "밝은 블루와 민트의 산뜻한 분위기",
        (221, 249, 245),
        (202, 227, 250),
        (38, 150, 142),
        "vivid_cheerful",
        "blue_mint",
        "right",
        "large",
        "narrow",
        True,
    ),
    Feature3Template(
        "premium_dark",
        "프리미엄 다크",
        "짙은 배경과 골드 포인트의 고급스러운 분위기",
        (47, 28, 48),
        (16, 28, 43),
        (205, 165, 85),
        "calm_premium",
        "dark_gold",
        "center",
        "default",
        "wide",
        True,
    ),
)

# 기능 1·2의 배너와 기능 3이 같은 스타일 이름과 안전 영역을 공유한다.
FEATURE3_TEMPLATES = (
    Feature3Template(
        "berry", "딸기 크림", BANNER_TEMPLATES[0].description,
        (255, 247, 244), (248, 199, 207), (205, 63, 98),
        "warm_soft", "pink_lavender", "right", "default", "default", True,
    ),
    Feature3Template(
        "cream", "미니멀 크림", BANNER_TEMPLATES[1].description,
        (255, 253, 246), (239, 224, 199), (181, 139, 96),
        "warm_soft", "white_cream", "center", "default", "wide", True,
    ),
    Feature3Template(
        "night", "프리미엄 나이트", BANNER_TEMPLATES[2].description,
        (31, 21, 42), (91, 38, 63), (235, 203, 137),
        "calm_premium", "dark_gold", "right", "default", "wide", True,
    ),
    Feature3Template(
        "clean", "클린 화이트", BANNER_TEMPLATES[3].description,
        (255, 255, 255), (234, 244, 244), (20, 120, 119),
        "bright_clean", "white_cream", "right", "large", "default", False,
    ),
    Feature3Template(
        "natural", "내추럴 베이커리", BANNER_TEMPLATES[4].description,
        (245, 235, 207), (218, 202, 157), (65, 83, 52),
        "warm_soft", "beige_brown", "center", "large", "default", True,
    ),
    Feature3Template(
        "promo", "프로모션 팝", BANNER_TEMPLATES[5].description,
        (255, 239, 149), (255, 203, 85), (221, 52, 54),
        "vivid_cheerful", "white_cream", "right", "large", "default", True,
    ),
)
FEATURE3_TEMPLATE_MAP = {
    template.template_id: template
    for template in (*FEATURE3_TEMPLATES, *LEGACY_FEATURE3_TEMPLATES)
}

COLOR_TONE_PALETTES = {
    "white_cream": ((255, 253, 247), (236, 227, 211), (173, 135, 93)),
    "beige_brown": ((244, 231, 202), (205, 177, 136), (112, 91, 62)),
    "pink_lavender": ((255, 231, 239), (222, 207, 246), (197, 99, 144)),
    "blue_mint": ((220, 249, 244), (198, 225, 250), (37, 148, 140)),
    "dark_gold": ((48, 29, 49), (15, 27, 42), (205, 166, 86)),
}
SIZE_RATIOS = {"small": 0.46, "default": 0.60, "large": 0.72}
MARGIN_PIXELS = {"narrow": 80, "default": 140, "wide": 205}


def _gradient(
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
) -> Image.Image:
    width, height = FEATURE3_OUTPUT_SIZE
    color_rows: list[tuple[int, int, int]] = []
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(
            round(start + (end - start) * ratio)
            for start, end in zip(start_color, end_color)
        )
        color_rows.append(color)
    strip = Image.new("RGB", (1, height))
    strip.putdata(color_rows)
    return strip.resize((width, height)).convert("RGBA")


def _decorate_background(
    canvas: Image.Image,
    *,
    style_id: str,
    mood: str,
    accent_color: tuple[int, int, int],
) -> None:
    width, height = canvas.size
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if style_id in {"clean_white", "clean"}:
        draw.ellipse(
            (width - 410, -150, width + 100, 360),
            fill=(*accent_color, 26),
        )
        draw.ellipse((-220, height - 330, 270, height + 150), fill=(255, 255, 255, 105))
    elif style_id in {"natural_beige", "natural"}:
        draw.rounded_rectangle(
            (-170, 150, 400, height - 90),
            radius=180,
            fill=(*accent_color, 23),
        )
        draw.arc((width - 500, -160, width + 130, 470), 75, 250, fill=(*accent_color, 70), width=9)
    elif style_id in {"pastel_pink", "berry"}:
        draw.ellipse((-140, -100, 400, 440), fill=(*accent_color, 24))
        draw.ellipse((width - 350, 120, width + 100, 570), fill=(255, 255, 255, 72))
        draw.ellipse((90, height - 250, 340, height), fill=(*accent_color, 28))
    elif style_id in {"fresh_mint", "promo"}:
        draw.ellipse((-130, height - 360, 350, height + 110), fill=(*accent_color, 45))
        draw.ellipse((width - 310, -100, width + 120, 330), fill=(255, 255, 255, 80))
        draw.ellipse((width - 180, 340, width + 30, 550), fill=(*accent_color, 34))
    elif style_id in {"premium_dark", "night"}:
        draw.ellipse((90, 110, width - 90, height - 70), outline=(*accent_color, 82), width=4)
        draw.arc((145, 165, width - 145, height - 125), 205, 335, fill=(*accent_color, 130), width=7)
    elif style_id in {"minimal_cream", "cream"} or mood == "warm_soft":
        draw.ellipse((-190, -130, 420, 480), fill=(*accent_color, 25))
        draw.ellipse(
            (width - 420, height - 390, width + 100, height + 100),
            fill=(255, 255, 255, 65),
        )
    elif mood == "calm_premium":
        draw.ellipse((90, 110, width - 90, height - 70), outline=(*accent_color, 82), width=4)
        draw.arc((145, 165, width - 145, height - 125), 205, 335, fill=(*accent_color, 130), width=7)
    else:
        draw.ellipse((-130, height - 360, 350, height + 110), fill=(*accent_color, 45))
        draw.ellipse((width - 310, -100, width + 120, 330), fill=(255, 255, 255, 80))
        draw.ellipse((width - 180, 340, width + 30, 550), fill=(*accent_color, 34))
    canvas.alpha_composite(overlay)


def _trim_transparent_edges(product_image: Image.Image) -> Image.Image:
    product = product_image.convert("RGBA")
    alpha_box = product.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("상품 이미지에 표시할 불투명 영역이 없습니다.")
    return product.crop(alpha_box)


def _scaled_product(
    product_image: Image.Image,
    *,
    size: str,
    margin: str,
) -> Image.Image:
    product = _trim_transparent_edges(product_image)
    canvas_width, canvas_height = FEATURE3_OUTPUT_SIZE
    ratio = SIZE_RATIOS[size]
    margin_pixels = MARGIN_PIXELS[margin]
    max_width = min(round(canvas_width * ratio), canvas_width - margin_pixels * 2)
    max_height = min(round(canvas_height * ratio), canvas_height - margin_pixels * 2)
    scale = min(max_width / product.width, max_height / product.height)
    target_size = (
        max(1, round(product.width * scale)),
        max(1, round(product.height * scale)),
    )
    return product.resize(target_size, Image.Resampling.LANCZOS)


def _paste_product(
    canvas: Image.Image,
    product_image: Image.Image,
    *,
    position: str,
    size: str,
    margin: str,
    include_shadow: bool,
) -> None:
    product = _scaled_product(product_image, size=size, margin=margin)
    canvas_width, canvas_height = canvas.size
    margin_pixels = MARGIN_PIXELS[margin]
    if position == "left":
        x = margin_pixels
    elif position == "right":
        x = canvas_width - margin_pixels - product.width
    else:
        x = (canvas_width - product.width) // 2
    y = (canvas_height - product.height) // 2

    if include_shadow:
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        alpha = Image.new("L", canvas.size, 0)
        alpha_draw = ImageDraw.Draw(alpha)
        contact_width = max(36, round(product.width * 0.68))
        contact_height = max(16, min(54, round(product.height * 0.075)))
        center_x = x + product.width // 2
        contact_y = min(canvas.height - 8, y + product.height)
        alpha_draw.ellipse(
            (
                center_x - contact_width // 2,
                contact_y - contact_height // 2,
                center_x + contact_width // 2,
                contact_y + contact_height // 2,
            ),
            fill=58,
        )
        alpha = alpha.filter(ImageFilter.GaussianBlur(max(12, contact_height // 2)))
        shadow.putalpha(alpha)
        canvas.alpha_composite(shadow)
    canvas.alpha_composite(product, (x, y))


def _resolve_render_settings(
    *,
    template_id: str | None,
    image_request: Mapping[str, Any] | None,
) -> tuple[Feature3Template, dict[str, Any]]:
    if template_id is not None and template_id not in FEATURE3_TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 기능 3 템플릿입니다: {template_id}")
    if template_id is None and image_request is None:
        raise ValueError("스타일 템플릿 또는 이미지 요청 사항이 필요합니다.")

    template = FEATURE3_TEMPLATE_MAP.get(template_id or "cream")
    assert template is not None
    settings: dict[str, Any] = {
        "background_mood": template.background_mood,
        "color_tone": template.color_tone,
        "product_position": template.product_position,
        "product_size": template.product_size,
        "margin": template.margin,
        "include_shadow": template.include_shadow,
    }
    if image_request is not None:
        settings.update(image_request)
    return template, settings


def feature3_product_box(
    product_image: Image.Image,
    *,
    template_id: str | None = None,
    image_request: Mapping[str, Any] | None = None,
    image_purpose: str = "image_only",
    text_space: str = "none",
) -> tuple[int, int, int, int]:
    """렌더링될 상품의 좌·상·우·하 좌표를 반환한다."""

    if template_id in {item.template_id for item in FEATURE3_TEMPLATES} and image_request is None:
        assert template_id is not None
        return banner_product_box(
            product_image,
            template_id=template_id,
            image_purpose=image_purpose,
            text_space=text_space,
        )

    _template, settings = _resolve_render_settings(
        template_id=template_id,
        image_request=image_request,
    )
    return feature3_product_box_for_settings(
        product_image,
        product_position=settings["product_position"],
        product_size=settings["product_size"],
        margin=settings["margin"],
    )


def feature3_product_box_for_settings(
    product_image: Image.Image,
    *,
    product_position: str,
    product_size: str,
    margin: str,
) -> tuple[int, int, int, int]:
    """상품 배치 설정으로 최종 캔버스의 상품 좌표를 계산한다."""

    if product_position not in {"left", "center", "right"}:
        raise ValueError(f"지원하지 않는 상품 위치입니다: {product_position}")
    if product_size not in SIZE_RATIOS:
        raise ValueError(f"지원하지 않는 상품 크기입니다: {product_size}")
    if margin not in MARGIN_PIXELS:
        raise ValueError(f"지원하지 않는 여백입니다: {margin}")

    product = _scaled_product(
        product_image,
        size=product_size,
        margin=margin,
    )
    canvas_width, canvas_height = FEATURE3_OUTPUT_SIZE
    margin_pixels = MARGIN_PIXELS[margin]
    if product_position == "left":
        x = margin_pixels
    elif product_position == "right":
        x = canvas_width - margin_pixels - product.width
    else:
        x = (canvas_width - product.width) // 2
    y = (canvas_height - product.height) // 2
    return (x, y, x + product.width, y + product.height)


def composite_feature3_background(
    background_image: Image.Image,
    product_image: Image.Image,
    *,
    product_position: str,
    product_size: str,
    margin: str,
    include_shadow: bool,
) -> Image.Image:
    """생성된 배경에 원본 상품 컷아웃을 Pillow로 합성한다."""

    if product_position not in {"left", "center", "right"}:
        raise ValueError(f"지원하지 않는 상품 위치입니다: {product_position}")
    if product_size not in SIZE_RATIOS:
        raise ValueError(f"지원하지 않는 상품 크기입니다: {product_size}")
    if margin not in MARGIN_PIXELS:
        raise ValueError(f"지원하지 않는 여백입니다: {margin}")
    if not isinstance(include_shadow, bool):
        raise ValueError("그림자 포함 여부는 참 또는 거짓이어야 합니다.")

    canvas = ImageOps.fit(
        background_image.convert("RGB"),
        FEATURE3_OUTPUT_SIZE,
        method=Image.Resampling.LANCZOS,
    ).convert("RGBA")
    _paste_product(
        canvas,
        product_image,
        position=product_position,
        size=product_size,
        margin=margin,
        include_shadow=include_shadow,
    )
    return canvas.convert("RGB")


def render_feature3_image(
    product_image: Image.Image,
    *,
    template_id: str | None = None,
    image_request: Mapping[str, Any] | None = None,
    image_purpose: str = "image_only",
    text_space: str = "none",
) -> Image.Image:
    """템플릿과 선택 요청으로 문구 없는 1080×1080 이미지를 만든다.

    템플릿과 요청을 함께 사용하면 요청 값이 템플릿 기본값보다 우선한다.
    """

    if template_id in {item.template_id for item in FEATURE3_TEMPLATES} and image_request is None:
        assert template_id is not None
        return generate_textless_banner(
            product_image,
            template_id=template_id,
            image_purpose=image_purpose,
            text_space=text_space,
        )

    template, settings = _resolve_render_settings(
        template_id=template_id,
        image_request=image_request,
    )

    if image_request is not None or template_id is None:
        start_color, end_color, accent_color = COLOR_TONE_PALETTES[settings["color_tone"]]
    else:
        start_color = template.start_color
        end_color = template.end_color
        accent_color = template.accent_color

    canvas = _gradient(start_color, end_color)
    _decorate_background(
        canvas,
        style_id=template.template_id if template_id is not None else "request_custom",
        mood=settings["background_mood"],
        accent_color=accent_color,
    )
    _paste_product(
        canvas,
        product_image,
        position=settings["product_position"],
        size=settings["product_size"],
        margin=settings["margin"],
        include_shadow=settings["include_shadow"],
    )
    return canvas.convert("RGB")


def feature3_image_to_png_bytes(image: Image.Image) -> bytes:
    if image.size != FEATURE3_OUTPUT_SIZE:
        raise ValueError("기능 3 결과는 1080×1080px이어야 합니다.")
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
