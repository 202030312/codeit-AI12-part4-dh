"""배경 제거 상품 이미지와 검증된 문구를 SNS 배너로 합성한다."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.features.ad_copy.generator import AdCopy


CANVAS_SIZE = (1080, 1080)
BANNER_DESIGN_VERSION = "v0.9.1"
BANNER_INFORMATION_PRIORITY = (
    "product_name",
    "headline",
    "product_image",
    "body_when_distinct_fact_exists",
    "price_or_promotion",
    "store_name",
)
BANNER_DESIGN_RULES = (
    "소상공인이 SNS와 매장 홍보에 사용할 수 있는 친근하고 명확한 광고물",
    "글자는 적고 크게 표시",
    "상품 이미지를 가장 눈에 띄게 표시",
    "선택한 광고 분위기에 맞는 색상 사용",
    "정보의 시각적 우선순위를 한눈에 구분",
    "의미 없는 영어 문구와 장식 제외",
    "작은 글씨의 과도한 사용 금지",
    "장식이 상품 사진이나 글자를 가리지 않도록 구성",
    "상품명은 광고 제목보다 크게 표시",
)
PRODUCT_IMAGE_PRESERVATION_RULES = (
    "상품은 생성형 모델로 다시 그리지 않음",
    "BiRefNet은 상품과 배경을 분리하는 용도로만 사용",
    "상품 내부의 RGB 픽셀은 원본 값을 유지",
    "상품의 색상·질감·형태·로고·포장을 변경하지 않음",
    "상품 종횡비를 유지하고 비례 확대·축소만 허용",
    "불투명한 상품 영역을 자르지 않음",
    "필터와 배경 효과는 상품 레이어에 적용하지 않음",
    "그림자는 상품 알파 마스크에서 별도 생성",
)
TEXT_SAFE_AREA_RULES = (
    "상품 배치 영역과 문구 안전 영역은 서로 겹치지 않음",
    "문구 없는 이미지에서도 이후 문구를 넣을 영역을 비워 둠",
    "문구와 장식은 상품의 불투명 영역을 가리지 않음",
)
HEADLINE_MAX_FONT_SIZE = 44
HEADLINE_MIN_FONT_SIZE = 30
PRODUCT_NAME_MAX_FONT_SIZE = 56
PRODUCT_NAME_MIN_FONT_SIZE = 46
FONT_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "fonts"
    / "NotoSansKR-Variable.ttf"
)


@dataclass(frozen=True)
class BannerTemplate:
    template_id: str
    label: str
    description: str
    product_box: tuple[int, int, int, int]
    text_safe_areas: tuple[tuple[int, int, int, int], ...]
    image_only_product_box: tuple[int, int, int, int]
    default_text_space: str
    shadow_opacity: int = 75
    scale_boost: float = 1.0
    portrait_scale: float = 1.0


BANNER_TEMPLATES = (
    BannerTemplate(
        "berry",
        "딸기 크림",
        "밝은 분홍색의 친근하고 감성적인 구성",
        (520, 205, 1065, 1020),
        ((58, 55, 500, 1018),),
        (130, 100, 950, 1030),
        "left",
        scale_boost=1.06,
        portrait_scale=0.94,
    ),
    BannerTemplate(
        "cream",
        "미니멀 크림",
        "여백이 넉넉한 따뜻하고 차분한 구성",
        (165, 315, 915, 835),
        ((80, 45, 1000, 290), (80, 850, 1000, 1030)),
        (130, 120, 950, 980),
        "top",
        shadow_opacity=65,
    ),
    BannerTemplate(
        "night",
        "프리미엄 나이트",
        "짙은 자주색과 금색 포인트의 고급스러운 구성",
        (540, 205, 1065, 1020),
        ((60, 62, 520, 1018),),
        (130, 100, 950, 1030),
        "left",
        shadow_opacity=90,
        scale_boost=1.06,
    ),
    BannerTemplate(
        "clean",
        "클린 화이트",
        "흰색과 청록색 포인트의 깔끔한 제품 중심 구성",
        (475, 175, 1050, 1005),
        ((30, 45, 455, 1030),),
        (130, 100, 950, 1030),
        "left",
        shadow_opacity=60,
        scale_boost=1.04,
        portrait_scale=0.95,
    ),
    BannerTemplate(
        "natural",
        "내추럴 베이커리",
        "베이지와 올리브색의 따뜻한 베이커리 구성",
        (95, 305, 985, 835),
        ((80, 45, 1000, 285), (70, 850, 1010, 1020)),
        (110, 120, 970, 970),
        "top",
        shadow_opacity=60,
        scale_boost=1.02,
    ),
    BannerTemplate(
        "promo",
        "프로모션 팝",
        "노란색과 빨간색으로 행사와 가격을 강조한 구성",
        (475, 195, 1055, 1025),
        ((40, 40, 455, 1030),),
        (130, 100, 950, 1030),
        "left",
        shadow_opacity=70,
        scale_boost=1.06,
        portrait_scale=0.94,
    ),
)
TEMPLATE_MAP = {template.template_id: template for template in BANNER_TEMPLATES}

COPY_LATER_LAYOUTS = {
    "left": ((475, 145, 1025, 1025), ((55, 75, 455, 1005),)),
    "right": ((55, 145, 605, 1025), ((625, 75, 1025, 1005),)),
    "top": ((125, 345, 955, 1025), ((65, 55, 1015, 325),)),
    "bottom": ((125, 55, 955, 725), ((65, 755, 1015, 1025),)),
}


def _font(size: int, weight: int = 500) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            "배너용 한글 폰트를 찾을 수 없습니다: assets/fonts/NotoSansKR-Variable.ttf"
        )
    font = ImageFont.truetype(str(FONT_PATH), size=size)
    if hasattr(font, "set_variation_by_axes"):
        try:
            font.set_variation_by_axes([float(weight)])
        except (OSError, ValueError):
            pass
    return font


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """단어를 우선 보존하고 긴 단어만 글자 단위로 줄바꿈한다."""

    normalized = " ".join(text.split())
    if not normalized:
        return []

    lines: list[str] = []
    current = ""
    for word in normalized.split(" "):
        candidate = f"{current} {word}" if current else word
        if font.getlength(candidate) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        remaining = word
        while remaining and font.getlength(remaining) > max_width:
            prefix = ""
            for character in remaining:
                next_prefix = prefix + character
                if prefix and font.getlength(next_prefix) > max_width:
                    break
                prefix = next_prefix
            lines.append(prefix)
            remaining = remaining[len(prefix) :]
        current = remaining
    if current:
        lines.append(current)
    return lines


def _balanced_two_lines(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str] | None:
    """두 줄 제목의 한쪽에 짧은 단어만 남지 않도록 가장 균형 있는 분할을 찾는다."""
    words = " ".join(text.split()).split(" ")
    if len(words) < 2:
        return None

    candidates: list[tuple[float, float, list[str]]] = []
    for split_at in range(1, len(words)):
        lines = [" ".join(words[:split_at]), " ".join(words[split_at:])]
        widths = [font.getlength(line) for line in lines]
        if max(widths) <= max_width:
            candidates.append((abs(widths[0] - widths[1]), max(widths), lines))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _truncate_lines(
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    if len(lines) <= max_lines:
        return list(lines)

    truncated = list(lines[:max_lines])
    final_line = truncated[-1].rstrip()
    while final_line and font.getlength(final_line + "…") > max_width:
        final_line = final_line[:-1].rstrip()
    truncated[-1] = final_line + "…"
    return truncated


def _fit_text(
    text: str,
    *,
    max_width: int,
    max_lines: int,
    max_size: int,
    min_size: int,
    weight: int,
    preferred_max_lines: int | None = None,
    balance_two_lines: bool = False,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    fallback: tuple[ImageFont.FreeTypeFont, list[str]] | None = None
    for size in range(max_size, min_size - 1, -2):
        font = _font(size, weight)
        lines = wrap_text(text, font, max_width)
        if balance_two_lines and len(lines) == 2:
            balanced = _balanced_two_lines(text, font, max_width)
            if balanced is not None:
                lines = balanced
        if len(lines) <= max_lines:
            if fallback is None:
                fallback = (font, lines)
            if preferred_max_lines is None or len(lines) <= preferred_max_lines:
                return font, lines
    if fallback is not None:
        return fallback
    font = _font(min_size, weight)
    return font, _truncate_lines(wrap_text(text, font, max_width), font, max_width, max_lines)


def _fit_single_line(
    text: str,
    *,
    max_width: int,
    max_size: int,
    min_size: int,
    weight: int,
) -> tuple[ImageFont.FreeTypeFont, str]:
    normalized = " ".join(text.split())
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, weight)
        if font.getlength(normalized) <= max_width:
            return font, normalized

    font = _font(min_size, weight)
    label = normalized
    while label and font.getlength(label + "…") > max_width:
        label = label[:-1].rstrip()
    return font, label + "…"


def _gradient(start: tuple[int, int, int], end: tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, start)
    draw = ImageDraw.Draw(canvas)
    for y in range(CANVAS_SIZE[1]):
        ratio = y / (CANVAS_SIZE[1] - 1)
        color = tuple(round(a + (b - a) * ratio) for a, b in zip(start, end))
        draw.line((0, y, CANVAS_SIZE[0], y), fill=color)
    return canvas.convert("RGBA")


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, ...],
    spacing: int,
    align: str = "left",
    area_width: int | None = None,
) -> int:
    current_y = y
    for line in lines:
        line_width = font.getlength(line)
        line_x = x
        if align == "center" and area_width is not None:
            line_x = x + (area_width - line_width) / 2
        draw.text((line_x, current_y), line, font=font, fill=fill)
        box = font.getbbox(line or "가")
        current_y += box[3] - box[1] + spacing
    return current_y


def _paste_product(
    canvas: Image.Image,
    product_image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    shadow_opacity: int = 75,
    scale_boost: float = 1.0,
    portrait_scale: float = 1.0,
) -> None:
    product, x, y = _prepare_product_placement(
        product_image,
        box,
        scale_boost=scale_boost,
        portrait_scale=portrait_scale,
    )

    # 제품 하단에 집중된 접지 그림자를 먼저 넣어 바닥에서 뜨 보이지 않게 한다.
    contact_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    contact = Image.new("L", canvas.size, 0)
    contact_draw = ImageDraw.Draw(contact)
    contact_width = max(36, round(product.width * 0.68))
    contact_height = max(16, min(54, round(product.height * 0.075)))
    center_x = x + product.width // 2
    contact_y = min(canvas.height - 8, y + product.height - contact_height // 3)
    contact_draw.ellipse(
        (
            center_x - contact_width // 2,
            contact_y - contact_height // 2,
            center_x + contact_width // 2,
            contact_y + contact_height // 2,
        ),
        fill=min(100, max(45, shadow_opacity // 2 + 24)),
    )
    contact = contact.filter(ImageFilter.GaussianBlur(max(12, contact_height // 2)))
    contact_layer.putalpha(contact)
    canvas.alpha_composite(contact_layer)

    canvas.alpha_composite(product, (x, y))


def _prepare_product_placement(
    product_image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    scale_boost: float = 1.0,
    portrait_scale: float = 1.0,
) -> tuple[Image.Image, int, int]:
    """원본 종횡비와 RGB를 유지한 채 상품의 크기와 좌표만 계산한다."""

    product = product_image.convert("RGBA")
    alpha_box = product.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("상품 이미지에 보이는 영역이 없습니다.")
    product = product.crop(alpha_box)

    left, top, right, bottom = box
    max_width = right - left
    max_height = bottom - top
    orientation_scale = (
        portrait_scale if product.height / product.width >= 1.3 else 1.0
    )
    fit_scale = min(max_width / product.width, max_height / product.height)
    # scale_boost를 사용하더라도 지정 영역보다 커져 제품 가장자리가 잘리지 않게 한다.
    # 원본 종횡비는 그대로 유지하며 템플릿은 제품 영역 자체를 넓혀 강조한다.
    scale = min(fit_scale * scale_boost * orientation_scale, fit_scale)
    resized_size = (
        max(1, round(product.width * scale)),
        max(1, round(product.height * scale)),
    )
    product = product.resize(resized_size, Image.Resampling.LANCZOS)
    x = left + (max_width - product.width) // 2
    y = top + (max_height - product.height) // 2
    return product, x, y


def banner_product_box(
    product_image: Image.Image,
    *,
    template_id: str,
    image_purpose: str = "image_only",
    text_space: str = "none",
) -> tuple[int, int, int, int]:
    """템플릿에서 실제로 차지하는 상품의 좌·상·우·하 좌표를 반환한다."""

    if template_id not in TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")
    template = TEMPLATE_MAP[template_id]
    text_space = resolve_auto_text_space(
        product_image,
        template_id=template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    product_box, _safe_areas = resolve_textless_layout(
        template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    product, x, y = _prepare_product_placement(
        product_image,
        product_box,
        scale_boost=template.scale_boost,
        portrait_scale=template.portrait_scale,
    )
    return (x, y, x + product.width, y + product.height)


def resolve_auto_text_space(
    product_image: Image.Image,
    *,
    template_id: str,
    image_purpose: str,
    text_space: str,
) -> str:
    """자동 문구 배치는 상단을 사용해 상품을 가로 중앙에 놓는다."""

    if image_purpose != "copy_later" or text_space != "auto":
        return text_space
    if template_id not in TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")
    product = product_image.convert("RGBA")
    alpha_box = product.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("상품 이미지에 보이는 영역이 없습니다.")
    return "top"


def banner_text_safe_areas(
    template_id: str,
    *,
    image_purpose: str = "copy_later",
    text_space: str = "auto",
) -> tuple[tuple[int, int, int, int], ...]:
    """상품을 가리지 않고 문구를 배치할 수 있는 고정 영역을 반환한다."""

    if template_id not in TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")
    _product_box, safe_areas = resolve_textless_layout(
        template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    return safe_areas


def resolve_textless_layout(
    template_id: str,
    *,
    image_purpose: str,
    text_space: str,
) -> tuple[
    tuple[int, int, int, int],
    tuple[tuple[int, int, int, int], ...],
]:
    """이미지 단독/문구 추가 예정 목적에 맞는 제품·안전 영역을 정한다."""

    if template_id not in TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")
    if image_purpose not in {"image_only", "copy_later"}:
        raise ValueError(f"지원하지 않는 이미지 사용 목적입니다: {image_purpose}")
    if text_space not in {"auto", "top", "bottom", "left", "right", "none"}:
        raise ValueError(f"지원하지 않는 문구 여백입니다: {text_space}")

    template = TEMPLATE_MAP[template_id]
    if image_purpose == "image_only":
        if text_space not in {"auto", "none"}:
            raise ValueError("이미지 단독 사용에서는 문구 여백을 지정할 수 없습니다.")
        return template.image_only_product_box, ()

    if text_space == "none":
        raise ValueError("나중에 문구를 추가하려면 문구 여백이 필요합니다.")
    resolved_text_space = (
        "top" if text_space == "auto" else text_space
    )
    return COPY_LATER_LAYOUTS[resolved_text_space]


def _boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _decoration_is_safe(
    box: tuple[int, int, int, int],
    safe_areas: Sequence[tuple[int, int, int, int]],
) -> bool:
    return all(not _boxes_overlap(box, area) for area in safe_areas)


def template_decoration_exclusion_boxes(
    template_id: str,
    *,
    image_purpose: str = "copy_later",
    text_space: str = "auto",
) -> tuple[tuple[int, int, int, int], ...]:
    """문구 안전 영역과 겹치지 않게 실제로 그려지는 장식 경계를 반환한다."""

    _product_box, safe_areas = resolve_textless_layout(
        template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    candidates: dict[str, tuple[tuple[int, int, int, int], ...]] = {
        "berry": ((790, -120, 1135, 225), (-150, 820, 210, 1180)),
        "cream": ((300, 350, 780, 830),),
        "night": (),
        "clean": ((585, 175, 1110, 700), (845, -105, 1135, 185)),
        "natural": ((-90, 790, 215, 1095), (895, -85, 1140, 160)),
        "promo": ((790, -70, 1110, 250),),
    }
    return tuple(
        box for box in candidates.get(template_id, ())
        if _decoration_is_safe(box, safe_areas)
    )


def _textless_template_background(
    template_id: str,
    *,
    safe_areas: Sequence[tuple[int, int, int, int]] = (),
) -> Image.Image:
    """장식과 문구 안전 영역이 겹치지 않는 문구 없는 배경을 만든다."""

    if template_id == "berry":
        canvas = _gradient((255, 247, 244), (248, 199, 207))
        draw = ImageDraw.Draw(canvas, "RGBA")
        if _decoration_is_safe((790, -120, 1135, 225), safe_areas):
            draw.ellipse((790, -120, 1135, 225), fill=(255, 255, 255, 78))
        if _decoration_is_safe((-150, 820, 210, 1180), safe_areas):
            draw.ellipse((-150, 820, 210, 1180), fill=(205, 63, 98, 35))
        return canvas
    if template_id == "cream":
        canvas = _gradient((255, 253, 246), (239, 224, 199))
        draw = ImageDraw.Draw(canvas, "RGBA")
        if _decoration_is_safe((300, 350, 780, 830), safe_areas):
            draw.ellipse((300, 350, 780, 830), fill=(255, 255, 255, 125))
        return canvas
    if template_id == "night":
        canvas = _gradient((31, 21, 42), (91, 38, 63))
        return canvas
    if template_id == "clean":
        canvas = _gradient((255, 255, 255), (234, 244, 244))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((0, 0, 18, 1080), fill=(20, 120, 119, 255))
        if _decoration_is_safe((585, 175, 1110, 700), safe_areas):
            draw.ellipse((585, 175, 1110, 700), fill=(213, 239, 237, 210))
        if _decoration_is_safe((845, -105, 1135, 185), safe_areas):
            draw.ellipse((845, -105, 1135, 185), fill=(20, 120, 119, 14))
        return canvas
    if template_id == "natural":
        canvas = _gradient((245, 235, 207), (218, 202, 157))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.ellipse((270, 180, 860, 820), fill=(255, 252, 239, 58))
        if _decoration_is_safe((-90, 790, 215, 1095), safe_areas):
            draw.ellipse((-90, 790, 215, 1095), fill=(96, 113, 69, 36))
        if _decoration_is_safe((895, -85, 1140, 160), safe_areas):
            draw.ellipse((895, -85, 1140, 160), fill=(174, 92, 58, 25))
        return canvas
    if template_id == "promo":
        canvas = _gradient((255, 239, 149), (255, 203, 85))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.polygon(
            ((0, 0), (690, 0), (520, 1080), (0, 1080)),
            fill=(255, 250, 228, 238),
        )
        if _decoration_is_safe((790, -70, 1110, 250), safe_areas):
            draw.ellipse((790, -70, 1110, 250), fill=(221, 52, 54, 210))
            draw.ellipse((840, 20, 1015, 195), outline=(255, 250, 228, 155), width=5)
        return canvas
    raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")


def generate_textless_banner(
    product_image: Image.Image,
    *,
    template_id: str,
    image_purpose: str = "image_only",
    text_space: str = "none",
) -> Image.Image:
    """원본 상품을 보존하고 문구 안전 영역을 비운 1080×1080 이미지를 만든다."""

    if template_id not in TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")
    template = TEMPLATE_MAP[template_id]
    text_space = resolve_auto_text_space(
        product_image,
        template_id=template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    product_box, _safe_areas = resolve_textless_layout(
        template_id,
        image_purpose=image_purpose,
        text_space=text_space,
    )
    canvas = _textless_template_background(template_id, safe_areas=_safe_areas)
    _paste_product(
        canvas,
        product_image,
        product_box,
        shadow_opacity=template.shadow_opacity,
        scale_boost=template.scale_boost,
        portrait_scale=template.portrait_scale,
    )
    return canvas.convert("RGB")


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    max_font_size: int,
    background: tuple[int, ...],
    foreground: tuple[int, ...],
    max_width: int,
    min_font_size: int = 16,
    weight: int = 650,
) -> int:
    font, lines = _fit_text(
        text,
        max_width=max_width - 44,
        max_lines=3,
        max_size=max_font_size,
        min_size=min_font_size,
        weight=weight,
        preferred_max_lines=3,
    )
    if any("…" in line for line in lines):
        raise ValueError(
            "프로모션 조건을 줄이지 않고 표시할 수 없습니다. 행사 문구를 60자 이내로 입력해주세요."
        )
    line_height = font.getbbox("가")[3] - font.getbbox("가")[1]
    width = min(max_width, round(max(font.getlength(line) for line in lines)) + 44)
    height = line_height * len(lines) + 8 * (len(lines) - 1) + 24
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=min(height // 2, 28),
        fill=background,
    )
    _draw_lines(
        draw,
        lines,
        x=x + 22,
        y=y + 9,
        font=font,
        fill=foreground,
        spacing=8,
    )
    return height


def _draw_store_name(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, ...],
    align: str = "left",
) -> None:
    """SNS 축소 화면에서도 읽히도록 매장명을 충분한 크기로 표시한다."""
    font, label = _fit_single_line(
        text,
        max_width=max_width,
        max_size=34,
        min_size=24,
        weight=700,
    )
    draw_x = x
    if align == "center":
        draw_x = x + (max_width - font.getlength(label)) / 2
    draw.text((draw_x, y), label, font=font, fill=fill)


def _draw_product_name(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, ...],
    align: str = "left",
    area_width: int | None = None,
) -> int:
    """광고 제목보다 크게 보이는 메뉴명을 최대 두 줄로 표시한다."""
    font, lines = _fit_text(
        text,
        max_width=max_width,
        max_lines=2,
        max_size=PRODUCT_NAME_MAX_FONT_SIZE,
        min_size=PRODUCT_NAME_MIN_FONT_SIZE,
        weight=800,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    return _draw_lines(
        draw,
        lines,
        x=x,
        y=y,
        font=font,
        fill=fill,
        spacing=8,
        align=align,
        area_width=area_width,
    )


def _price_labels(
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
) -> tuple[str | None, str | None]:
    """기본가·할인가의 관계가 명확한 배너 라벨을 만든다."""
    if sale_price:
        return (
            f"기본가 {base_price}" if base_price else None,
            f"할인가 {sale_price}",
        )
    if base_price:
        return (f"가격 {base_price}", None)
    return (None, None)


def _draw_call_to_action(
    draw: ImageDraw.ImageDraw,
    text: str | None,
    *,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, ...],
) -> None:
    """방문·주문 안내를 다른 정보와 섞이지 않는 한 줄로 표시한다."""
    if not text:
        return
    font, label = _fit_single_line(
        f"→ {text}",
        max_width=max_width,
        max_size=22,
        min_size=16,
        weight=650,
    )
    draw.text((x, y), label, font=font, fill=fill)


def _draw_price_block(
    draw: ImageDraw.ImageDraw,
    *,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    anchor_x: int,
    y: int,
    max_width: int,
    foreground: tuple[int, ...],
    align: str = "left",
    max_size: int = 58,
) -> None:
    """기본가와 할인가를 한 블록으로 그려 의미가 뒤섞이지 않게 한다."""
    base_label, sale_label = _price_labels(base_price, sale_price, promotion)
    if not base_label and not sale_label:
        return

    if sale_label:
        base_font, fitted_base = _fit_single_line(
            base_label or "",
            max_width=max_width,
            max_size=29,
            min_size=20,
            weight=550,
        )
        sale_font, fitted_sale = _fit_single_line(
            sale_label,
            max_width=max_width,
            max_size=max_size,
            min_size=34,
            weight=850,
        )
        base_x = anchor_x
        sale_x = anchor_x
        if align == "right":
            base_x -= base_font.getlength(fitted_base)
            sale_x -= sale_font.getlength(fitted_sale)
        draw.text((base_x, y), fitted_base, font=base_font, fill=foreground)
        base_box = base_font.getbbox(fitted_base or "가")
        sale_y = y + (base_box[3] - base_box[1]) + 10
        draw.text((sale_x, sale_y), fitted_sale, font=sale_font, fill=foreground)
        return

    font, label = _fit_single_line(
        base_label or "",
        max_width=max_width,
        max_size=max_size,
        min_size=32,
        weight=800,
    )
    draw_x = anchor_x - font.getlength(label) if align == "right" else anchor_x
    draw.text((draw_x, y), label, font=font, fill=foreground)


def _render_berry(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    call_to_action: str | None,
) -> Image.Image:
    canvas = _gradient((255, 247, 244), (248, 199, 207))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.ellipse((760, -150, 1170, 260), fill=(255, 255, 255, 95))
    draw.ellipse((-170, 760, 280, 1210), fill=(205, 63, 98, 40))
    draw.rounded_rectangle((58, 55, 610, 1018), radius=42, fill=(255, 255, 255, 218))

    _draw_store_name(
        draw,
        store_name or "오늘의 메뉴",
        x=96,
        y=84,
        max_width=455,
        fill=(132, 46, 68, 255),
    )
    draw.line((96, 148, 220, 148), fill=(220, 77, 108, 255), width=6)

    headline_font, headline_lines = _fit_text(
        copy["headline"],
        max_width=455,
        max_lines=2,
        max_size=HEADLINE_MAX_FONT_SIZE,
        min_size=HEADLINE_MIN_FONT_SIZE,
        weight=800,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    next_y = _draw_lines(
        draw,
        headline_lines,
        x=96,
        y=188,
        font=headline_font,
        fill=(74, 34, 48, 255),
        spacing=14,
    )

    product_bottom = _draw_product_name(
        draw,
        product_name,
        x=96,
        y=next_y + 16,
        max_width=445,
        fill=(174, 49, 82, 255),
    )
    body_font, body_lines = _fit_text(
        copy["body"], max_width=445, max_lines=2, max_size=33, min_size=27, weight=430
    )
    body_y = _draw_lines(
        draw,
        body_lines,
        x=96,
        y=product_bottom + 16,
        font=body_font,
        fill=(89, 69, 75, 255),
        spacing=13,
    )

    if promotion:
        body_y += 20
        body_y += _draw_pill(
            draw,
            promotion,
            x=96,
            y=min(body_y, 760),
            max_font_size=28,
            background=(219, 67, 105, 255),
            foreground=(255, 255, 255, 255),
            max_width=470,
            min_font_size=18,
        )

    _draw_price_block(
        draw,
        base_price=base_price,
        sale_price=sale_price,
        promotion=promotion,
        anchor_x=96,
        y=880,
        max_width=455,
        foreground=(169, 39, 73, 255),
        max_size=56,
    )
    _draw_call_to_action(
        draw,
        call_to_action,
        x=96,
        y=836,
        max_width=455,
        fill=(132, 46, 68, 255),
    )

    _paste_product(
        canvas,
        product_image,
        (520, 205, 1065, 1020),
        scale_boost=1.06,
        portrait_scale=0.84,
    )
    return canvas.convert("RGB")


def _render_cream(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    call_to_action: str | None,
) -> Image.Image:
    canvas = _gradient((255, 253, 246), (239, 224, 199))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.ellipse((270, 320, 810, 860), fill=(255, 255, 255, 150))

    _draw_store_name(
        draw,
        store_name or "오늘의 추천",
        x=105,
        y=54,
        max_width=870,
        fill=(105, 77, 53, 255),
        align="center",
    )

    headline_font, headline_lines = _fit_text(
        copy["headline"],
        max_width=870,
        max_lines=2,
        max_size=HEADLINE_MAX_FONT_SIZE,
        min_size=HEADLINE_MIN_FONT_SIZE,
        weight=800,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    headline_bottom = _draw_lines(
        draw,
        headline_lines,
        x=105,
        y=116,
        font=headline_font,
        fill=(62, 51, 39, 255),
        spacing=12,
        align="center",
        area_width=870,
    )

    product_bottom = _draw_product_name(
        draw,
        product_name,
        x=105,
        y=headline_bottom + 10,
        max_width=870,
        fill=(128, 91, 58, 255),
        align="center",
        area_width=870,
    )

    product_top = max(315, round(product_bottom + 28))
    _paste_product(
        canvas,
        product_image,
        (165, product_top, 915, 835),
        shadow_opacity=65,
    )

    body_text = copy["body"].strip()
    has_footer_price = bool(base_price or sale_price)
    if body_text:
        draw.rounded_rectangle((92, 842, 988, 1022), radius=30, fill=(255, 255, 255, 225))
    elif call_to_action:
        draw.rounded_rectangle((92, 890, 988, 1022), radius=30, fill=(255, 255, 255, 225))
    elif has_footer_price:
        draw.rounded_rectangle((620, 882, 988, 1022), radius=30, fill=(255, 255, 255, 225))
    body_max_width = 500 if base_price or sale_price else 820
    if body_text:
        body_font, body_lines = _fit_text(
            body_text,
            max_width=body_max_width,
            max_lines=2,
            max_size=31,
            min_size=25,
            weight=430,
        )
        _draw_lines(
            draw,
            body_lines,
            x=126,
            y=870,
            font=body_font,
            fill=(82, 70, 58, 255),
            spacing=10,
        )
    _draw_call_to_action(
        draw,
        call_to_action,
        x=126,
        y=972 if body_text else 930,
        max_width=500,
        fill=(105, 77, 53, 255),
    )
    _draw_price_block(
        draw,
        base_price=base_price,
        sale_price=sale_price,
        promotion=promotion,
        anchor_x=950,
        y=886 if body_text else 916,
        max_width=270,
        foreground=(120, 78, 48, 255),
        align="right",
        max_size=40,
    )
    if promotion:
        _draw_pill(
            draw,
            promotion,
            x=92,
            y=720,
            max_font_size=27,
            background=(117, 85, 56, 235),
            foreground=(255, 255, 255, 255),
            max_width=600,
            min_font_size=18,
        )
    return canvas.convert("RGB")


def _render_night(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    call_to_action: str | None,
) -> Image.Image:
    canvas = _gradient((31, 21, 42), (91, 38, 63))
    draw = ImageDraw.Draw(canvas, "RGBA")
    gold = (235, 203, 137, 255)
    draw.rounded_rectangle((60, 62, 625, 1018), radius=40, fill=(16, 12, 26, 135), outline=(235, 203, 137, 65), width=2)

    _draw_store_name(
        draw,
        store_name or "대표 메뉴",
        x=100,
        y=88,
        max_width=455,
        fill=gold,
    )
    draw.line((100, 151, 245, 151), fill=gold, width=4)
    headline_font, headline_lines = _fit_text(
        copy["headline"],
        max_width=455,
        max_lines=2,
        max_size=HEADLINE_MAX_FONT_SIZE,
        min_size=HEADLINE_MIN_FONT_SIZE,
        weight=800,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    next_y = _draw_lines(
        draw,
        headline_lines,
        x=100,
        y=195,
        font=headline_font,
        fill=(255, 250, 242, 255),
        spacing=16,
    )
    product_bottom = _draw_product_name(
        draw,
        product_name,
        x=100,
        y=next_y + 16,
        max_width=450,
        fill=gold,
    )

    body_font, body_lines = _fit_text(
        copy["body"], max_width=450, max_lines=2, max_size=32, min_size=26, weight=430
    )
    body_y = _draw_lines(
        draw,
        body_lines,
        x=100,
        y=product_bottom + 16,
        font=body_font,
        fill=(231, 222, 227, 255),
        spacing=14,
    )
    if promotion:
        _draw_pill(
            draw,
            promotion,
            x=100,
            y=min(body_y + 24, 760),
            max_font_size=28,
            background=(235, 203, 137, 255),
            foreground=(50, 30, 45, 255),
            max_width=470,
            min_font_size=18,
        )
    _draw_price_block(
        draw,
        base_price=base_price,
        sale_price=sale_price,
        promotion=promotion,
        anchor_x=100,
        y=880,
        max_width=455,
        foreground=gold,
        max_size=54,
    )
    _draw_call_to_action(
        draw,
        call_to_action,
        x=100,
        y=836,
        max_width=455,
        fill=gold,
    )

    _paste_product(
        canvas,
        product_image,
        (540, 205, 1065, 1020),
        shadow_opacity=90,
        scale_boost=1.06,
    )
    return canvas.convert("RGB")


def _render_clean(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    call_to_action: str | None,
) -> Image.Image:
    canvas = _gradient((255, 255, 255), (234, 244, 244))
    draw = ImageDraw.Draw(canvas, "RGBA")
    teal = (20, 120, 119, 255)
    ink = (28, 49, 54, 255)
    draw.rectangle((0, 0, 18, 1080), fill=teal)
    draw.ellipse((560, 155, 1130, 725), fill=(213, 239, 237, 230))
    draw.ellipse((780, -150, 1170, 240), fill=(20, 120, 119, 18))

    _draw_store_name(
        draw,
        store_name or "오늘의 추천",
        x=72,
        y=62,
        max_width=445,
        fill=teal,
    )
    draw.line((72, 126, 190, 126), fill=teal, width=5)
    headline_font, headline_lines = _fit_text(
        copy["headline"],
        max_width=445,
        max_lines=2,
        max_size=HEADLINE_MAX_FONT_SIZE,
        min_size=HEADLINE_MIN_FONT_SIZE,
        weight=800,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    headline_bottom = _draw_lines(
        draw,
        headline_lines,
        x=72,
        y=166,
        font=headline_font,
        fill=ink,
        spacing=14,
    )
    product_bottom = _draw_product_name(
        draw,
        product_name,
        x=72,
        y=headline_bottom + 14,
        max_width=445,
        fill=teal,
    )
    body_font, body_lines = _fit_text(
        copy["body"], max_width=445, max_lines=2, max_size=32, min_size=26, weight=430
    )
    body_bottom = _draw_lines(
        draw,
        body_lines,
        x=72,
        y=product_bottom + 16,
        font=body_font,
        fill=(64, 80, 83, 255),
        spacing=13,
    )
    if promotion:
        _draw_pill(
            draw,
            promotion,
            x=72,
            y=min(body_bottom + 20, 760),
            max_font_size=27,
            background=teal,
            foreground=(255, 255, 255, 255),
            max_width=455,
            min_font_size=18,
        )
    _draw_price_block(
        draw,
        base_price=base_price,
        sale_price=sale_price,
        promotion=promotion,
        anchor_x=72,
        y=880,
        max_width=445,
        foreground=ink,
        max_size=54,
    )
    _draw_call_to_action(
        draw,
        call_to_action,
        x=72,
        y=836,
        max_width=445,
        fill=teal,
    )

    _paste_product(
        canvas,
        product_image,
        (475, 175, 1050, 1005),
        shadow_opacity=60,
        scale_boost=1.04,
        portrait_scale=0.88,
    )
    return canvas.convert("RGB")


def _render_natural(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    call_to_action: str | None,
) -> Image.Image:
    canvas = _gradient((245, 235, 207), (218, 202, 157))
    draw = ImageDraw.Draw(canvas, "RGBA")
    olive = (65, 83, 52, 255)
    terracotta = (174, 92, 58, 255)
    draw.rounded_rectangle((54, 46, 1026, 1034), radius=46, fill=(255, 252, 239, 218))
    draw.ellipse((-115, 735, 245, 1095), fill=(96, 113, 69, 42))
    draw.ellipse((850, -120, 1160, 190), fill=(174, 92, 58, 30))

    _draw_store_name(
        draw,
        store_name or "오늘의 베이커리",
        x=115,
        y=58,
        max_width=850,
        fill=olive,
        align="center",
    )
    headline_font, headline_lines = _fit_text(
        copy["headline"],
        max_width=850,
        max_lines=2,
        max_size=HEADLINE_MAX_FONT_SIZE,
        min_size=HEADLINE_MIN_FONT_SIZE,
        weight=800,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    headline_bottom = _draw_lines(
        draw,
        headline_lines,
        x=115,
        y=122,
        font=headline_font,
        fill=olive,
        spacing=12,
        align="center",
        area_width=850,
    )
    product_bottom = _draw_product_name(
        draw,
        product_name,
        x=115,
        y=headline_bottom + 10,
        max_width=850,
        fill=terracotta,
        align="center",
        area_width=850,
    )
    product_top = max(305, round(product_bottom + 24))
    _paste_product(
        canvas,
        product_image,
        (95, product_top, 985, 835),
        shadow_opacity=60,
        scale_boost=1.02,
    )

    footer_fill = (236, 229, 205, 246)
    body_text = copy["body"].strip()
    has_footer_price = bool(base_price or sale_price)
    if body_text:
        footer_box = (76, 842, 1004, 1010)
    elif call_to_action:
        footer_box = (76, 884, 1004, 1010)
    elif has_footer_price:
        footer_box = (642, 876, 1004, 1010)
    else:
        footer_box = None
    if footer_box:
        draw.rounded_rectangle(
            footer_box,
            radius=32,
            fill=footer_fill,
            outline=(96, 113, 69, 145),
            width=2,
        )
    if body_text:
        draw.line((108, 854, 972, 854), fill=olive, width=3)
    body_max_width = 510 if base_price or sale_price else 820
    if body_text:
        body_font, body_lines = _fit_text(
            body_text,
            max_width=body_max_width,
            max_lines=2,
            max_size=31,
            min_size=24,
            weight=430,
        )
        _draw_lines(
            draw,
            body_lines,
            x=108,
            y=872,
            font=body_font,
            fill=olive,
            spacing=10,
        )
    _draw_call_to_action(
        draw,
        call_to_action,
        x=108,
        y=966 if body_text else 924,
        max_width=510,
        fill=terracotta,
    )
    _draw_price_block(
        draw,
        base_price=base_price,
        sale_price=sale_price,
        promotion=promotion,
        anchor_x=960,
        y=886 if body_text else 912,
        max_width=285,
        foreground=terracotta,
        align="right",
        max_size=40,
    )
    if promotion:
        _draw_pill(
            draw,
            promotion,
            x=78,
            y=720,
            max_font_size=26,
            background=terracotta,
            foreground=(255, 255, 255, 255),
            max_width=620,
            min_font_size=18,
        )
    return canvas.convert("RGB")


def _render_promo(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    call_to_action: str | None,
) -> Image.Image:
    canvas = _gradient((255, 239, 149), (255, 203, 85))
    draw = ImageDraw.Draw(canvas, "RGBA")
    red = (221, 52, 54, 255)
    ink = (52, 38, 31, 255)
    draw.polygon(((0, 0), (690, 0), (520, 1080), (0, 1080)), fill=(255, 250, 228, 238))
    draw.ellipse((735, -105, 1125, 285), fill=(221, 52, 54, 220))
    draw.ellipse((800, 30, 1015, 245), outline=(255, 250, 228, 170), width=5)

    _draw_store_name(
        draw,
        store_name or "오늘의 혜택",
        x=70,
        y=60,
        max_width=450,
        fill=red,
    )
    draw.line((70, 126, 230, 126), fill=red, width=7)
    headline_font, headline_lines = _fit_text(
        copy["headline"],
        max_width=450,
        max_lines=2,
        max_size=HEADLINE_MAX_FONT_SIZE,
        min_size=HEADLINE_MIN_FONT_SIZE,
        weight=850,
        preferred_max_lines=2,
        balance_two_lines=True,
    )
    headline_bottom = _draw_lines(
        draw,
        headline_lines,
        x=70,
        y=165,
        font=headline_font,
        fill=ink,
        spacing=13,
    )
    product_bottom = _draw_product_name(
        draw,
        product_name,
        x=70,
        y=headline_bottom + 14,
        max_width=450,
        fill=red,
    )
    body_font, body_lines = _fit_text(
        copy["body"], max_width=450, max_lines=2, max_size=31, min_size=25, weight=430
    )
    body_bottom = _draw_lines(
        draw,
        body_lines,
        x=70,
        y=product_bottom + 16,
        font=body_font,
        fill=(79, 61, 50, 255),
        spacing=12,
    )
    if promotion:
        _draw_pill(
            draw,
            promotion,
            x=70,
            y=min(body_bottom + 22, 760),
            max_font_size=29,
            background=red,
            foreground=(255, 255, 255, 255),
            max_width=470,
            min_font_size=18,
            weight=750,
        )
    _draw_price_block(
        draw,
        base_price=base_price,
        sale_price=sale_price,
        promotion=promotion,
        anchor_x=70,
        y=872,
        max_width=450,
        foreground=red,
        max_size=58,
    )
    _draw_call_to_action(
        draw,
        call_to_action,
        x=70,
        y=828,
        max_width=450,
        fill=red,
    )

    _paste_product(
        canvas,
        product_image,
        (475, 195, 1055, 1025),
        shadow_opacity=70,
        scale_boost=1.06,
        portrait_scale=0.86,
    )
    return canvas.convert("RGB")


def generate_banner(
    product_image: Image.Image,
    copy: AdCopy,
    *,
    store_name: str | None,
    product_name: str,
    base_price: str | None = None,
    sale_price: str | None = None,
    promotion: str | None,
    call_to_action: str | None = None,
    template_id: str,
    price: str | None = None,
) -> Image.Image:
    """선택한 템플릿으로 1080×1080 SNS 배너를 만든다."""

    if template_id not in TEMPLATE_MAP:
        raise ValueError(f"지원하지 않는 배너 템플릿입니다: {template_id}")
    if not product_name.strip():
        raise ValueError("상품명은 비어 있을 수 없습니다.")
    if base_price and price and base_price != price:
        raise ValueError("기본 가격과 기존 가격 값이 서로 다릅니다.")
    resolved_base_price = base_price or price
    if sale_price and not resolved_base_price:
        raise ValueError("할인가를 표시하려면 기본 가격이 필요합니다.")

    renderers = {
        "berry": _render_berry,
        "cream": _render_cream,
        "night": _render_night,
        "clean": _render_clean,
        "natural": _render_natural,
        "promo": _render_promo,
    }
    return renderers[template_id](
        product_image,
        copy,
        store_name=store_name,
        product_name=product_name,
        base_price=resolved_base_price,
        sale_price=sale_price,
        promotion=promotion,
        call_to_action=call_to_action,
    )


def banner_to_png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
