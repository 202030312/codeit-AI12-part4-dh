"""기능 3의 문구 없는 이미지에 검증된 광고 문구와 판매 정보를 배치한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.features.ad_copy.generator import AdCopy
from src.features.ad_copy.promotion import BADGE_ONLY, promotion_badge_text


COMBINED_BANNER_VERSION = "combined_ad_v0.2.3"
FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "NotoSansKR-Variable.ttf"


PALETTES: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "berry": {
        "primary": (174, 49, 82, 255),
        "headline": (74, 34, 48, 255),
        "body": (89, 69, 75, 255),
        "muted": (132, 46, 68, 255),
        "badge_bg": (187, 43, 78, 255),
        "badge_fg": (255, 255, 255, 255),
    },
    "cream": {
        "primary": (120, 78, 48, 255),
        "headline": (76, 57, 42, 255),
        "body": (82, 70, 58, 255),
        "muted": (105, 77, 53, 255),
        "badge_bg": (117, 85, 56, 235),
        "badge_fg": (255, 255, 255, 255),
    },
    "night": {
        "primary": (235, 203, 137, 255),
        "headline": (255, 250, 242, 255),
        "body": (231, 222, 227, 255),
        "muted": (235, 203, 137, 255),
        "badge_bg": (235, 203, 137, 255),
        "badge_fg": (50, 30, 45, 255),
    },
    "clean": {
        "primary": (20, 120, 119, 255),
        "headline": (28, 49, 54, 255),
        "body": (64, 80, 83, 255),
        "muted": (11, 94, 94, 255),
        "badge_bg": (20, 120, 119, 255),
        "badge_fg": (255, 255, 255, 255),
    },
    "natural": {
        "primary": (174, 92, 58, 255),
        "headline": (75, 88, 58, 255),
        "body": (75, 88, 58, 255),
        "muted": (75, 88, 58, 255),
        "badge_bg": (174, 92, 58, 255),
        "badge_fg": (255, 255, 255, 255),
    },
    "promo": {
        "primary": (221, 52, 54, 255),
        "headline": (52, 38, 31, 255),
        "body": (79, 61, 50, 255),
        "muted": (184, 37, 40, 255),
        "badge_bg": (221, 52, 54, 255),
        "badge_fg": (255, 255, 255, 255),
    },
}


@dataclass(frozen=True)
class CombinedBannerResult:
    image: Image.Image
    text_boxes: tuple[tuple[int, int, int, int], ...]
    promotion_badge_text: str | None
    layout: str
    contrast_checks: tuple[dict[str, Any], ...]


def _font(size: int, weight: int = 500) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONT_PATH), size=size)
    if hasattr(font, "set_variation_by_axes"):
        try:
            font.set_variation_by_axes([float(weight)])
        except (OSError, ValueError):
            pass
    return font


def _wrap(
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
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
        current = word
    if current:
        lines.append(current)
    if 1 < len(lines) <= 3:
        balanced = _balanced_wrap(normalized, font=font, max_width=max_width, line_count=len(lines))
        if balanced is not None:
            return balanced
    return lines


def _balanced_wrap(
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
    line_count: int,
) -> list[str] | None:
    """단어를 나누지 않고 각 줄의 길이 차이와 고아 단어를 최소화한다."""

    words = text.split()
    if line_count < 2 or len(words) < line_count:
        return None
    candidates: list[tuple[float, float, list[str]]] = []

    def visit(start: int, remaining: int, built: list[str]) -> None:
        if remaining == 1:
            line = " ".join(words[start:])
            if line and font.getlength(line) <= max_width:
                lines = [*built, line]
                widths = [float(font.getlength(item)) for item in lines]
                average = sum(widths) / len(widths)
                raggedness = sum((value - average) ** 2 for value in widths)
                orphan_penalty = sum(
                    max_width * 1.8
                    for item, value in zip(lines[1:], widths[1:])
                    if len(item) <= 3 or value < max_width * 0.25
                )
                candidates.append((raggedness + orphan_penalty, max(widths), lines))
            return
        max_split = len(words) - remaining + 1
        for split in range(start + 1, max_split + 1):
            line = " ".join(words[start:split])
            if font.getlength(line) > max_width:
                break
            visit(split, remaining - 1, [*built, line])

    visit(0, line_count, [])
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _relative_luminance(color: tuple[int, int, int]) -> float:
    values = []
    for channel in color:
        value = channel / 255
        values.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def _contrast_ratio(
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
) -> float:
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _box_contrast_ratio(
    image: Image.Image,
    box: tuple[int, int, int, int],
    foreground: tuple[int, int, int, int],
) -> float:
    crop = image.convert("RGB").crop(box)
    if crop.width <= 0 or crop.height <= 0:
        return 1.0
    ratios: list[float] = []
    step_x = max(1, crop.width // 24)
    step_y = max(1, crop.height // 12)
    pixels = crop.load()
    assert pixels is not None
    for y in range(0, crop.height, step_y):
        for x in range(0, crop.width, step_x):
            ratios.append(_contrast_ratio(foreground[:3], pixels[x, y]))
    ratios.sort()
    # 국소 장식 한 픽셀보다 문구 영역 전체의 보수적인 하위 10% 대비를 사용한다.
    return ratios[max(0, round((len(ratios) - 1) * 0.10))]


def _fit_lines(
    text: str,
    *,
    max_width: int,
    max_lines: int,
    max_size: int,
    min_size: int,
    weight: int,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    for size in range(max_size, min_size - 1, -2):
        font = _font(size, weight)
        lines = _wrap(text, font=font, max_width=max_width)
        if lines and len(lines) <= max_lines:
            return font, lines
    raise ValueError(f"문구를 안전 영역에 원문 그대로 배치할 수 없습니다: {text}")


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    spacing: int,
) -> tuple[int, tuple[int, int, int, int]]:
    start_y = y
    max_right = x
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        box = draw.textbbox((x, y), line or "가", font=font)
        max_right = max(max_right, box[2])
        y = box[3] + spacing
    return y, (x, start_y, max_right, y - spacing)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    max_width: int,
    max_lines: int,
    max_size: int,
    min_size: int,
    weight: int,
    fill: tuple[int, int, int, int],
    spacing: int = 7,
) -> tuple[int, tuple[int, int, int, int]]:
    font, lines = _fit_lines(
        text,
        max_width=max_width,
        max_lines=max_lines,
        max_size=max_size,
        min_size=min_size,
        weight=weight,
    )
    return _draw_lines(draw, lines, x=x, y=y, font=font, fill=fill, spacing=spacing)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    max_width: int,
    background: tuple[int, int, int, int],
    foreground: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int]]:
    font, lines = _fit_lines(
        text,
        max_width=max_width - 44,
        max_lines=2,
        max_size=25,
        min_size=17,
        weight=720,
    )
    line_height = font.getbbox("가")[3] - font.getbbox("가")[1]
    height = line_height * len(lines) + 8 * (len(lines) - 1) + 36
    width = min(max_width, round(max(font.getlength(line) for line in lines)) + 44)
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=min(24, height // 2),
        fill=background,
    )
    _draw_lines(
        draw,
        lines,
        x=x + 22,
        y=y + 10,
        font=font,
        fill=foreground,
        spacing=8,
    )
    return y + height, (x, y, x + width, y + height)


def _draw_price(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    max_width: int,
    base_price: str | None,
    sale_price: str | None,
    color: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int]] | None:
    if not base_price and not sale_price:
        return None
    start_y = y
    right = x
    if sale_price:
        y, base_box = _draw_text(
            draw,
            f"기본가 {base_price}",
            x=x,
            y=y,
            max_width=max_width,
            max_lines=1,
            max_size=23,
            min_size=17,
            weight=520,
            fill=color,
            spacing=5,
        )
        y, sale_box = _draw_text(
            draw,
            sale_price,
            x=x,
            y=y,
            max_width=max_width,
            max_lines=1,
            max_size=38,
            min_size=27,
            weight=850,
            fill=color,
        )
        right = max(base_box[2], sale_box[2])
    else:
        y, box = _draw_text(
            draw,
            base_price,
            x=x,
            y=y,
            max_width=max_width,
            max_lines=1,
            max_size=45,
            min_size=32,
            weight=850,
            fill=color,
        )
        right = box[2]
    return y, (x, start_y, right, y)


def _inside(box: tuple[int, int, int, int], area: tuple[int, int, int, int]) -> bool:
    return area[0] <= box[0] and area[1] <= box[1] and box[2] <= area[2] and box[3] <= area[3]


def render_combined_banner(
    base_image: Image.Image,
    *,
    template_id: str,
    text_safe_area: tuple[int, int, int, int],
    copy: AdCopy,
    store_name: str | None,
    product_name: str,
    base_price: str | None,
    sale_price: str | None,
    promotion: str | None,
    promotion_copy_mode: str = BADGE_ONLY,
    call_to_action: str | None = None,
) -> CombinedBannerResult:
    """빈 카드 없이 안전 영역에 문구·프로모션·가격을 직접 배치한다."""

    if template_id not in PALETTES:
        raise ValueError(f"지원하지 않는 결합 템플릿입니다: {template_id}")
    if base_image.size != (1080, 1080):
        raise ValueError("결합 배너의 기반 이미지는 1080×1080px이어야 합니다.")

    background = base_image.convert("RGB")
    canvas = base_image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    palette = PALETTES[template_id]
    left, top, right, bottom = text_safe_area
    width = right - left
    height = bottom - top
    horizontal = width > height * 1.65
    layout = "horizontal" if horizontal else "vertical"
    boxes: list[tuple[int, int, int, int]] = []
    contrast_checks: list[dict[str, Any]] = []

    def record_box(
        role: str,
        box: tuple[int, int, int, int],
        foreground: tuple[int, int, int, int],
        threshold: float,
        *,
        solid_background: tuple[int, int, int, int] | None = None,
    ) -> None:
        boxes.append(box)
        ratio = (
            _contrast_ratio(foreground[:3], solid_background[:3])
            if solid_background is not None
            else _box_contrast_ratio(background, box, foreground)
        )
        contrast_checks.append({
            "role": role,
            "box": box,
            "ratio": round(ratio, 2),
            "threshold": threshold,
            "passed": ratio >= threshold,
        })
    badge_text = promotion_badge_text(
        promotion,
        headline=copy["headline"],
        promotion_copy_mode=promotion_copy_mode,
    )

    if horizontal:
        left_inset = 190 if template_id == "natural" and top >= 700 else 22
        right_inset, pad_y = 22, 15
        content_left = left + left_inset
        content_top = top + pad_y
        inner_width = width - left_inset - right_inset
        left_width = round(inner_width * (0.52 if badge_text else 0.67))
        right_x = content_left + left_width + 24
        right_width = right - right_inset - right_x
        y = content_top
        if store_name:
            y, box = _draw_text(
                draw, store_name, x=content_left, y=y, max_width=left_width,
                max_lines=1, max_size=25, min_size=20, weight=700,
                fill=palette["muted"], spacing=4,
            )
            record_box("store_name", box, palette["muted"], 4.5)
        y, box = _draw_text(
            draw, copy["headline"], x=content_left, y=y + 3, max_width=left_width,
            max_lines=1, max_size=34, min_size=18, weight=790,
            fill=palette["headline"], spacing=5,
        )
        record_box("headline", box, palette["headline"], 3.0)
        y, box = _draw_text(
            draw, product_name, x=content_left, y=y + 3, max_width=left_width,
            max_lines=1, max_size=49, min_size=37, weight=850,
            fill=palette["primary"], spacing=4,
        )
        record_box("product_name", box, palette["primary"], 3.0)
        if copy.get("body"):
            y, box = _draw_text(
                draw, copy["body"], x=content_left, y=y + 3, max_width=left_width,
                max_lines=2, max_size=25, min_size=20, weight=480,
                fill=palette["body"], spacing=4,
            )
            record_box("body", box, palette["body"], 4.5)

        right_y = content_top
        if badge_text:
            right_y, box = _draw_badge(
                draw, badge_text, x=right_x, y=right_y, max_width=right_width,
                background=palette["badge_bg"], foreground=palette["badge_fg"],
            )
            record_box(
                "promotion_badge", box, palette["badge_fg"], 4.5,
                solid_background=palette["badge_bg"],
            )
        price_y = max(
            right_y + 10,
            bottom - (108 if sale_price else 72),
        )
        price_x = content_left if template_id == "promo" else right_x
        price_width = left_width if template_id == "promo" else right_width
        price_result = _draw_price(
            draw, x=price_x, y=price_y, max_width=price_width,
            base_price=base_price, sale_price=sale_price, color=palette["primary"],
        )
        if price_result:
            _, box = price_result
            record_box("price", box, palette["primary"], 3.0)
        if call_to_action:
            _, box = _draw_text(
                draw, f"→ {call_to_action}", x=content_left, y=bottom - 31,
                max_width=left_width, max_lines=1, max_size=21, min_size=17,
                weight=650, fill=palette["muted"],
            )
            record_box("call_to_action", box, palette["muted"], 4.5)
    else:
        pad_x, pad_y = 24, 24
        x = left + pad_x
        y = top + pad_y
        inner_width = width - 2 * pad_x
        if store_name:
            y, box = _draw_text(
                draw, store_name, x=x, y=y, max_width=inner_width,
                max_lines=1, max_size=30, min_size=23, weight=700,
                fill=palette["muted"], spacing=7,
            )
            record_box("store_name", box, palette["muted"], 4.5)
        y, box = _draw_text(
            draw, copy["headline"], x=x, y=y + 4, max_width=inner_width,
            max_lines=1, max_size=43, min_size=18, weight=800,
            fill=palette["headline"], spacing=9,
        )
        record_box("headline", box, palette["headline"], 3.0)
        y, box = _draw_text(
            draw, product_name, x=x, y=y + 5, max_width=inner_width,
            max_lines=2, max_size=60, min_size=45, weight=860,
            fill=palette["primary"], spacing=8,
        )
        record_box("product_name", box, palette["primary"], 3.0)
        if copy.get("body"):
            y, box = _draw_text(
                draw, copy["body"], x=x, y=y + 10, max_width=inner_width,
                max_lines=3, max_size=30, min_size=23, weight=480,
                fill=palette["body"], spacing=7,
            )
            record_box("body", box, palette["body"], 4.5)
        if badge_text:
            y, box = _draw_badge(
                draw, badge_text, x=x, y=y + 16, max_width=inner_width,
                background=palette["badge_bg"], foreground=palette["badge_fg"],
            )
            record_box(
                "promotion_badge", box, palette["badge_fg"], 4.5,
                solid_background=palette["badge_bg"],
            )
        price_y = bottom - (105 if sale_price else 64) - pad_y
        if template_id == "berry" and right <= 500:
            price_y = min(price_y, 700)
        price_result = _draw_price(
            draw, x=x, y=price_y, max_width=inner_width,
            base_price=base_price, sale_price=sale_price, color=palette["primary"],
        )
        if price_result:
            _, box = price_result
            record_box("price", box, palette["primary"], 3.0)
        if call_to_action:
            _, box = _draw_text(
                draw, f"→ {call_to_action}", x=x, y=price_y - 38,
                max_width=inner_width, max_lines=1, max_size=22, min_size=17,
                weight=650, fill=palette["muted"],
            )
            record_box("call_to_action", box, palette["muted"], 4.5)

    if not all(_inside(box, text_safe_area) for box in boxes):
        raise ValueError("결합 배너의 문구가 안전 영역을 벗어났습니다.")
    return CombinedBannerResult(
        image=canvas.convert("RGB"),
        text_boxes=tuple(boxes),
        promotion_badge_text=badge_text,
        layout=layout,
        contrast_checks=tuple(contrast_checks),
    )
