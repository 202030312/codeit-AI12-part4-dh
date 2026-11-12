"""기능 3의 고정된 입출력 범위와 요청 형식을 정의한다."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FEATURE3_VERSION = "image_ad_v0.4.1"
FEATURE3_SCHEMA_VERSION = "1.0"
FEATURE3_OUTPUT_SIZE = (1080, 1080)
FEATURE3_OUTPUT_FORMAT = "PNG"
BACKGROUND_MOOD_OPTIONS = (
    "bright_clean",
    "warm_soft",
    "calm_premium",
    "vivid_cheerful",
)
COLOR_TONE_OPTIONS = (
    "white_cream",
    "beige_brown",
    "pink_lavender",
    "blue_mint",
    "dark_gold",
)
PRODUCT_POSITION_OPTIONS = ("left", "center", "right")
PRODUCT_SIZE_OPTIONS = ("small", "default", "large")
MARGIN_OPTIONS = ("narrow", "default", "wide")
MENU_CATEGORY_OPTIONS = (
    "bread",
    "hot_drink",
    "iced_drink",
    "dessert",
    "set_gift",
)
IMAGE_PURPOSE_OPTIONS = ("image_only", "copy_later")
TEXT_SPACE_OPTIONS = ("auto", "top", "bottom", "left", "right", "none")
TEMPLATE_RECOMMENDATIONS = {
    "bread": ("natural", "cream", "promo"),
    "hot_drink": ("cream", "night", "natural"),
    "iced_drink": ("clean", "berry", "promo"),
    "dessert": ("berry", "night", "cream"),
    "set_gift": ("night", "cream", "promo"),
}


def recommended_template_ids(menu_category: str) -> tuple[str, str, str]:
    """메뉴 카테고리에 맞는 우선 추천 템플릿 3종을 반환한다."""

    normalized = _choice(
        menu_category,
        field_name="메뉴 카테고리",
        choices=MENU_CATEGORY_OPTIONS,
    )
    return TEMPLATE_RECOMMENDATIONS[normalized]

REQUEST_FIELDS = (
    "background_mood",
    "color_tone",
    "product_position",
    "product_size",
    "margin",
    "include_shadow",
)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _required_text(value: str | None, field_name: str) -> str:
    normalized = _optional_text(value)
    if normalized is None:
        raise ValueError(f"{field_name}은(는) 필수 입력값입니다.")
    return normalized


def _choice(value: str, *, field_name: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(
            f"{field_name} 값은 다음 중 하나여야 합니다: {', '.join(choices)}"
        )
    return value


def build_image_request(
    *,
    background_mood: str,
    color_tone: str,
    product_position: str,
    product_size: str,
    margin: str,
    include_shadow: bool,
) -> dict[str, Any]:
    """기능 3에서 우선 지원하는 요청 사항만 고정 구조로 만든다."""

    if not isinstance(include_shadow, bool):
        raise ValueError("그림자 포함 여부는 참 또는 거짓이어야 합니다.")

    return {
        "background_mood": _choice(
            background_mood,
            field_name="배경 분위기",
            choices=BACKGROUND_MOOD_OPTIONS,
        ),
        "color_tone": _choice(
            color_tone,
            field_name="배경 색감",
            choices=COLOR_TONE_OPTIONS,
        ),
        "product_position": _choice(
            product_position,
            field_name="상품 위치",
            choices=PRODUCT_POSITION_OPTIONS,
        ),
        "product_size": _choice(
            product_size,
            field_name="상품 크기",
            choices=PRODUCT_SIZE_OPTIONS,
        ),
        "margin": _choice(
            margin,
            field_name="여백",
            choices=MARGIN_OPTIONS,
        ),
        "include_shadow": include_shadow,
    }


def _normalize_image_request(
    image_request: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if image_request is None:
        return None
    if set(image_request) != set(REQUEST_FIELDS):
        raise ValueError(
            "이미지 요청 사항에는 배경 분위기·색감, 상품 위치·크기, 여백, "
            "그림자 포함 여부만 사용할 수 있습니다."
        )
    return build_image_request(
        background_mood=image_request["background_mood"],
        color_tone=image_request["color_tone"],
        product_position=image_request["product_position"],
        product_size=image_request["product_size"],
        margin=image_request["margin"],
        include_shadow=image_request["include_shadow"],
    )


def build_feature3_request(
    *,
    image_filename: str | None,
    image_content_type: str | None,
    menu_category: str,
    image_purpose: str,
    text_space: str,
    image_request: Mapping[str, Any] | None = None,
    style_template_id: str | None = None,
) -> dict[str, Any]:
    """기능 3 요청을 고정된 JSON 호환 구조로 변환한다.

    상품 사진은 항상 필요하며, 이미지 요청 사항과 스타일 템플릿 중 하나 이상을
    입력해야 한다. 이 계약에는 상품 특징, 가격, 광고 문구 등 기능 3 범위 밖의
    정보가 들어가지 않는다.
    """

    normalized_request = _normalize_image_request(image_request)
    normalized_template_id = _optional_text(style_template_id)
    if normalized_request is None and normalized_template_id is None:
        raise ValueError("이미지 요청 사항 또는 스타일 템플릿을 하나 이상 입력해주세요.")

    if normalized_request is not None and normalized_template_id is not None:
        request_source = "direct_request_and_style_template"
    elif normalized_request is not None:
        request_source = "direct_request"
    else:
        request_source = "style_template"

    normalized_menu_category = _choice(
        menu_category,
        field_name="메뉴 카테고리",
        choices=MENU_CATEGORY_OPTIONS,
    )
    normalized_image_purpose = _choice(
        image_purpose,
        field_name="이미지 사용 목적",
        choices=IMAGE_PURPOSE_OPTIONS,
    )
    normalized_text_space = _choice(
        text_space,
        field_name="문구 여백",
        choices=TEXT_SPACE_OPTIONS,
    )
    if normalized_image_purpose == "image_only" and normalized_text_space not in {
        "auto",
        "none",
    }:
        raise ValueError("이미지 단독 사용에서는 문구 여백을 지정할 수 없습니다.")
    if normalized_image_purpose == "copy_later" and normalized_text_space == "none":
        raise ValueError("나중에 문구를 추가하려면 문구 여백이 필요합니다.")

    return {
        "schema_version": FEATURE3_SCHEMA_VERSION,
        "feature_version": FEATURE3_VERSION,
        "request_source": request_source,
        "menu_category": normalized_menu_category,
        "image_purpose": normalized_image_purpose,
        "text_space": normalized_text_space,
        "image": {
            "file_name": _required_text(image_filename, "상품 사진"),
            "content_type": _optional_text(image_content_type),
        },
        "image_request": normalized_request,
        "style_template_id": normalized_template_id,
        "output": {
            "width": FEATURE3_OUTPUT_SIZE[0],
            "height": FEATURE3_OUTPUT_SIZE[1],
            "format": FEATURE3_OUTPUT_FORMAT,
            "include_text": False,
        },
    }
