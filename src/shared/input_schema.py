"""입력 폼 데이터를 광고 생성 요청 형식으로 변환한다."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.2"


def _optional_text(value: str | None) -> str | None:
    """선택 입력값의 앞뒤 공백을 제거하고 빈 값은 ``None``으로 바꾼다."""
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _required_text(value: str | None, field_name: str) -> str:
    """필수 문자열을 정리하고 비어 있으면 명확한 오류를 발생시킨다."""
    normalized = _optional_text(value)
    if normalized is None:
        raise ValueError(f"{field_name}은(는) 필수 입력값입니다.")
    return normalized


def build_ad_input(
    *,
    image_filename: str | None,
    image_content_type: str | None,
    store_name: str | None,
    product_name: str | None,
    base_price: str | None = None,
    sale_price: str | None = None,
    features: str | None,
    promotion: str | None,
    tone: str | None,
    copy_request: str | None = None,
    call_to_action: str | None = None,
    price: str | None = None,
) -> dict[str, Any]:
    """폼 입력값을 항상 동일한 키를 갖는 JSON 호환 딕셔너리로 변환한다.

    ``price``는 1.0 호출부와의 호환을 위한 별칭이다. 새 코드에서는 가격의
    의미가 모호해지지 않도록 ``base_price``와 ``sale_price``를 사용한다.
    """
    normalized_base_price = _optional_text(base_price)
    legacy_price = _optional_text(price)
    if normalized_base_price and legacy_price and normalized_base_price != legacy_price:
        raise ValueError("기본 가격과 기존 가격 값이 서로 다릅니다.")
    normalized_base_price = normalized_base_price or legacy_price
    normalized_sale_price = _optional_text(sale_price)
    if normalized_sale_price and not normalized_base_price:
        raise ValueError("할인가를 입력하려면 기본 가격도 입력해주세요.")

    return {
        "schema_version": SCHEMA_VERSION,
        "image": {
            "file_name": _required_text(image_filename, "상품 사진"),
            "content_type": _optional_text(image_content_type),
        },
        "store": {
            "name": _optional_text(store_name),
        },
        "product": {
            "name": _required_text(product_name, "상품명"),
            "base_price": normalized_base_price,
            "sale_price": normalized_sale_price,
            "features": _required_text(features, "상품 특징"),
        },
        "campaign": {
            "promotion": _optional_text(promotion),
            "tone": _required_text(tone, "광고 분위기"),
            "copy_request": _optional_text(copy_request),
            "call_to_action": _optional_text(call_to_action),
        },
    }
