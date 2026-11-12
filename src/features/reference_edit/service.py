"""레퍼런스 이미지와 자연어 요청으로 새 광고 이미지를 생성한다."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


REFERENCE_IMAGE_EDIT_VERSION = "reference_image_edit_v0.3.2-ko"
REFERENCE_IMAGE_EDIT_MODEL = "gpt-image-2.5-sunburst"
REFERENCE_IMAGE_EDIT_SIZE = "1024x1024"
REFERENCE_IMAGE_EDIT_QUALITY = "medium"
MAX_USER_REQUEST_CHARS = 4_000
MAX_INPUT_IMAGE_BYTES = 50 * 1024 * 1024
SUPPORTED_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ReferenceImageEditError(RuntimeError):
    """레퍼런스 이미지 편집을 완료하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class ReferenceImageEditResult:
    """생성 결과와 재현에 필요한 실행 정보를 보관한다."""

    output_path: Path
    prompt: str
    prompt_version: str
    model: str
    size: str
    quality: str


REFERENCE_IMAGE_EDIT_PROMPT_TEMPLATE = """편집 유형: 정밀 객체 편집
결과물 유형: 레퍼런스 기반 카페 광고 이미지
입력 이미지: 이미지 1은 편집 대상인 동시에 시각적 레퍼런스다.

사용자 요청사항(내용을 정확히 반영할 것):
{user_request}

요청 유형과 변경 권한:
- 사용자가 한두 요소의 제한적인 수정과 나머지 유지·그대로 보존을 요청하면 국소 편집으로 처리한다.
- 사용자가 광고, 캠페인, 신메뉴 비주얼, 광고 키 비주얼 또는 새로운 광고 이미지 제작을 요청하면 광고 장면 재구성으로 처리한다.
- 사용자가 배경, 조명, 색상, 소품 또는 광고 레이아웃의 구체적인 선택을 맡기는 것도 해당 요소를 변경해도 된다는 명시적인 허용으로 해석한다.
- 사용자가 명시한 변경 대상과 보존 대상은 항상 가장 높은 우선순위로 적용한다.

광고 의도 확인:
- 사용자 요청에서 광고 목적, 전달하려는 핵심 감정, 강조할 상품 특징을 확인한다.
- 세 항목 가운데 사용자 요청에 명시되지 않은 내용은 새로운 광고 사실이나 효능으로 임의로 만들지 않는다.

시각 요소 선택 원칙:
- 배경, 조명과 소품은 확인한 광고 목적, 핵심 감정 또는 강조할 상품 특징에 기여할 때만 사용한다.
- 각 시각 요소는 상품을 돋보이게 하는 분명한 역할이 있어야 하며, 관련 없는 장식은 추가하지 않거나 단순화한다.
- 분위기 표현보다 상품의 식별성, 먹음직스러운 질감과 요청한 광고 메시지의 전달을 우선한다.

입력 이미지 확인 항목:
최종 이미지를 만들기 전에 이미지 1을 다음 항목으로 확인하고, 확인 결과는 이미지 편집에만 사용한다.
- 주요 피사체: 상품 종류, 형태, 개수, 재질과 질감, 용기 또는 접시
- 구도: 카메라 시점, 화면 비율, 원근, 크롭, 피사체의 위치와 크기
- 색상과 조명: 주요 색상, 광원의 방향과 세기, 명암, 반사와 그림자
- 배경과 소품: 배경의 형태, 주변 사물의 종류와 공간 배치
- 분위기와 후처리: 계절, 날씨, 감정, 대비, 채도, 심도와 광고 사진의 마감
- 카피 공간: 비어 있는 영역의 위치와 크기, 시각적으로 복잡한 영역

보존 범위와 재구성 범위:
- 상품 핵심: 사용자가 변경을 요청하지 않았다면 상품 종류, 개수, 핵심 형태와 구조, 용기 또는 접시, 식별 가능한 특징을 보존한다.
- 광고 장면: 광고 장면 재구성에서는 배경, 조명, 색감, 소품, 날씨·계절 표현, 주변 사물, 상품의 화면 내 위치와 크기, 카피 공간과 전체 레이아웃을 새롭게 구성할 수 있다.
- 광고 장면 재구성에서는 사용자가 장면 요소의 구체적인 종류를 말하지 않았다는 이유만으로 기존 배경과 주변 장면을 그대로 유지하지 않는다.
- 국소 편집에서는 사용자가 지정한 변경과 이를 자연스럽게 완성하는 데 필요한 연관 부분만 바꾸고 나머지는 유지한다.

충돌 시 우선순위:
1. 사용자가 명시한 변경과 보존 조건
2. 변경 요청이 없는 상품 핵심의 식별성
3. 광고 목적, 핵심 감정과 강조할 상품 특징
4. 요청한 카피 공간과 광고 구도
5. 배경, 조명과 소품의 관련성
6. 위 조건과 충돌하지 않는 나머지 원본 요소의 보존

편집 실행 규칙:
1. 이미지 1을 기본 이미지로 사용하고, 사용자 요청사항에 명시된 변경을 빠짐없이 반영한다.
2. 국소 편집에서는 사용자 요청을 충족하는 데 필요한 부분만 변경하고 보존 범위의 원본 요소를 최대한 유지한다.
3. 광고 장면 재구성에서는 상품 핵심을 시각적 기준으로 사용하되, 기존 배경을 단순 보정하거나 그대로 복제하지 않는다. 광고 목적과 핵심 감정에 맞는 새로운 배경, 조명, 필요한 최소한의 소품과 광고 레이아웃으로 완성된 키 비주얼을 만든다.
4. 광고 장면 재구성 결과는 원본 사진과 시각적으로 구분되어야 한다. 다만 상품이 여전히 주인공으로 분명하게 식별되고 카메라 시점과 원근이 자연스럽게 이어져야 한다.
5. 광고 구도나 카피 공간을 위해 상품의 화면 내 위치와 크기를 조절할 수 있다. 정확한 원래 좌표보다 상품 식별성, 광고 여백과 시각적 균형을 우선한다.
6. 음식이나 음료를 교체할 때에는 결과가 물리적으로 자연스럽고 시각적으로 분명하게 보이도록 한다. 요청한 교체 상품과 기존 용기 또는 접시가 어울린다면 기존 용기 또는 접시를 유지한다.
7. 카피 공간을 요청한 경우 지정된 영역을 충분히 크고 단순하게 비운다. 그 영역에 상품, 소품, 강한 하이라이트 또는 복잡한 무늬를 배치하지 않는다.
8. 결과는 사실적이고 상업 광고에 사용할 수 있는 완성도를 갖춰야 한다. 조명, 원근, 재질, 반사와 그림자가 서로 일관되게 보이도록 한다.
9. 사용자가 명시적으로 요청하지 않았다면 글자, 문자, 로고, 브랜드명, 가격, 라벨, 워터마크, 테두리 또는 사람을 추가하지 않는다.
10. 분석 결과, 작업 계획, 설명 문구를 이미지 안에 표시하지 않는다.
11. 콜라주, 전후 비교, 분할 화면 또는 여러 시안을 만들지 않는다. 하나의 일관된 편집 이미지 한 장만 반환한다.
"""


def normalize_user_request(user_request: str) -> str:
    """사용자 요청을 검증하고 앞뒤 공백만 제거한다."""

    if not isinstance(user_request, str):
        raise ValueError("이미지 요청사항은 문자열이어야 합니다.")

    normalized = user_request.strip()
    if not normalized:
        raise ValueError("이미지 요청사항을 입력해주세요.")
    if len(normalized) > MAX_USER_REQUEST_CHARS:
        raise ValueError(
            f"이미지 요청사항은 {MAX_USER_REQUEST_CHARS:,}자 이하로 입력해주세요."
        )
    return normalized


def build_reference_image_edit_prompt(user_request: str) -> str:
    """자연어 요청을 불변 조건이 포함된 공통 내부 프롬프트로 감싼다."""

    normalized = normalize_user_request(user_request)
    return REFERENCE_IMAGE_EDIT_PROMPT_TEMPLATE.format(user_request=normalized)


def validate_reference_image(image_path: str | Path) -> Path:
    """Image API에 전달할 로컬 입력 이미지의 기본 조건을 검사한다."""

    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"레퍼런스 이미지 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"지원하지 않는 이미지 형식입니다. 허용 형식: {allowed}")
    if path.stat().st_size > MAX_INPUT_IMAGE_BYTES:
        raise ValueError("레퍼런스 이미지는 50MB 이하여야 합니다.")
    return path


def _create_openai_client(
    *,
    api_key: str | None = None,
    timeout_seconds: float = 300.0,
) -> Any:
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise ReferenceImageEditError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ReferenceImageEditError(
            "OpenAI 패키지가 설치되지 않았습니다. requirements.txt를 설치해주세요."
        ) from exc

    return OpenAI(api_key=key, timeout=timeout_seconds, max_retries=1)


def edit_reference_image(
    *,
    image_path: str | Path,
    user_request: str,
    output_path: str | Path,
    client: Any | None = None,
    api_key: str | None = None,
    model: str = REFERENCE_IMAGE_EDIT_MODEL,
    size: str = REFERENCE_IMAGE_EDIT_SIZE,
    quality: str = REFERENCE_IMAGE_EDIT_QUALITY,
    overwrite: bool = False,
) -> ReferenceImageEditResult:
    """레퍼런스 이미지를 편집하고 PNG 결과를 로컬 파일로 저장한다."""

    source_path = validate_reference_image(image_path)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"출력 파일이 이미 존재합니다: {destination}. 덮어쓰려면 overwrite=True를 사용하세요."
        )
    if destination.suffix.lower() != ".png":
        raise ValueError("출력 파일 확장자는 .png여야 합니다.")

    prompt = build_reference_image_edit_prompt(user_request)
    api_client = client or _create_openai_client(api_key=api_key)

    try:
        with source_path.open("rb") as image_file:
            response = api_client.images.edit(
                model=model,
                image=image_file,
                prompt=prompt,
                size=size,
                quality=quality,
                background="opaque",
                output_format="png",
                n=1,
            )
    except Exception as exc:
        raise ReferenceImageEditError(
            f"레퍼런스 이미지 편집 API 호출에 실패했습니다: {exc}"
        ) from exc

    data = getattr(response, "data", None) or []
    encoded = getattr(data[0], "b64_json", None) if data else None
    if not encoded:
        raise ReferenceImageEditError("Image API가 이미지 데이터를 반환하지 않았습니다.")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ReferenceImageEditError(
            "Image API가 유효한 base64 이미지 데이터를 반환하지 않았습니다."
        ) from exc

    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ReferenceImageEditError("Image API 결과가 PNG 형식이 아닙니다.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(image_bytes)

    return ReferenceImageEditResult(
        output_path=destination,
        prompt=prompt,
        prompt_version=REFERENCE_IMAGE_EDIT_VERSION,
        model=model,
        size=size,
        quality=quality,
    )
