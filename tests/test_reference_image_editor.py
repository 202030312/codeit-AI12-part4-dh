from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.features.reference_edit.service import (
    REFERENCE_IMAGE_EDIT_MODEL,
    REFERENCE_IMAGE_EDIT_VERSION,
    ReferenceImageEditError,
    build_reference_image_edit_prompt,
    edit_reference_image,
    validate_reference_image,
)


class FakeImages:
    def __init__(self, png_bytes: bytes) -> None:
        self.png_bytes = png_bytes
        self.calls: list[dict[str, object]] = []

    def edit(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(self.png_bytes).decode("ascii")
                )
            ]
        )


class FakeClient:
    def __init__(self, png_bytes: bytes) -> None:
        self.images = FakeImages(png_bytes)


def make_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "pink").save(buffer, format="PNG")
    return buffer.getvalue()


def make_source_image(path: Path) -> None:
    Image.new("RGB", (8, 8), "brown").save(path, format="PNG")


def test_build_prompt_wraps_user_request_and_preservation_rules() -> None:
    request = "딸기 라떼로 바꿔주고 여름 분위기로 만들어줘"

    prompt = build_reference_image_edit_prompt(f"  {request}  ")

    assert request in prompt
    assert "이미지 1은 편집 대상인 동시에 시각적 레퍼런스" in prompt
    assert "요청 유형과 변경 권한" in prompt
    assert "국소 편집" in prompt
    assert "광고 장면 재구성" in prompt
    assert "해당 요소를 변경해도 된다는 명시적인 허용" in prompt
    assert "광고 목적, 전달하려는 핵심 감정, 강조할 상품 특징" in prompt
    assert "새로운 광고 사실이나 효능" in prompt
    assert "광고 목적, 핵심 감정 또는 강조할 상품 특징에 기여할 때만" in prompt
    assert "관련 없는 장식" in prompt
    assert "먹음직스러운 질감" in prompt
    assert "입력 이미지 확인 항목" in prompt
    assert "주요 피사체" in prompt
    assert "색상과 조명" in prompt
    assert "배경과 소품" in prompt
    assert "분위기와 후처리" in prompt
    assert "카피 공간" in prompt
    assert "보존 범위와 재구성 범위" in prompt
    assert "상품 핵심" in prompt
    assert "기존 배경과 주변 장면을 그대로 유지하지 않는다" in prompt
    assert "충돌 시 우선순위" in prompt
    assert "원본 사진과 시각적으로 구분" in prompt
    assert "정확한 원래 좌표보다 상품 식별성" in prompt
    assert "글자, 문자, 로고" in prompt
    assert REFERENCE_IMAGE_EDIT_VERSION == "reference_image_edit_v0.3.2-ko"


@pytest.mark.parametrize("user_request", ["", "   ", "\n\t"])
def test_build_prompt_rejects_blank_request(user_request: str) -> None:
    with pytest.raises(ValueError, match="요청사항"):
        build_reference_image_edit_prompt(user_request)


def test_validate_reference_image_rejects_unsupported_format(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.gif"
    image_path.write_bytes(b"GIF89a")

    with pytest.raises(ValueError, match="지원하지 않는 이미지 형식"):
        validate_reference_image(image_path)


def test_edit_reference_image_calls_api_and_writes_png(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    make_source_image(source)
    fake_client = FakeClient(make_png_bytes())

    result = edit_reference_image(
        image_path=source,
        user_request="딸기 라떼로 바꿔줘",
        output_path=destination,
        client=fake_client,
    )

    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.output_path == destination
    assert result.model == REFERENCE_IMAGE_EDIT_MODEL
    assert result.prompt_version == REFERENCE_IMAGE_EDIT_VERSION

    assert len(fake_client.images.calls) == 1
    call = fake_client.images.calls[0]
    assert call["model"] == "gpt-image-2.5-sunburst"
    assert call["size"] == "1024x1024"
    assert call["quality"] == "medium"
    assert call["background"] == "opaque"
    assert call["output_format"] == "png"
    assert call["n"] == 1
    assert "딸기 라떼로 바꿔줘" in str(call["prompt"])


def test_edit_reference_image_does_not_overwrite_by_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    make_source_image(source)
    destination.write_bytes(make_png_bytes())

    with pytest.raises(FileExistsError, match="이미 존재"):
        edit_reference_image(
            image_path=source,
            user_request="여름 분위기로 바꿔줘",
            output_path=destination,
            client=FakeClient(make_png_bytes()),
        )


def test_edit_reference_image_rejects_missing_image_data(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    make_source_image(source)
    client = SimpleNamespace(
        images=SimpleNamespace(
            edit=lambda **_: SimpleNamespace(data=[SimpleNamespace(b64_json=None)])
        )
    )

    with pytest.raises(ReferenceImageEditError, match="이미지 데이터를 반환하지"):
        edit_reference_image(
            image_path=source,
            user_request="여름 분위기로 바꿔줘",
            output_path=destination,
            client=client,
        )
