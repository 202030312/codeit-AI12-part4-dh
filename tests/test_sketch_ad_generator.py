from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from src.features.sketch.generator import (
    SKETCH_AD_MODEL,
    SKETCH_AD_PROMPT_VERSION,
    build_sketch_ad_prompt,
    prepare_sketch_image,
    validate_sketch_image,
)


def test_build_prompt_includes_request_and_annotation_policy() -> None:
    request = "햄치즈 토스트를 가운데 크게 배치하고 오른쪽 위는 문구 공간으로 비워주세요."

    prompt = build_sketch_ad_prompt(f"  {request}  ")

    assert request in prompt
    assert "rough layout sketch" in prompt
    assert "Treat arrows" in prompt
    assert "Do not reproduce the visual shape" in prompt
    assert "When a rectangle indicates copy space" in prompt
    assert "natural continuation" in prompt
    assert "Do not render letters" in prompt
    assert SKETCH_AD_PROMPT_VERSION == "sketch_ad_v0.2.0-annotation-policy"
    assert SKETCH_AD_MODEL == "black-forest-labs/FLUX.2-klein-4B"


@pytest.mark.parametrize("user_request", ["", "   ", "\n\t"])
def test_build_prompt_rejects_blank_request(user_request: str) -> None:
    with pytest.raises(ValueError, match="요청사항"):
        build_sketch_ad_prompt(user_request)


def test_prepare_sketch_centers_rectangular_image_on_square_canvas(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sketch.png"
    Image.new("RGB", (400, 200), "black").save(source)

    prepared = prepare_sketch_image(source, size=100)

    assert prepared.size == (100, 100)
    assert prepared.getpixel((50, 50)) == (0, 0, 0)
    assert prepared.getpixel((50, 5)) == (255, 255, 255)


def test_validate_sketch_rejects_invalid_image(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="읽을 수 없습니다"):
        validate_sketch_image(source)
