"""BiRefNet 기반 상품 이미지 배경 제거 기능."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps


MODEL_ID = "ZhengPeng7/BiRefNet"
# trust_remote_code로 불러오는 코드가 예고 없이 바뀌지 않도록 검토한 커밋에 고정한다.
MODEL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
DEFAULT_INFERENCE_SIZE = 1024


class BackgroundRemovalError(RuntimeError):
    """배경 제거 모델 준비 또는 추론 실패를 나타낸다."""


@dataclass(frozen=True)
class BiRefNetBundle:
    """한 번 로드한 모델과 실행 장치를 함께 보관한다."""

    model: Any
    device: Any

    @property
    def device_label(self) -> str:
        return "GPU" if str(self.device).startswith("cuda") else "CPU"


def load_birefnet_model(
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> BiRefNetBundle:
    """공식 Hugging Face BiRefNet 모델을 로드한다.

    첫 호출에서는 모델 파일을 내려받고, 이후 호출에서는 Hugging Face의
    로컬 캐시를 재사용한다.
    """

    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise BackgroundRemovalError(
            "배경 제거 패키지가 설치되지 않았습니다. requirements.txt를 다시 설치해주세요."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise BackgroundRemovalError(
            "BiRefNet 모델을 준비하지 못했습니다. 인터넷 연결을 확인하고 다시 시도해주세요."
        ) from exc

    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.half()
    return BiRefNetBundle(model=model, device=device)


def predict_foreground_mask(
    image: Image.Image,
    bundle: BiRefNetBundle,
    inference_size: int = DEFAULT_INFERENCE_SIZE,
) -> Image.Image:
    """BiRefNet으로 전경 알파 마스크를 예측한다."""

    if inference_size < 256:
        raise ValueError("추론 이미지 크기는 256 이상이어야 합니다.")

    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise BackgroundRemovalError(
            "배경 제거 패키지가 설치되지 않았습니다. requirements.txt를 다시 설치해주세요."
        ) from exc

    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    transform_image = transforms.Compose(
        [
            transforms.Resize((inference_size, inference_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    input_tensor = transform_image(rgb_image).unsqueeze(0).to(bundle.device)
    if str(bundle.device).startswith("cuda"):
        input_tensor = input_tensor.half()

    try:
        with torch.inference_mode():
            outputs = bundle.model(input_tensor)
            logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            prediction = logits.sigmoid().detach().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(prediction).convert("L")
    except Exception as exc:
        raise BackgroundRemovalError(
            "이미지에서 상품 영역을 찾는 중 오류가 발생했습니다."
        ) from exc

    return mask.resize(rgb_image.size, Image.Resampling.LANCZOS)


def compose_cutout(image: Image.Image, mask: Image.Image) -> Image.Image:
    """원본 이미지와 전경 마스크를 투명 배경 RGBA 이미지로 합성한다."""

    source = ImageOps.exif_transpose(image).convert("RGBA")
    alpha = mask.convert("L")
    if alpha.size != source.size:
        alpha = alpha.resize(source.size, Image.Resampling.LANCZOS)
    source.putalpha(alpha)
    return source


def remove_background(
    image: Image.Image,
    bundle: BiRefNetBundle,
    inference_size: int = DEFAULT_INFERENCE_SIZE,
) -> Image.Image:
    """상품 이미지를 투명 배경 PNG로 만들 수 있는 RGBA 이미지로 반환한다."""

    mask = predict_foreground_mask(image, bundle, inference_size=inference_size)
    return compose_cutout(image, mask)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Pillow 이미지를 다운로드 가능한 PNG 바이트로 변환한다."""

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
