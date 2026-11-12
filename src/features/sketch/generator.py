from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import secrets
from time import perf_counter
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


SKETCH_AD_PROMPT_VERSION = "sketch_ad_v0.2.0-annotation-policy"
SKETCH_AD_MODEL = "black-forest-labs/FLUX.2-klein-4B"
SKETCH_AD_SIZE = "1024x1024"
SKETCH_AD_WIDTH = 1024
SKETCH_AD_HEIGHT = 1024
SKETCH_AD_STEPS = 4
SKETCH_AD_GUIDANCE_SCALE = 1.0
MIN_RECOMMENDED_GPU_MEMORY_GB = 20.0
MAX_USER_REQUEST_CHARS = 4_000
MAX_INPUT_IMAGE_BYTES = 50 * 1024 * 1024
SUPPORTED_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class SketchAdGenerationError(RuntimeError):
    """스케치 기반 광고 이미지 생성을 완료하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class SketchBackendStatus:
    available: bool
    message: str
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None


@dataclass(frozen=True)
class SketchAdGenerationResult:
    output_path: Path
    prompt: str
    prompt_version: str
    model: str
    size: str
    seed: int
    elapsed_seconds: float


SKETCH_AD_INTERNAL_RULES = """The uploaded image is a rough layout sketch for a small cafe product advertisement, not a visual style reference and not a finished image.
Before rendering, distinguish intended scene objects from production annotations.
Treat arrows, rectangular outlines, guide boxes, guide lines, labels, and handwritten marks as production annotations by default.
Use annotations only to infer emphasis, arrangement direction, approximate position, relative scale, camera viewpoint, prop placement, and copy-space location.
Do not reproduce the visual shape, color, stroke, outline, or shadow of any annotation in the final image.
Only include a visible arrow, box, panel, frame, sign, or handwritten mark when the primary request explicitly asks for that element in the final advertisement.
When a rectangle indicates copy space, use only its approximate location and extent. Do not render the rectangle itself.
Render copy space as a seamless and natural continuation of the surrounding wall, tabletop, or background. It must not appear as a white patch, differently colored region, translucent overlay, card, panel, sign, frame, border, or box.
Keep copy space visually simple and low-detail. Do not place products, props, highlights, strong shadows, or busy patterns in that area.
Turn the described product into appetizing photorealistic commercial food photography.
Preserve the intended layout but use physically plausible perspective, lighting, materials, shadows, and food texture.
Only add props that are directly required by the request or clearly indicated as intended scene objects.
Do not render letters, words, prices, logos, watermarks, interface elements, or illustration lines.
The final output must look like a real professionally photographed cafe advertisement image."""


def normalize_user_request(user_request: str) -> str:
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


def build_sketch_ad_prompt(user_request: str) -> str:
    """사용자 요청을 검증된 v0.2 스케치 해석 규칙과 결합한다."""

    normalized = normalize_user_request(user_request)
    return (
        "Primary request:\n"
        f"{normalized}\n\n"
        "Composition and rendering rules:\n"
        f"{SKETCH_AD_INTERNAL_RULES}"
    )


def validate_sketch_image(image_path: str | Path) -> Path:
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"스케치 이미지 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"지원하지 않는 이미지 형식입니다. 허용 형식: {allowed}")
    if path.stat().st_size > MAX_INPUT_IMAGE_BYTES:
        raise ValueError("스케치 이미지는 50MB 이하여야 합니다.")

    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("스케치 이미지를 읽을 수 없습니다.") from exc
    return path


def prepare_sketch_image(image_path: str | Path, size: int = SKETCH_AD_WIDTH) -> Image.Image:
    """비율을 유지하면서 스케치를 흰색 정사각형 캔버스에 맞춘다."""

    path = validate_sketch_image(image_path)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def inspect_sketch_backend() -> SketchBackendStatus:
    """모델을 내려받지 않고 현재 런타임의 실행 가능 여부만 확인한다."""

    if importlib.util.find_spec("diffusers") is None:
        return SketchBackendStatus(
            available=False,
            message="FLUX.2 실행 패키지(diffusers)가 설치되지 않았습니다.",
        )

    try:
        import torch
    except ImportError:
        return SketchBackendStatus(
            available=False,
            message="PyTorch가 설치되지 않았습니다.",
        )

    if not torch.cuda.is_available():
        return SketchBackendStatus(
            available=False,
            message="CUDA GPU를 찾지 못했습니다. L4급 GPU 실행 환경이 필요합니다.",
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb < MIN_RECOMMENDED_GPU_MEMORY_GB:
        return SketchBackendStatus(
            available=False,
            message=(
                f"GPU 메모리가 {gpu_memory_gb:.1f}GB입니다. 1024px 생성에는 "
                f"{MIN_RECOMMENDED_GPU_MEMORY_GB:.0f}GB 이상을 권장합니다."
            ),
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
        )

    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401
    except (ImportError, AttributeError) as exc:
        return SketchBackendStatus(
            available=False,
            message=f"설치된 diffusers가 FLUX.2 Klein을 지원하지 않습니다: {exc}",
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
        )

    return SketchBackendStatus(
        available=True,
        message=f"{gpu_name} ({gpu_memory_gb:.1f}GB)에서 로컬 생성할 수 있습니다.",
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


@lru_cache(maxsize=1)
def _load_flux_pipeline() -> Any:
    status = inspect_sketch_backend()
    if not status.available:
        raise SketchAdGenerationError(status.message)

    import torch
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        pipeline = Flux2KleinPipeline.from_pretrained(
            SKETCH_AD_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipeline.enable_model_cpu_offload()
        if getattr(pipeline, "vae", None) is not None:
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        pipeline.set_progress_bar_config(disable=True)
    except Exception as exc:
        raise SketchAdGenerationError(f"FLUX.2 Klein 모델을 불러오지 못했습니다: {exc}") from exc
    return pipeline


def generate_sketch_ad(
    *,
    image_path: str | Path,
    user_request: str,
    output_path: str | Path,
    seed: int | None = None,
    pipeline: Any | None = None,
) -> SketchAdGenerationResult:
    """스케치 한 장과 요청사항으로 1024px 광고 이미지를 생성한다."""

    source_path = validate_sketch_image(image_path)
    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError("출력 파일 확장자는 .png여야 합니다.")
    if destination.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {destination}")

    prompt = build_sketch_ad_prompt(user_request)
    sketch = prepare_sketch_image(source_path)
    resolved_seed = seed if seed is not None else secrets.randbelow(2**31)

    try:
        import torch

        active_pipeline = pipeline or _load_flux_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(resolved_seed)
        started_at = perf_counter()
        with torch.inference_mode():
            result = active_pipeline(
                image=sketch,
                prompt=prompt,
                height=SKETCH_AD_HEIGHT,
                width=SKETCH_AD_WIDTH,
                num_inference_steps=SKETCH_AD_STEPS,
                guidance_scale=SKETCH_AD_GUIDANCE_SCALE,
                generator=generator,
            ).images[0].convert("RGB")
        elapsed_seconds = perf_counter() - started_at
    except SketchAdGenerationError:
        raise
    except Exception as exc:
        raise SketchAdGenerationError(f"스케치 광고 이미지 생성에 실패했습니다: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, "PNG", optimize=True)
    return SketchAdGenerationResult(
        output_path=destination,
        prompt=prompt,
        prompt_version=SKETCH_AD_PROMPT_VERSION,
        model=SKETCH_AD_MODEL,
        size=SKETCH_AD_SIZE,
        seed=resolved_seed,
        elapsed_seconds=elapsed_seconds,
    )
