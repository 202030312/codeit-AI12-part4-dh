"""앱에서 사용하는 최종 광고 문구 생성 파이프라인.

원본 이미지는 Gemini 시각 정보 추출 단계에만 전달하고, GPT 문구 생성
단계에는 검증된 시각 정보가 결합된 구조화 입력만 전달한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.features.ad_copy.visual_analysis import VisualFactExtractionResult
from src.features.ad_copy.workflow import (
    BannerCopyGenerationResult,
    attach_visual_context,
    extract_verified_visual_facts,
    generate_safe_banner_copy_options,
)


PIPELINE_VERSION = "gemini-visual-v1.1+gpt-copy-v1.10.18"
VISUAL_MODEL_ID = "gemini-3.5-flash-lite"
COPY_MODEL_ID = "gpt-6-luna"


@dataclass(frozen=True)
class AppCopyPipelineResult:
    """앱 화면과 이후 배너 단계에 전달할 통합 결과."""

    enriched_input: dict[str, Any]
    visual_result: VisualFactExtractionResult
    copy_result: BannerCopyGenerationResult
    pipeline_version: str = PIPELINE_VERSION


def generate_app_copy(
    ad_input: dict[str, Any],
    *,
    image_bytes: bytes,
    image_mime_type: str,
    gemini_api_key: str,
    openai_api_key: str,
    visual_client: Any | None = None,
    copy_client: Any | None = None,
) -> AppCopyPipelineResult:
    """Gemini 시각 추출 후 정제된 시각 사실로 GPT v1.10.18 문구를 생성한다."""

    visual_result = extract_verified_visual_facts(
        menu_name=str(ad_input.get("product", {}).get("name") or ""),
        model_id=VISUAL_MODEL_ID,
        api_key=gemini_api_key,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        client=visual_client,
    )
    enriched_input = attach_visual_context(ad_input, visual_result)
    copy_result = generate_safe_banner_copy_options(
        enriched_input,
        model_id=COPY_MODEL_ID,
        api_key=openai_api_key,
        client=copy_client,
    )
    return AppCopyPipelineResult(
        enriched_input=enriched_input,
        visual_result=visual_result,
        copy_result=copy_result,
    )
