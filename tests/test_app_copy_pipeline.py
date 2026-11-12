from __future__ import annotations

from unittest.mock import patch

from src.features.ad_copy.pipeline import (
    COPY_MODEL_ID,
    PIPELINE_VERSION,
    VISUAL_MODEL_ID,
    generate_app_copy,
)
from src.features.ad_copy.models import BannerCopyGenerationResult
from src.features.ad_copy.visual_analysis import VisualFactExtractionResult


def test_app_pipeline_sends_image_only_to_gemini_visual_stage() -> None:
    ad_input = {
        "product": {"name": "블루베리 머핀", "features": "블루베리 풍미"},
        "campaign": {
            "focus_feature": "블루베리 풍미",
            "generation_mode": "headline_and_subcopy",
        },
    }
    visual_result = VisualFactExtractionResult(
        verified_facts=(
            {
                "category": "visible_component",
                "value": "보라색 열매 토핑",
                "confidence": "high",
            },
        ),
        rejected_observations=(),
        excluded_inferences=("토핑의 맛",),
        status="extracted",
        prompt_version="gemini_visual_facts_v1.1.0-korean-retry",
    )
    copy_result = BannerCopyGenerationResult(
        copies=[
            {"headline": "블루베리 풍미의 한입", "body": "보라색 열매 토핑", "hashtags": []},
            {"headline": "머핀에 담은 블루베리", "body": "보라색 열매가 보이는 토핑", "hashtags": []},
        ],
        generation_mode="headline_and_subcopy",
        status="generated",
        attempts=1,
        prompt_version="openai_copy_v1.10.14-product-type-dedup",
    )

    with (
        patch(
            "src.features.ad_copy.pipeline.extract_verified_visual_facts",
            return_value=visual_result,
        ) as visual_mock,
        patch(
            "src.features.ad_copy.pipeline.generate_safe_banner_copy_options",
            return_value=copy_result,
        ) as copy_mock,
    ):
        result = generate_app_copy(
            ad_input,
            image_bytes=b"raw-image",
            image_mime_type="image/png",
            gemini_api_key="dummy",
            openai_api_key="dummy",
        )

    visual_kwargs = visual_mock.call_args.kwargs
    assert visual_kwargs["model_id"] == VISUAL_MODEL_ID
    assert visual_kwargs["image_bytes"] == b"raw-image"
    assert visual_kwargs["image_mime_type"] == "image/png"

    copy_args = copy_mock.call_args.args
    copy_kwargs = copy_mock.call_args.kwargs
    assert copy_kwargs["model_id"] == COPY_MODEL_ID
    assert "image_bytes" not in copy_kwargs
    assert "image_mime_type" not in copy_kwargs
    assert copy_args[0]["campaign"]["verified_visual_facts"] == [
        {
            "category": "visible_component",
            "value": "보라색 열매 토핑",
            "confidence": "high",
        }
    ]
    assert result.copy_result is copy_result
    assert result.pipeline_version == PIPELINE_VERSION


def test_app_pipeline_keeps_running_when_visual_extraction_is_omitted() -> None:
    ad_input = {
        "product": {"name": "콜드브루", "features": "깔끔한 커피 풍미"},
        "campaign": {
            "focus_feature": "깔끔한 커피 풍미",
            "generation_mode": "headline_and_subcopy",
        },
    }
    visual_result = VisualFactExtractionResult(
        verified_facts=(),
        rejected_observations=(),
        excluded_inferences=(),
        status="omitted",
        notice="시각 정보 추출을 생략했습니다.",
    )
    copy_result = BannerCopyGenerationResult(
        copies=[
            {"headline": "깔끔한 커피 풍미", "body": "", "hashtags": []},
            {"headline": "콜드브루의 깔끔함", "body": "", "hashtags": []},
        ],
        generation_mode="headline_only",
        status="generated",
        attempts=1,
    )

    with (
        patch(
            "src.features.ad_copy.pipeline.extract_verified_visual_facts",
            return_value=visual_result,
        ),
        patch(
            "src.features.ad_copy.pipeline.generate_safe_banner_copy_options",
            return_value=copy_result,
        ) as copy_mock,
    ):
        result = generate_app_copy(
            ad_input,
            image_bytes=b"raw-image",
            image_mime_type="image/jpeg",
            gemini_api_key="dummy",
            openai_api_key="dummy",
        )

    enriched = copy_mock.call_args.args[0]
    assert enriched["campaign"]["verified_visual_facts"] == []
    assert enriched["campaign"]["visual_extraction_status"] == "omitted"
    assert result.copy_result.generation_mode == "headline_only"
