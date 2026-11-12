from datetime import datetime, timezone
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image

from src.shared.banner_generator import banner_to_png_bytes
from src.shared.export_bundle import build_copy_text, build_export_package, safe_filename_stem


COPY = {
    "headline": "생딸기의 향, 한 잔에 담다",
    "body": "생딸기를 사용한 딸기 라떼를 만나보세요.",
    "hashtags": ["#딸기라떼", "#생딸기"],
}

AD_INPUT = {
    "schema_version": "1.1",
    "image": {"file_name": "latte.png", "content_type": "image/png"},
    "store": {"name": "오늘의 작은 카페"},
    "product": {
        "name": "딸기 라떼",
        "base_price": "7,000원",
        "sale_price": None,
        "features": "생딸기",
    },
    "campaign": {"promotion": None, "tone": "감성적인"},
}


def test_copy_text_contains_selected_copy() -> None:
    text = build_copy_text(COPY, product_name="딸기 라떼", store_name="오늘의 작은 카페")

    assert "생딸기의 향" in text
    assert "#딸기라떼 #생딸기" in text


def test_copy_text_omits_empty_body_section() -> None:
    copy_without_body = {**COPY, "body": ""}

    text = build_copy_text(
        copy_without_body,
        product_name="딸기 라떼",
        store_name="오늘의 작은 카페",
    )

    assert "본문\n" not in text
    assert "오늘의 작은 카페" in text


def test_export_zip_contains_png_text_and_metadata() -> None:
    banner_png = banner_to_png_bytes(Image.new("RGB", (1080, 1080), "pink"))
    package = build_export_package(
        banner_png=banner_png,
        copy=COPY,
        copy_index=2,
        model_id="model-id",
        model_label="테스트 모델",
        template_id="berry",
        template_label="딸기 크림",
        ad_input=AD_INPUT,
        generated_at=datetime(2026, 10, 16, 6, 0, tzinfo=timezone.utc),
    )

    with ZipFile(BytesIO(package.zip_bytes)) as archive:
        assert set(archive.namelist()) == {
            package.banner_filename,
            package.copy_filename,
            package.metadata_filename,
        }
        assert archive.read(package.banner_filename) == banner_png
        assert "생딸기" in archive.read(package.copy_filename).decode("utf-8")
        metadata = json.loads(archive.read(package.metadata_filename).decode("utf-8"))

    assert metadata["model"]["id"] == "model-id"
    assert metadata["selected_copy"]["index"] == 2
    assert metadata["output"]["width"] == 1080
    assert "API_KEY" not in package.metadata_json


def test_safe_filename_stem_removes_windows_reserved_characters() -> None:
    assert safe_filename_stem('딸기:라떼/특가?') == "딸기_라떼_특가_"
