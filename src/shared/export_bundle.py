"""완성 배너와 광고 문구를 재현 가능한 다운로드 묶음으로 만든다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import re
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from src.features.ad_copy.generator import AdCopy


EXPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ExportPackage:
    banner_filename: str
    copy_filename: str
    metadata_filename: str
    zip_filename: str
    copy_text: str
    metadata_json: str
    zip_bytes: bytes


def safe_filename_stem(value: str) -> str:
    """주요 운영체제에서 안전한 짧은 파일명 본문을 만든다."""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return cleaned[:60] or "sns_banner"


def build_copy_text(
    copy: AdCopy,
    *,
    product_name: str,
    store_name: str | None,
) -> str:
    """선택한 광고 문구를 사람이 바로 사용할 수 있는 TXT로 만든다."""

    sections = []
    if store_name:
        sections.extend(["매장명", store_name, ""])
    sections.extend(["상품명", product_name, "", "헤드라인", copy["headline"], ""])
    if copy["body"].strip():
        sections.extend(["본문", copy["body"], ""])
    sections.extend(["해시태그", " ".join(copy["hashtags"]), ""])
    return "\n".join(sections)


def build_export_package(
    *,
    banner_png: bytes,
    copy: AdCopy,
    copy_index: int,
    model_id: str,
    model_label: str,
    template_id: str,
    template_label: str,
    ad_input: dict[str, Any],
    generated_at: datetime | None = None,
) -> ExportPackage:
    """배너 PNG, 문구 TXT, 생성 정보 JSON을 ZIP으로 묶는다."""

    if not banner_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("배너 데이터가 PNG 형식이 아닙니다.")

    product_name = str(ad_input.get("product", {}).get("name") or "sns_banner")
    store_name = ad_input.get("store", {}).get("name")
    timestamp = generated_at or datetime.now().astimezone()
    base_name = safe_filename_stem(product_name)
    banner_filename = f"{base_name}_banner.png"
    copy_filename = f"{base_name}_copy.txt"
    metadata_filename = f"{base_name}_metadata.json"
    zip_filename = f"{base_name}_result.zip"

    copy_text = build_copy_text(
        copy,
        product_name=product_name,
        store_name=store_name,
    )
    metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": timestamp.isoformat(timespec="seconds"),
        "model": {"id": model_id, "label": model_label},
        "template": {"id": template_id, "label": template_label},
        "selected_copy": {
            "index": copy_index,
            "headline": copy["headline"],
            "body": copy["body"],
            "hashtags": copy["hashtags"],
        },
        "input": ad_input,
        "output": {
            "banner_filename": banner_filename,
            "width": 1080,
            "height": 1080,
            "format": "PNG",
        },
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"

    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(banner_filename, banner_png)
        archive.writestr(copy_filename, copy_text.encode("utf-8"))
        archive.writestr(metadata_filename, metadata_json.encode("utf-8"))

    return ExportPackage(
        banner_filename=banner_filename,
        copy_filename=copy_filename,
        metadata_filename=metadata_filename,
        zip_filename=zip_filename,
        copy_text=copy_text,
        metadata_json=metadata_json,
        zip_bytes=archive_buffer.getvalue(),
    )
