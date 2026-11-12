from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from src.shared.banner_generator import (
    BANNER_TEMPLATES,
    TEMPLATE_MAP,
    banner_text_safe_areas,
    banner_to_png_bytes,
    generate_banner,
    resolve_auto_text_space,
)
from src.shared.background_remover import (
    BackgroundRemovalError,
    image_to_png_bytes,
    load_birefnet_model,
    remove_background,
)
from src.features.ad_copy.pipeline import (
    COPY_MODEL_ID,
    PIPELINE_VERSION,
    VISUAL_MODEL_ID,
    generate_app_copy,
)
from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    CopyGenerationError,
    resolve_api_key,
)
from src.shared.combined_banner_renderer import render_combined_banner
from src.shared.export_bundle import build_export_package
from src.features.product_preserve.service import (
    build_feature3_request,
    recommended_template_ids,
    FEATURE3_TEMPLATES,
    feature3_image_to_png_bytes,
    render_feature3_image,
)
from src.shared.input_schema import build_ad_input
from src.features.ad_copy.promotion import (
    BADGE_ONLY,
    HEADLINE_ALLOWED,
    headline_mentions_promotion,
    promotion_badge_text,
)
from src.features.reference_edit.service import (
    REFERENCE_IMAGE_EDIT_MODEL,
    REFERENCE_IMAGE_EDIT_QUALITY,
    REFERENCE_IMAGE_EDIT_SIZE,
    REFERENCE_IMAGE_EDIT_VERSION,
    ReferenceImageEditError,
    edit_reference_image,
)
from src.features.sketch.service import (
    SKETCH_AD_PROMPT_VERSION,
    SketchAdGenerationError,
    generate_sketch_ad,
)


APP_TITLE = "가게 한컷"
MENU_COPY = "광고 문구 추천"
MENU_REFERENCE_EDIT = "사진 새롭게 바꾸기"
MENU_PRODUCT_PRESERVE = "상품 사진 꾸미기"
MENU_SKETCH_AD = "스케치를 광고로 완성하기"
MAIN_MENU_OPTIONS = (
    MENU_COPY,
    MENU_PRODUCT_PRESERVE,
    MENU_REFERENCE_EDIT,
    MENU_SKETCH_AD,
)
MAIN_MENU_DESCRIPTIONS = {
    MENU_COPY: "상품에 어울리는 제목과 홍보 문구를 제안합니다.",
    MENU_REFERENCE_EDIT: "입력 사진의 상품·배경·분위기·구도를 요청대로 바꿉니다.",
    MENU_PRODUCT_PRESERVE: "입력한 상품은 유지하고 배경과 광고 연출을 새롭게 만듭니다.",
    MENU_SKETCH_AD: "스케치의 배치를 참고해 요청한 카페 광고 이미지를 새로 만듭니다.",
}
VISUAL_MODEL_CONFIG = MODEL_CONFIGS[VISUAL_MODEL_ID]
COPY_MODEL_CONFIG = MODEL_CONFIGS[COPY_MODEL_ID]
DEFAULT_COPY_TONE = "자연스럽고 명확하게"
COPY_TONE_OPTIONS = (
    "auto",
    "친근하고 자연스럽게",
    "따뜻하고 감성적으로",
    "밝고 활기차게",
    "차분하고 고급스럽게",
    "트렌디하고 감각적으로",
    "짧고 직설적으로",
    "담백하고 신뢰감 있게",
)
COPY_TONE_LABELS = {
    "auto": "자동 추천",
    **{value: value for value in COPY_TONE_OPTIONS if value != "auto"},
}
BRAND_STYLE_DETAILS = {
    "berry": ("핑크·크림", "왼쪽 상단"),
    "cream": ("크림·브라운", "상단 중앙"),
    "night": ("자주·골드", "왼쪽 상단"),
    "clean": ("화이트·청록", "왼쪽 상단"),
    "natural": ("베이지·올리브", "상단 중앙"),
    "promo": ("옐로·레드", "왼쪽 상단"),
}
FEATURE3_CATEGORY_LABELS = {
    "bread": "빵",
    "hot_drink": "핫 음료",
    "iced_drink": "아이스 음료",
    "dessert": "디저트",
    "set_gift": "세트·선물",
}
FEATURE3_PURPOSE_LABELS = {
    "image_only": "이미지 단독 사용",
    "copy_later": "나중에 문구 추가",
}
FEATURE3_TEXT_SPACE_LABELS = {
    "auto": "자동",
    "top": "상단",
    "bottom": "하단",
    "left": "왼쪽",
    "right": "오른쪽",
    "none": "없음",
}
COPY_TEXT_POSITION_OPTIONS = ("auto", "top", "bottom", "left", "right")


load_dotenv()


def apply_cafe_theme() -> None:
    """따뜻한 카페 분위기의 공통 Streamlit 스타일을 적용한다."""

    st.markdown(
        """
        <style>
        :root {
            --cafe-espresso: #51372b;
            --cafe-coffee: #8f563b;
            --cafe-caramel: #b97a52;
            --cafe-cream: #fbf7f1;
            --cafe-oat: #f2e9de;
            --cafe-paper: #fffdf9;
            --cafe-sage: #76836b;
            --cafe-ink: #352a24;
            --cafe-muted: #75665d;
            --cafe-line: rgba(106, 74, 56, 0.16);
            --cafe-shadow: 0 14px 40px rgba(88, 57, 40, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(185, 122, 82, 0.10), transparent 26rem),
                radial-gradient(circle at 8% 94%, rgba(118, 131, 107, 0.09), transparent 28rem),
                var(--cafe-cream);
            color: var(--cafe-ink);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .main .block-container {
            max-width: 1240px;
            padding-top: 2.1rem;
            padding-bottom: 5rem;
        }

        .cafe-hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.1rem;
            padding: 2.15rem 2.5rem 2rem;
            border: 1px solid var(--cafe-line);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255, 253, 249, 0.98), rgba(242, 233, 222, 0.92));
            box-shadow: var(--cafe-shadow);
        }

        .cafe-hero::after {
            content: "";
            position: absolute;
            width: 190px;
            height: 190px;
            right: -48px;
            top: -82px;
            border-radius: 50%;
            border: 34px solid rgba(143, 86, 59, 0.08);
        }

        .cafe-hero__eyebrow {
            margin-bottom: 0.55rem;
            color: var(--cafe-coffee);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.18em;
        }

        .cafe-hero h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            color: var(--cafe-espresso);
            font-size: clamp(2.1rem, 4vw, 3.25rem);
            line-height: 1.08;
            letter-spacing: -0.055em;
        }

        .cafe-hero p {
            position: relative;
            z-index: 1;
            max-width: 680px;
            margin: 0.8rem 0 0;
            color: var(--cafe-muted);
            font-size: 1rem;
            line-height: 1.7;
        }

        h1, h2, h3, h4, h5 {
            color: var(--cafe-espresso) !important;
            letter-spacing: -0.035em !important;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--cafe-line);
            background:
                linear-gradient(180deg, rgba(245, 237, 226, 0.98), rgba(235, 224, 210, 0.98));
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        [data-testid="stSidebar"] h2 {
            margin-bottom: 0.15rem;
            font-size: 1.28rem;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] > div > p {
            color: var(--cafe-muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        [data-testid="stSidebar"] label[data-baseweb="radio"] {
            width: 100%;
            margin: 0.22rem 0;
            padding: 0.72rem 0.8rem;
            border: 1px solid transparent;
            border-radius: 14px;
            transition: all 160ms ease;
        }

        [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
            border-color: rgba(143, 86, 59, 0.18);
            background: rgba(255, 253, 249, 0.58);
        }

        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
            border-color: rgba(143, 86, 59, 0.25);
            background: var(--cafe-paper);
            box-shadow: 0 8px 22px rgba(88, 57, 40, 0.09);
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            margin-top: 0.7rem;
            padding: 0.8rem 0.9rem;
            border-left: 3px solid var(--cafe-caramel);
            border-radius: 0 10px 10px 0;
            background: rgba(255, 253, 249, 0.52);
        }

        div[data-testid="stAlert"] {
            border: 1px solid rgba(118, 131, 107, 0.20);
            border-radius: 16px;
            background: rgba(239, 241, 234, 0.88);
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 132px;
            border: 1.5px dashed rgba(143, 86, 59, 0.42);
            border-radius: 18px;
            background: rgba(255, 253, 249, 0.82);
        }

        [data-baseweb="input"],
        [data-baseweb="textarea"],
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"] {
            border-color: rgba(106, 74, 56, 0.22) !important;
            border-radius: 12px !important;
            background: rgba(255, 253, 249, 0.92) !important;
        }

        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"]:focus-within,
        [data-baseweb="select"] > div:focus-within {
            border-color: var(--cafe-caramel) !important;
            box-shadow: 0 0 0 3px rgba(185, 122, 82, 0.12) !important;
        }

        [data-testid="stSegmentedControl"] button {
            min-height: 2.35rem;
            border-color: rgba(143, 86, 59, 0.16) !important;
            color: var(--cafe-espresso) !important;
            background: rgba(255, 253, 249, 0.86) !important;
            font-weight: 700;
        }

        [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            border-color: var(--cafe-coffee) !important;
            color: #fffaf5 !important;
            background: var(--cafe-coffee) !important;
            box-shadow: 0 5px 14px rgba(88, 57, 40, 0.13);
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.8rem;
            border: 1px solid rgba(143, 86, 59, 0.20);
            border-radius: 13px;
            color: var(--cafe-espresso);
            background: var(--cafe-paper);
            font-weight: 750;
            box-shadow: 0 5px 14px rgba(88, 57, 40, 0.06);
            transition: transform 150ms ease, box-shadow 150ms ease, background 150ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: var(--cafe-caramel);
            color: var(--cafe-coffee);
            box-shadow: 0 9px 22px rgba(88, 57, 40, 0.11);
        }

        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            border-color: var(--cafe-coffee);
            color: #fffaf5;
            background: linear-gradient(135deg, var(--cafe-coffee), #a96b49);
        }

        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover {
            color: #ffffff;
            background: linear-gradient(135deg, #7f492f, var(--cafe-coffee));
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid var(--cafe-line);
            border-radius: 15px;
            background: rgba(255, 253, 249, 0.64);
        }

        [data-testid="stImage"] img {
            border: 1px solid var(--cafe-line);
            border-radius: 18px;
            box-shadow: var(--cafe-shadow);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--cafe-line) !important;
            border-radius: 18px !important;
            background: rgba(255, 253, 249, 0.72);
            box-shadow: 0 8px 26px rgba(88, 57, 40, 0.05);
        }

        [data-testid="stCaptionContainer"] p,
        small {
            color: var(--cafe-muted) !important;
        }

        hr {
            border-color: var(--cafe-line) !important;
        }

        @media (max-width: 760px) {
            .main .block-container {
                padding-top: 1rem;
            }
            .cafe-hero {
                padding: 1.55rem 1.35rem;
                border-radius: 21px;
            }
            .cafe-hero p {
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <section class="cafe-hero">
            <div class="cafe-hero__eyebrow">CAFE SOCIAL STUDIO</div>
            <h1>가게 한컷</h1>
            <p>카페·디저트 매장을 위한 SNS 광고 문구와 이미지 제작 도구</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def set_workflow_step(step: str) -> None:
    st.session_state["workflow_step"] = step


def set_main_menu(menu: str) -> None:
    st.session_state["main_menu"] = menu


def open_copy_menu() -> None:
    st.session_state["main_menu"] = MENU_COPY
    st.session_state["workflow_step"] = "copy"


def skip_image_step() -> None:
    st.session_state.pop("feature3_final_result", None)
    open_copy_menu()


def skip_copy_step() -> None:
    st.session_state.pop("copy_results", None)
    st.session_state.pop("selected_copy_result", None)
    st.session_state.pop("banner_result", None)
    st.session_state["workflow_step"] = "results"


class SessionUploadedImage:
    def __init__(self, *, name: str, content_type: str, data: bytes) -> None:
        self.name = name
        self.type = content_type
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def load_uploaded_image(uploaded_file) -> Image.Image:
    """업로드 파일을 안전하게 열고 화면 표시용 RGB 이미지로 변환한다."""
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    image.verify()
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    return image.convert("RGB")


@st.cache_resource(show_spinner=False)
def get_background_removal_model():
    """한 번 로드한 BiRefNet 모델을 Streamlit 재실행에서도 재사용한다."""

    return load_birefnet_model()


def render_copy_record(record: dict) -> None:
    """세션에 저장된 문구 생성 상태와 검증 정보를 표시한다."""

    with st.container(border=True):
        if record.get("error"):
            st.warning(record["error"])
            return

        status = record["status"]
        if status == "generated":
            st.success("문구 후보 생성 · 입력 근거 검증 통과")
        elif status == "retried":
            st.warning("문구 재생성 성공 · 첫 응답의 검증 문제 수정")
        elif status == "mixed":
            st.warning("모델 문구 유지 · 부족한 후보만 안전 문구로 보완")
        else:
            st.warning("안전한 fallback 사용 · 확인된 입력만으로 구성")

        st.caption(
            f"총 처리 시간: {record['elapsed_seconds']:.2f}초 · "
            f"모델 호출 {record['attempts']}회"
        )
        if record.get("notice"):
            st.caption(record["notice"])
        visual_status = record.get("visual_status")
        if visual_status == "extracted":
            st.caption(
                f"Gemini 시각 정보 {len(record.get('visual_facts', []))}개를 "
                "검증된 입력으로 사용했습니다."
            )
        elif record.get("visual_notice"):
            st.caption(record["visual_notice"])
        st.caption(f"선택 가능한 문구 후보 {len(record.get('copies', []))}개")
        with st.expander("Gemini 시각 정보 추출 결과"):
            st.json(
                {
                    "status": visual_status,
                    "verified_visual_facts": record.get("visual_facts", []),
                    "rejected_visual_observations": record.get(
                        "rejected_visual_observations", []
                    ),
                    "excluded_visual_inferences": record.get(
                        "excluded_visual_inferences", []
                    ),
                }
            )
        if record.get("validation_issues"):
            issue_label = (
                "첫 응답에서 자동 수정된 항목"
                if status == "retried"
                else "기본 문구로 대체하게 된 검증 문제"
            )
            with st.expander(issue_label):
                for issue in record["validation_issues"]:
                    st.write(f"- {issue}")


def render_reference_edit_workflow() -> None:
    """레퍼런스 사진과 요청사항으로 기능 2 광고 이미지를 생성한다."""

    st.info(
        "참고할 사진 한 장을 올리고 바꾸고 싶은 내용을 적어주세요. 상품 교체, "
        "배경과 계절 분위기, 카메라 각도와 카피 공간 등을 함께 요청할 수 있습니다."
    )
    input_column, preview_column = st.columns([1, 1], gap="large")

    with input_column:
        st.subheader("1. 참고할 사진 업로드")
        uploaded_file = st.file_uploader(
            "참고할 사진",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG 또는 WEBP 파일 한 장을 업로드하세요. 최대 50MB까지 지원합니다.",
            key="reference_edit_source_image",
        )
        st.subheader("2. 어떻게 바꿔드릴까요?")
        user_request = st.text_area(
            "요청사항",
            placeholder=(
                "예: 망고빙수를 딸기빙수로 바꾸고 생딸기와 딸기 소스가 "
                "잘 보이는 밝은 여름 광고 사진으로 만들어주세요."
            ),
            help=(
                "바꿀 상품이나 배경, 분위기, 카메라 각도와 여백을 자유롭게 적어주세요. "
                "유지할 컵·접시·상품이 있다면 함께 적는 것이 좋습니다."
            ),
            max_chars=4_000,
            height=160,
            key="reference_edit_user_request",
        )
        st.caption(f"{len(user_request):,}/4,000자")
        generate_clicked = st.button(
            "광고 이미지 만들기",
            type="primary",
            width="stretch",
            key="reference_edit_generate",
        )

    preview_image = None
    uploaded_bytes = None
    image_key = None
    with preview_column:
        st.subheader("3. 원본 미리보기")
        if uploaded_file is None:
            st.warning("사진을 업로드하면 여기에 원본 미리보기가 표시됩니다.")
        else:
            uploaded_bytes = uploaded_file.getvalue()
            image_key = sha256(uploaded_bytes).hexdigest()
            try:
                preview_image = load_uploaded_image(uploaded_file)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("이미지를 읽을 수 없습니다. 다른 JPG, PNG 또는 WEBP 파일을 선택해주세요.")
            else:
                st.image(
                    preview_image,
                    caption=(
                        f"{uploaded_file.name} · "
                        f"{preview_image.width}×{preview_image.height}px"
                    ),
                    width="stretch",
                )

    normalized_request = user_request.strip()
    request_signature = None
    if image_key and normalized_request:
        request_signature = sha256(
            f"{image_key}\n{normalized_request}\n{REFERENCE_IMAGE_EDIT_VERSION}".encode(
                "utf-8"
            )
        ).hexdigest()

    if generate_clicked:
        if uploaded_file is None or uploaded_bytes is None or preview_image is None:
            st.error("정상적인 참고 사진을 업로드해주세요.")
        elif not normalized_request:
            st.error("어떻게 바꿀지 요청사항을 입력해주세요.")
        else:
            suffix = Path(uploaded_file.name).suffix.lower()
            started_at = perf_counter()
            try:
                with st.spinner("요청사항을 반영해 광고 이미지를 만들고 있습니다..."):
                    with TemporaryDirectory(prefix="reference-image-edit-") as temp_dir:
                        temp_root = Path(temp_dir)
                        source_path = temp_root / f"source{suffix}"
                        output_path = temp_root / "result.png"
                        source_path.write_bytes(uploaded_bytes)
                        result = edit_reference_image(
                            image_path=source_path,
                            user_request=normalized_request,
                            output_path=output_path,
                        )
                        result_png = output_path.read_bytes()
            except (ValueError, OSError, ReferenceImageEditError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("광고 이미지를 만드는 중 예상하지 못한 오류가 발생했습니다.")
            else:
                output_stem = Path(uploaded_file.name).stem or "reference"
                st.session_state["reference_edit_result"] = {
                    "signature": request_signature,
                    "png_bytes": result_png,
                    "file_name": f"{output_stem}_edited_ad.png",
                    "request": normalized_request,
                    "model": result.model,
                    "prompt_version": result.prompt_version,
                    "size": result.size,
                    "quality": result.quality,
                    "elapsed_seconds": perf_counter() - started_at,
                }

    result_state = st.session_state.get("reference_edit_result") or {}
    if request_signature and result_state.get("signature") == request_signature:
        st.divider()
        st.subheader("4. 생성 결과")
        original_column, result_column = st.columns(2, gap="large")
        with original_column:
            st.image(preview_image, caption="원본 사진", width="stretch")
        with result_column:
            st.image(
                result_state["png_bytes"],
                caption="요청사항을 반영한 광고 이미지",
                width="stretch",
            )
        st.success("광고 이미지가 완성되었습니다. 원본과 비교한 뒤 다운로드해주세요.")
        st.download_button(
            "광고 이미지 PNG 다운로드",
            data=result_state["png_bytes"],
            file_name=result_state["file_name"],
            mime="image/png",
            type="primary",
            width="stretch",
        )
        with st.expander("생성 정보 확인"):
            st.json(
                {
                    "요청사항": result_state["request"],
                    "모델": result_state["model"],
                    "내부 프롬프트 버전": result_state["prompt_version"],
                    "크기": result_state["size"],
                    "품질": result_state["quality"],
                    "처리 시간(초)": round(result_state["elapsed_seconds"], 2),
                }
            )

    with st.expander("지원하는 요청 예시"):
        st.markdown(
            """
            - `말차라떼를 딸기라떼로 바꾸고 컵과 받침은 그대로 유지해주세요.`
            - `상품은 그대로 두고 비 오는 저녁 카페 분위기로 바꿔주세요.`
            - `위에서 45도 내려다본 구도로 바꾸고 오른쪽 위를 비워주세요.`
            """
        )


def render_sketch_ad_workflow() -> None:
    """스케치와 자유 요청사항으로 기능 4 광고 이미지를 생성한다."""

    st.info(
        "손으로 그린 스케치나 간단한 배치 이미지를 올리고 만들고 싶은 상품과 "
        "분위기를 적어주세요. 스케치의 위치·크기·구도를 참고해 문구 없는 광고 "
        "이미지 한 장을 만듭니다."
    )
    input_column, preview_column = st.columns([1, 1], gap="large")

    with input_column:
        st.subheader("1. 스케치 이미지 업로드")
        uploaded_file = st.file_uploader(
            "스케치 이미지",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG 또는 WEBP 파일 한 장을 업로드하세요. 최대 50MB까지 지원합니다.",
            key="sketch_ad_source_image",
        )
        st.subheader("2. 어떤 광고 이미지로 만들까요?")
        user_request = st.text_area(
            "요청사항",
            placeholder=(
                "예: 햄치즈 토스트를 가운데 크게 배치하고 바삭한 빵과 녹은 치즈가 "
                "잘 보이게 해주세요. 따뜻한 아침 햇살 분위기로 만들고 오른쪽 위는 "
                "문구를 넣을 수 있도록 자연스럽게 비워주세요."
            ),
            help=(
                "상품명, 강조할 특징, 배경과 조명, 계절·분위기, 카메라 각도, "
                "문구를 넣을 여백을 자유롭게 적어주세요."
            ),
            max_chars=4_000,
            height=180,
            key="sketch_ad_user_request",
        )
        st.caption(f"{len(user_request):,}/4,000자")
        generate_clicked = st.button(
            "스케치로 광고 이미지 만들기",
            type="primary",
            width="stretch",
            key="sketch_ad_generate",
        )

    preview_image = None
    uploaded_bytes = None
    image_key = None
    with preview_column:
        st.subheader("3. 스케치 미리보기")
        if uploaded_file is None:
            st.warning("스케치를 업로드하면 여기에 미리보기가 표시됩니다.")
        else:
            uploaded_bytes = uploaded_file.getvalue()
            image_key = sha256(uploaded_bytes).hexdigest()
            try:
                preview_image = load_uploaded_image(uploaded_file)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("이미지를 읽을 수 없습니다. 다른 JPG, PNG 또는 WEBP 파일을 선택해주세요.")
            else:
                st.image(
                    preview_image,
                    caption=(
                        f"{uploaded_file.name} · "
                        f"{preview_image.width}×{preview_image.height}px"
                    ),
                    width="stretch",
                )

    normalized_request = user_request.strip()
    request_signature = None
    if image_key and normalized_request:
        request_signature = sha256(
            f"{image_key}\n{normalized_request}\n{SKETCH_AD_PROMPT_VERSION}".encode(
                "utf-8"
            )
        ).hexdigest()

    if generate_clicked:
        if uploaded_file is None or uploaded_bytes is None or preview_image is None:
            st.error("정상적인 스케치 이미지를 업로드해주세요.")
        elif not normalized_request:
            st.error("만들고 싶은 광고 이미지의 요청사항을 입력해주세요.")
        else:
            suffix = Path(uploaded_file.name).suffix.lower()
            try:
                with st.spinner(
                    "스케치의 배치와 요청사항을 반영해 광고 이미지를 만들고 있습니다..."
                ):
                    with TemporaryDirectory(prefix="sketch-ad-") as temp_dir:
                        temp_root = Path(temp_dir)
                        source_path = temp_root / f"sketch{suffix}"
                        output_path = temp_root / "result.png"
                        source_path.write_bytes(uploaded_bytes)
                        result = generate_sketch_ad(
                            image_path=source_path,
                            user_request=normalized_request,
                            output_path=output_path,
                        )
                        result_png = output_path.read_bytes()
            except (ValueError, OSError, SketchAdGenerationError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("스케치 광고 이미지를 만드는 중 예상하지 못한 오류가 발생했습니다.")
            else:
                output_stem = Path(uploaded_file.name).stem or "sketch"
                st.session_state["sketch_ad_result"] = {
                    "signature": request_signature,
                    "png_bytes": result_png,
                    "file_name": f"{output_stem}_sketch_ad.png",
                    "request": normalized_request,
                    "model": result.model,
                    "prompt_version": result.prompt_version,
                    "size": result.size,
                    "seed": result.seed,
                    "elapsed_seconds": result.elapsed_seconds,
                }

    result_state = st.session_state.get("sketch_ad_result") or {}
    if request_signature and result_state.get("signature") == request_signature:
        st.divider()
        st.subheader("4. 생성 결과")
        sketch_column, result_column = st.columns(2, gap="large")
        with sketch_column:
            st.image(preview_image, caption="입력 스케치", width="stretch")
        with result_column:
            st.image(
                result_state["png_bytes"],
                caption="스케치와 요청사항을 반영한 광고 이미지",
                width="stretch",
            )
        st.success("광고 이미지가 완성되었습니다. 스케치와 비교한 뒤 다운로드해주세요.")
        st.download_button(
            "광고 이미지 PNG 다운로드",
            data=result_state["png_bytes"],
            file_name=result_state["file_name"],
            mime="image/png",
            type="primary",
            width="stretch",
        )
        with st.expander("생성 정보 확인"):
            st.json(
                {
                    "요청사항": result_state["request"],
                    "모델": result_state["model"],
                    "내부 프롬프트 버전": result_state["prompt_version"],
                    "크기": result_state["size"],
                    "시드": result_state["seed"],
                    "처리 시간(초)": round(result_state["elapsed_seconds"], 2),
                }
            )

    with st.expander("요청사항 작성 예시"):
        st.markdown(
            """
            - `복숭아 아이스티를 왼쪽 아래에 크게 배치하고 맑은 얼음과 복숭아 조각이 잘 보이게 해주세요. 오른쪽은 자연스럽게 비워주세요.`
            - `딸기 케이크 한 조각을 중앙에 놓고 부드러운 생크림과 생딸기를 강조한 봄 신메뉴 광고로 만들어주세요.`
            - `소금빵 세 개를 스케치 구도대로 배치하고 따뜻한 아침 햇살과 담백한 베이커리 분위기로 만들어주세요.`
            """
        )


def render_feature3_workflow() -> None:
    """기능 3의 카테고리·용도·템플릿 입력 화면을 그린다."""

    st.info(
        "상품 사진이 보존된 상태에서 메뉴 카테고리·사용 목적·스타일 템플릿에 "
        "맞춘 문구 없는 1080×1080 광고 이미지를 준비합니다. 문구 추가 예정이면 "
        "빈 카드 없이 자연스러운 여백만 확보합니다."
    )
    selected_template = st.session_state.get("feature3_selected_template") or {}
    input_column, preview_column = st.columns([1, 1], gap="large")

    with input_column:
        st.subheader("1. 상품 사진")
        uploaded_file = st.file_uploader(
            "상품 사진",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG 또는 WEBP 파일 한 장을 업로드하세요.",
            key="feature3_product_image",
        )

        st.subheader("2. 카테고리와 사용 목적")
        menu_category = st.selectbox(
            "메뉴 카테고리",
            options=list(FEATURE3_CATEGORY_LABELS),
            format_func=FEATURE3_CATEGORY_LABELS.get,
            key="feature3_menu_category",
        )
        image_purpose = st.radio(
            "이미지 사용 목적",
            options=list(FEATURE3_PURPOSE_LABELS),
            format_func=FEATURE3_PURPOSE_LABELS.get,
            horizontal=True,
            key="feature3_image_purpose",
        )
        text_space_options = (
            ["none"]
            if image_purpose == "image_only"
            else ["auto", "top", "bottom", "left", "right"]
        )
        text_space = st.radio(
            "문구 여백",
            options=text_space_options,
            format_func=FEATURE3_TEXT_SPACE_LABELS.get,
            horizontal=True,
            key=f"feature3_text_space_{image_purpose}",
        )
        recommendation_ids = recommended_template_ids(menu_category)
        recommendation_labels = [
            next(
                template.label
                for template in FEATURE3_TEMPLATES
                if template.template_id == template_id
            )
            for template_id in recommendation_ids
        ]
        st.caption("추천 템플릿: " + " · ".join(recommendation_labels))
        submitted = st.button(
            "광고 이미지 설정 저장",
            type="primary",
            width="stretch",
        )

    preview_image = None
    image_key = None
    layout_key = f"{image_purpose}:{text_space}"
    prepare_previews_clicked = False
    with preview_column:
        st.subheader("3. 상품 미리보기")
        if uploaded_file is None:
            st.warning("상품 사진을 업로드하면 여기에 미리보기가 표시됩니다.")
        else:
            try:
                preview_image = load_uploaded_image(uploaded_file)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("이미지를 읽을 수 없습니다. 다른 JPG, PNG 또는 WEBP 파일을 선택해주세요.")
            else:
                uploaded_bytes = uploaded_file.getvalue()
                image_key = sha256(uploaded_bytes).hexdigest()
                st.session_state["shared_product_image"] = {
                    "image_key": image_key,
                    "name": uploaded_file.name,
                    "content_type": uploaded_file.type,
                    "bytes": uploaded_bytes,
                }
                st.image(
                    preview_image,
                    caption=(
                        f"{uploaded_file.name} · "
                        f"{preview_image.width}×{preview_image.height}px"
                    ),
                    width="stretch",
                )
                prepare_previews_clicked = st.button(
                    "템플릿 6종 미리보기 만들기",
                    type="secondary",
                    width="stretch",
                    help="상품만 분리한 뒤 문구 없는 6개 템플릿에 적용합니다.",
                )

    preview_state = st.session_state.get("feature3_template_previews") or {}
    if image_key and (
        preview_state.get("image_key") != image_key
        or preview_state.get("layout_key") != layout_key
    ):
        st.session_state.pop("feature3_template_previews", None)
        st.session_state.pop("feature3_selected_template", None)
        st.session_state.pop("feature3_request", None)
        st.session_state.pop("feature3_final_result", None)
        preview_state = {}
        selected_template = {}

    if prepare_previews_clicked and preview_image is not None and image_key is not None:
        try:
            with st.spinner("상품을 분리하고 템플릿 6종을 만들고 있습니다..."):
                bundle = get_background_removal_model()
                product_cutout = remove_background(preview_image, bundle)
                rendered_previews = {
                    template.template_id: feature3_image_to_png_bytes(
                        render_feature3_image(
                            product_cutout,
                            template_id=template.template_id,
                            image_purpose=image_purpose,
                            text_space=text_space,
                        )
                    )
                    for template in FEATURE3_TEMPLATES
                }
                cutout_png = image_to_png_bytes(product_cutout)
        except BackgroundRemovalError as exc:
            st.error(str(exc))
        except Exception:
            st.error("템플릿 미리보기를 만드는 중 예상하지 못한 오류가 발생했습니다.")
        else:
            st.session_state["feature3_template_previews"] = {
                "image_key": image_key,
                "layout_key": layout_key,
                "cutout_png": cutout_png,
                "previews": rendered_previews,
                "image_request": None,
            }
            st.rerun()

    preview_state = st.session_state.get("feature3_template_previews") or {}
    if image_key and preview_state.get("image_key") == image_key:
        st.subheader("4. 스타일 템플릿 선택")
        st.caption(
            "모든 미리보기는 같은 상품 사진을 사용하며 최종 출력과 같은 규칙으로 "
            "만듭니다. 추천 3종을 먼저 표시하지만 전체 6종 중에서 선택할 수 있습니다."
        )
        ordered_templates = sorted(
            FEATURE3_TEMPLATES,
            key=lambda item: (
                item.template_id not in recommendation_ids,
                recommendation_ids.index(item.template_id)
                if item.template_id in recommendation_ids
                else len(recommendation_ids),
            ),
        )
        preview_columns = st.columns(3, gap="medium")
        for index, template in enumerate(ordered_templates):
            with preview_columns[index % 3]:
                is_selected = selected_template.get("template_id") == template.template_id
                st.image(
                    preview_state["previews"][template.template_id],
                    caption=(
                        f"추천 · {template.label} · {template.description}"
                        if template.template_id in recommendation_ids
                        else f"{template.label} · {template.description}"
                    ),
                    width="stretch",
                )
                if st.button(
                    "선택됨" if is_selected else "이 템플릿 선택",
                    key=f"feature3_template_{template.template_id}",
                    type="primary" if is_selected else "secondary",
                    disabled=is_selected,
                    width="stretch",
                ):
                    st.session_state["feature3_selected_template"] = {
                        "image_key": image_key,
                        "layout_key": layout_key,
                        "template_id": template.template_id,
                    }
                    st.rerun()

        selected_template = st.session_state.get("feature3_selected_template") or {}
        if (
            selected_template.get("image_key") == image_key
            and selected_template.get("layout_key") == layout_key
        ):
            template = next(
                item
                for item in FEATURE3_TEMPLATES
                if item.template_id == selected_template["template_id"]
            )
            st.success(f"선택한 템플릿: {template.label}")

    if submitted:
        if uploaded_file is None or preview_image is None or image_key is None:
            st.error("정상적인 상품 사진을 반드시 업로드해주세요.")
        else:
            image_request = None
            template_id = (
                selected_template.get("template_id")
                if selected_template.get("image_key") == image_key
                and selected_template.get("layout_key") == layout_key
                else None
            )
            try:
                request = build_feature3_request(
                    image_filename=uploaded_file.name,
                    image_content_type=uploaded_file.type,
                    menu_category=menu_category,
                    image_purpose=image_purpose,
                    text_space=text_space,
                    image_request=image_request,
                    style_template_id=template_id,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["feature3_request"] = {
                    "image_key": image_key,
                    "layout_key": layout_key,
                    "request": request,
                }
                if preview_state.get("image_key") == image_key:
                    try:
                        with Image.open(BytesIO(preview_state["cutout_png"])) as source:
                            product_cutout = source.convert("RGBA")
                        refreshed_previews = {
                            template.template_id: feature3_image_to_png_bytes(
                                render_feature3_image(
                                    product_cutout,
                                    template_id=template.template_id,
                                    image_request=image_request,
                                    image_purpose=image_purpose,
                                    text_space=text_space,
                                )
                            )
                            for template in FEATURE3_TEMPLATES
                        }
                    except Exception:
                        st.error("저장한 요청 사항으로 템플릿 미리보기를 갱신하지 못했습니다.")
                    else:
                        st.session_state["feature3_template_previews"] = {
                            **preview_state,
                            "layout_key": layout_key,
                            "previews": refreshed_previews,
                            "image_request": image_request,
                        }
                        st.session_state.pop("feature3_final_result", None)
                        st.rerun()

    request_state = st.session_state.get("feature3_request")
    if (
        request_state
        and request_state.get("image_key") == image_key
        and request_state.get("layout_key") == layout_key
    ):
        st.success("광고 이미지 설정을 저장했습니다.")
        with st.expander("생성 요청 정보 확인", expanded=True):
            st.json(request_state["request"])
        preview_state = st.session_state.get("feature3_template_previews") or {}
        if preview_state.get("image_key") != image_key:
            st.warning("최종 이미지를 만들려면 먼저 템플릿 미리보기를 만들어주세요.")
            return

        request = request_state["request"]
        request_signature = sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        generate_clicked = st.button(
            "1080×1080 광고 이미지 생성",
            type="primary",
            width="stretch",
        )
        if generate_clicked:
            try:
                with Image.open(BytesIO(preview_state["cutout_png"])) as source:
                    product_cutout = source.convert("RGBA")
                final_image = render_feature3_image(
                    product_cutout,
                    template_id=request["style_template_id"],
                    image_request=request["image_request"],
                    image_purpose=request["image_purpose"],
                    text_space=request["text_space"],
                )
                final_png = feature3_image_to_png_bytes(final_image)
            except Exception:
                st.error("광고 이미지를 만드는 중 오류가 발생했습니다.")
            else:
                selected_preview = None
                if request["style_template_id"]:
                    selected_preview = preview_state["previews"].get(
                        request["style_template_id"]
                    )
                st.session_state["feature3_final_result"] = {
                    "image_key": image_key,
                    "request_signature": request_signature,
                    "png_bytes": final_png,
                    "file_name": f"{Path(uploaded_file.name).stem}_ad_1080.png",
                    "preview_matches_final": (
                        selected_preview == final_png
                        if selected_preview is not None
                        else None
                    ),
                }

        final_result = st.session_state.get("feature3_final_result") or {}
        if (
            final_result.get("image_key") == image_key
            and final_result.get("request_signature") == request_signature
        ):
            st.image(
                final_result["png_bytes"],
                caption="문구 없는 1080×1080 광고 이미지",
                width="stretch",
            )
            if final_result["preview_matches_final"] is True:
                st.success("선택한 템플릿 미리보기와 최종 결과가 일치합니다.")
            st.download_button(
                "광고 이미지 PNG 다운로드",
                data=final_result["png_bytes"],
                file_name=f"{Path(uploaded_file.name).stem}_ad_1080.png",
                mime="image/png",
                width="stretch",
            )
            st.button(
                "다음: 광고 문구 만들기",
                type="secondary",
                width="stretch",
                on_click=open_copy_menu,
            )

    navigation_result = st.session_state.get("feature3_final_result") or {}
    has_current_image_result = bool(
        image_key and navigation_result.get("image_key") == image_key
    )
    if not has_current_image_result:
        st.divider()
        st.button(
            "이미지 생성 건너뛰고 문구 만들기",
            type="secondary",
            width="stretch",
            on_click=skip_image_step,
        )


def collect_copy_options(
    copy_state: dict | None,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, str]]:
    available_copies: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    labels: dict[str, str] = {}
    if not copy_state:
        return available_copies, metadata, labels
    for record in copy_state.get("records", []):
        for index, copy in enumerate(record.get("copies", []), start=1):
            option_key = f"{record['model_id']}::{index}"
            available_copies[option_key] = copy
            metadata[option_key] = {
                "copy_index": index,
                "model_id": record["model_id"],
                "model_label": record["model_label"],
            }
            body_label = f" / {copy['body']}" if copy.get("body") else ""
            labels[option_key] = f"문구 후보 {index} · {copy['headline']}{body_label}"
    return available_copies, metadata, labels


def render_workflow_progress(current_step: str) -> None:
    steps = (
        ("image", "1. 광고 이미지"),
        ("copy", "2. 광고 문구"),
        ("results", "3. 결과 다운로드"),
    )
    columns = st.columns(len(steps))
    for column, (step_id, label) in zip(columns, steps):
        with column:
            if step_id == current_step:
                st.markdown(f"**▶ {label}**")
            else:
                st.caption(label)


def render_results_workflow() -> None:
    st.subheader("완성된 결과")
    image_result = st.session_state.get("feature3_final_result") or {}
    copy_state = st.session_state.get("copy_results")
    available_copies, copy_metadata, _ = collect_copy_options(copy_state)
    selected_copy = None
    selected_metadata = None

    if not image_result and not available_copies:
        st.warning("아직 생성된 이미지나 광고 문구가 없습니다.")

    if image_result:
        st.markdown("### 광고 이미지")
        st.image(
            image_result["png_bytes"],
            caption="문구 없는 1080×1080 광고 이미지",
            width="stretch",
        )
        st.download_button(
            "광고 이미지 PNG 다운로드",
            data=image_result["png_bytes"],
            file_name=image_result.get("file_name", "ad_image_1080.png"),
            mime="image/png",
            width="stretch",
        )

    if available_copies:
        st.markdown("### 광고 문구")
        saved_copy = st.session_state.get("selected_copy_result") or {}
        copy_keys = list(available_copies)
        selected_copy_key = (
            saved_copy.get("key")
            if saved_copy.get("key") in available_copies
            else copy_keys[0]
        )
        selected_copy = available_copies[selected_copy_key]
        selected_metadata = copy_metadata[selected_copy_key]
        with st.container(border=True):
            st.caption("선택한 광고 문구")
            st.markdown(f"### {selected_copy['headline']}")
            if selected_copy.get("body"):
                st.write(selected_copy["body"])
            else:
                st.caption("보조 문구 없이 제목만 사용")
        copy_text = selected_copy["headline"]
        if selected_copy.get("body"):
            copy_text += f"\n{selected_copy['body']}"
        copy_json = json.dumps(
            {
                "copy": selected_copy,
                "copy_metadata": selected_metadata,
                "text_position": copy_state.get("text_position", "auto")
                if copy_state
                else "auto",
                "ad_input": copy_state.get("ad_input") if copy_state else None,
            },
            ensure_ascii=False,
            indent=2,
        )
        text_column, json_column = st.columns(2)
        with text_column:
            st.download_button(
                "광고 문구 TXT 다운로드",
                data=copy_text.encode("utf-8"),
                file_name="ad_copy.txt",
                mime="text/plain; charset=utf-8",
                width="stretch",
            )
        with json_column:
            st.download_button(
                "광고 문구 JSON 다운로드",
                data=copy_json.encode("utf-8"),
                file_name="ad_copy.json",
                mime="application/json",
                width="stretch",
            )

    if image_result and selected_copy is not None and copy_state:
        request_state = st.session_state.get("feature3_request") or {}
        preview_state = st.session_state.get("feature3_template_previews") or {}
        request = request_state.get("request") or {}
        template_id = request.get("style_template_id")
        cutout_png = preview_state.get("cutout_png")
        if template_id in TEMPLATE_MAP and cutout_png:
            combined_text_space = copy_state.get("text_position", "auto")
            try:
                with Image.open(BytesIO(cutout_png)) as source:
                    combined_cutout = source.convert("RGBA")
                    combined_text_space = resolve_auto_text_space(
                        combined_cutout,
                        template_id=template_id,
                        image_purpose="copy_later",
                        text_space=combined_text_space,
                    )
                    combined_base = render_feature3_image(
                        combined_cutout,
                        template_id=template_id,
                        image_purpose="copy_later",
                        text_space=combined_text_space,
                    )
                safe_areas = banner_text_safe_areas(
                    template_id,
                    image_purpose="copy_later",
                    text_space=combined_text_space,
                )
                if len(safe_areas) != 1:
                    raise ValueError("문구 안전 영역을 하나로 결정할 수 없습니다.")
                ad_input = copy_state["ad_input"]
                combined = render_combined_banner(
                    combined_base,
                    template_id=template_id,
                    text_safe_area=safe_areas[0],
                    copy=selected_copy,
                    store_name=ad_input["store"]["name"],
                    product_name=ad_input["product"]["name"],
                    base_price=ad_input["product"]["base_price"],
                    sale_price=ad_input["product"]["sale_price"],
                    promotion=ad_input["campaign"]["promotion"],
                    promotion_copy_mode=ad_input["campaign"].get(
                        "promotion_copy_mode",
                        BADGE_ONLY,
                    ),
                    call_to_action=ad_input["campaign"]["call_to_action"],
                )
                combined_png = feature3_image_to_png_bytes(combined.image)
            except Exception as exc:
                st.warning(f"최종 광고 배너를 합성하지 못했습니다: {exc}")
            else:
                st.markdown("### 이미지 + 문구 최종 광고 배너")
                st.image(
                    combined_png,
                    caption="1080×1080 최종 광고 배너",
                    width="stretch",
                )
                st.download_button(
                    "최종 광고 배너 PNG 다운로드",
                    data=combined_png,
                    file_name="final_ad_banner_1080.png",
                    mime="image/png",
                    type="primary",
                    width="stretch",
                )
                st.caption(
                    "이미지 단독용 배치였다면 같은 템플릿의 문구 추가용 배치로 "
                    "자동 전환하여 제품과 글자가 겹치지 않게 합성했습니다."
                )
        else:
            st.info(
                "최종 합성에는 고정 스타일 템플릿과 상품 컷아웃이 필요합니다. "
                "이미지·문구는 각각 다운로드할 수 있습니다."
            )

    back_column, restart_column = st.columns(2)
    with back_column:
        st.button(
            "광고 문구로 돌아가기",
            width="stretch",
            on_click=set_workflow_step,
            args=("copy",),
        )
    with restart_column:
        st.button(
            MENU_PRODUCT_PRESERVE,
            width="stretch",
            on_click=set_main_menu,
            args=(MENU_PRODUCT_PRESERVE,),
        )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="☕",
        layout="wide",
    )

    apply_cafe_theme()
    render_app_header()

    with st.sidebar:
        st.header("☕ 광고 제작")
        selected_menu = st.radio(
            "메뉴",
            options=MAIN_MENU_OPTIONS,
            key="main_menu",
        )
        st.caption(MAIN_MENU_DESCRIPTIONS[selected_menu])

    previous_menu = st.session_state.get("_active_main_menu")
    if previous_menu != selected_menu:
        st.session_state["_active_main_menu"] = selected_menu
        if selected_menu == MENU_COPY:
            st.session_state["workflow_step"] = "copy"

    st.divider()

    if selected_menu == MENU_REFERENCE_EDIT:
        st.header(MENU_REFERENCE_EDIT)
        st.caption(MAIN_MENU_DESCRIPTIONS[MENU_REFERENCE_EDIT])
        render_reference_edit_workflow()
        return

    if selected_menu == MENU_PRODUCT_PRESERVE:
        st.header(MENU_PRODUCT_PRESERVE)
        st.caption(MAIN_MENU_DESCRIPTIONS[MENU_PRODUCT_PRESERVE])
        render_feature3_workflow()
        return

    if selected_menu == MENU_SKETCH_AD:
        st.header(MENU_SKETCH_AD)
        st.caption(MAIN_MENU_DESCRIPTIONS[MENU_SKETCH_AD])
        render_sketch_ad_workflow()
        return

    workflow_step = st.session_state.get("workflow_step", "copy")
    if workflow_step not in {"copy", "results"}:
        workflow_step = "copy"
        st.session_state["workflow_step"] = workflow_step
    if workflow_step == "results":
        render_results_workflow()
        return

    st.header(MENU_COPY)
    st.caption(MAIN_MENU_DESCRIPTIONS[MENU_COPY])
    st.caption(
        "업로드한 이미지는 상품 분석을 위해 Google Gemini API로 전송되며, "
        "입력 정보와 검증된 시각 사실은 광고 문구 생성을 위해 OpenAI API로 "
        "전송됩니다. 원본 이미지는 OpenAI 문구 생성 단계에 전달하지 않습니다."
    )

    input_column, preview_column = st.columns([1, 1], gap="large")
    preview_image = None
    current_image_key = None
    cutout_record = None

    with input_column:
        st.subheader("1. 상품 정보 입력")

        manual_uploaded_file = st.file_uploader(
            "상품 이미지 *",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG 또는 WEBP 파일 한 장을 업로드하세요.",
            key="copy_product_image",
        )
        shared_image = st.session_state.get("shared_product_image") or {}
        if manual_uploaded_file is not None:
            uploaded_file = manual_uploaded_file
        elif shared_image.get("bytes"):
            uploaded_file = SessionUploadedImage(
                name=shared_image["name"],
                content_type=shared_image.get("content_type") or "image/png",
                data=shared_image["bytes"],
            )
            st.success("광고 이미지 단계에서 업로드한 제품 사진을 사용합니다.")
        else:
            uploaded_file = None

        product_name = st.text_input(
            "메뉴명 *",
            placeholder="예: 딸기 크림 라떼",
            max_chars=40,
        )
        features = st.text_area(
            "강조하고 싶은 장점 *",
            placeholder="예: 생딸기를 사용해 만든 메뉴",
            help=(
                "제목에서 가장 먼저 전달할 실제 장점을 적어주세요. "
                "입력하지 않은 맛·재료·효능은 AI가 새로 만들지 않습니다."
            ),
            max_chars=120,
            height=90,
        )
        copy_text_position = st.segmented_control(
            "배너 문구 위치",
            options=list(COPY_TEXT_POSITION_OPTIONS),
            format_func=FEATURE3_TEXT_SPACE_LABELS.get,
            default="auto",
            selection_mode="single",
            help=(
                "최종 광고 배너에서 제목과 보조 문구를 배치할 위치입니다. "
                "자동은 상품을 가리지 않는 영역을 선택합니다."
            ),
            key="copy_text_position",
        ) or "auto"

        with st.expander("문구 세부 설정", expanded=False):
            selected_tone = st.selectbox(
                "문구 스타일·톤",
                options=COPY_TONE_OPTIONS,
                format_func=COPY_TONE_LABELS.get,
                help=(
                    "자동 추천을 선택하면 상품 정보에 맞는 기본 톤을 사용합니다. "
                    "스타일·톤은 새로운 상품 사실의 근거로 사용하지 않습니다."
                ),
            )
            copy_request = st.text_area(
                "추가 요청사항 (선택)",
                placeholder=(
                    "예: 직장인을 대상으로 퇴근 후 간식 상황을 강조하고, 제목은 "
                    "18자 이내로 작성해주세요."
                ),
                help=(
                    "타깃 고객, 이용 상황, 글자 수, 반드시 넣거나 제외할 표현 중 "
                    "필요한 내용만 입력하세요."
                ),
                max_chars=240,
                height=100,
            )
            generation_mode = st.radio(
                "문구 구성",
                options=["headline_and_subcopy", "headline_only"],
                format_func=lambda value: (
                    "제목 + 보조 문구 (권장)"
                    if value == "headline_and_subcopy"
                    else "제목만"
                ),
                help=(
                    "두 번째 근거가 부족하면 선택과 관계없이 제목만으로 전환될 수 있습니다."
                ),
                horizontal=True,
            )
        with st.expander("배너에 표시할 정보 (선택)", expanded=False):
            st.caption("입력한 내용은 AI가 바꾸지 않고 배너에 그대로 표시합니다.")
            store_name = st.text_input(
                "매장명",
                placeholder="예: 오늘의 작은 카페",
                max_chars=30,
            )
            base_price = st.text_input(
                "기본 가격",
                placeholder="예: 6,500원",
                max_chars=20,
            )
            sale_price = st.text_input(
                "할인가 (선택)",
                placeholder="예: 5,500원",
                max_chars=20,
                help="할인가가 있을 때만 입력하세요. 앱이 가격을 자동 계산하지 않습니다.",
            )
            promotion = st.text_input(
                "행사 정보",
                placeholder="예: 이번 주말 10% 할인",
                max_chars=60,
            )
            promotion_copy_mode = st.radio(
                "프로모션 문구 사용 방식",
                options=[BADGE_ONLY, HEADLINE_ALLOWED],
                format_func=lambda value: (
                    "일반 광고 · 배지에만 표시 (권장)"
                    if value == BADGE_ONLY
                    else "프로모션 중심 광고 · 제목 사용 허용"
                ),
                help=(
                    "일반 광고에서는 행사 정보를 제목 후보에서 제외하고 배지에 원문으로 "
                    "표시합니다."
                ),
            )
            promotion_request_conflict = (
                promotion_copy_mode == BADGE_ONLY
                and bool(promotion.strip())
                and headline_mentions_promotion(copy_request)
            )
            if promotion_request_conflict:
                st.warning(
                    "행사 정보를 제목에 사용하려면 ‘프로모션 중심 광고’를 선택해주세요."
                )
            call_to_action = st.text_input(
                "방문·주문 안내 (선택)",
                placeholder="예: 매장에서 메뉴를 만나보세요.",
                max_chars=40,
                help="실제로 가능한 방문 또는 주문 방법만 입력하세요.",
            )

        can_generate_copy = bool(
            uploaded_file is not None and product_name.strip() and features.strip()
        )
        submitted = st.button(
            "광고 문구 만들기",
            type="primary",
            width="stretch",
            disabled=not can_generate_copy,
        )
        if not can_generate_copy:
            st.caption("상품 이미지, 메뉴명과 강조할 장점을 입력하면 문구를 만들 수 있습니다.")

    with preview_column:
        st.subheader("2. 사진 미리보기 및 배경 제거")

        if uploaded_file is None:
            st.warning("왼쪽에서 상품 사진을 업로드하면 여기에 미리보기가 표시됩니다.")
        else:
            try:
                preview_image = load_uploaded_image(uploaded_file)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("이미지를 읽을 수 없습니다. 다른 JPG, PNG 또는 WEBP 파일을 선택해주세요.")
            else:
                image_key = sha256(uploaded_file.getvalue()).hexdigest()
                current_image_key = image_key
                cutout_record = st.session_state.get("cutout_result")
                if cutout_record and cutout_record.get("image_key") != image_key:
                    st.session_state.pop("cutout_result", None)
                    cutout_record = None
                feature3_previews = st.session_state.get("feature3_template_previews") or {}
                if not cutout_record and feature3_previews.get("image_key") == image_key:
                    cutout_record = {
                        "image_key": image_key,
                        "png_bytes": feature3_previews["cutout_png"],
                        "device_label": "이전 단계 결과 재사용",
                    }
                    st.session_state["cutout_result"] = cutout_record

                if cutout_record:
                    original_column, result_column = st.columns(2)
                    with original_column:
                        st.image(preview_image, caption="원본", width="stretch")
                    with result_column:
                        st.image(
                            cutout_record["png_bytes"],
                            caption="배경 제거 결과",
                            width="stretch",
                        )
                else:
                    st.image(
                        preview_image,
                        caption=(
                            f"{uploaded_file.name} · "
                            f"{preview_image.width}×{preview_image.height}px"
                        ),
                        width="stretch",
                    )

                st.caption(
                    "BiRefNet은 무료 오픈소스 모델입니다. 첫 실행 때 약 445MB 모델을 "
                    "한 번 내려받으며, CPU에서는 처리에 시간이 걸릴 수 있습니다."
                )
                remove_clicked = st.button(
                    "배경 제거 다시 실행" if cutout_record else "배경 제거 실행",
                    type="secondary",
                    width="stretch",
                )
                if remove_clicked:
                    try:
                        with st.spinner("상품 영역을 찾고 투명 배경을 만들고 있습니다..."):
                            bundle = get_background_removal_model()
                            cutout = remove_background(preview_image, bundle)
                            cutout_png = image_to_png_bytes(cutout)
                    except BackgroundRemovalError as exc:
                        st.error(str(exc))
                    except Exception:
                        st.error("배경 제거 중 예상하지 못한 오류가 발생했습니다.")
                    else:
                        st.session_state["cutout_result"] = {
                            "image_key": image_key,
                            "png_bytes": cutout_png,
                            "device_label": bundle.device_label,
                        }
                        st.rerun()

                if cutout_record:
                    st.success(
                        f"배경 제거 완료 · {cutout_record['device_label']} 처리 · "
                        "투명 PNG"
                    )
                    st.download_button(
                        "투명 PNG 다운로드",
                        data=cutout_record["png_bytes"],
                        file_name=f"{Path(uploaded_file.name).stem}_cutout.png",
                        mime="image/png",
                        width="stretch",
                    )

        with st.expander("입력 내용 확인", expanded=False):
            st.write(
                {
                    "매장명": store_name or "미입력",
                    "메뉴명": product_name or "미입력",
                    "기본 가격": base_price or "미입력",
                    "할인가": sale_price or "미입력",
                    "강조하고 싶은 장점": features or "미입력",
                    "행사 정보": promotion or "미입력",
                    "프로모션 문구 사용": (
                        "배지 전용"
                        if promotion_copy_mode == BADGE_ONLY
                        else "프로모션 제목 허용"
                    ),
                    "방문·주문 안내": call_to_action or "미입력",
                    "추가 요청사항": copy_request or "미입력",
                    "문구 스타일·톤": COPY_TONE_LABELS[selected_tone],
                    "문구 구성": (
                        "제목 + 보조 문구"
                        if generation_mode == "headline_and_subcopy"
                        else "제목만"
                    ),
                    "배너 문구 위치": FEATURE3_TEXT_SPACE_LABELS[
                        copy_text_position
                    ],
                }
            )

    copy_state = st.session_state.get("copy_results")
    if copy_state and copy_state.get("image_key") != current_image_key:
        st.session_state.pop("copy_results", None)
        st.session_state.pop("banner_result", None)
        copy_state = None

    if submitted:
        if (
            uploaded_file is None
            or preview_image is None
            or not product_name.strip()
            or not features.strip()
        ):
            st.error(
                "정상적인 상품 이미지, 메뉴명, 강조하고 싶은 장점을 "
                "반드시 입력해주세요."
            )
        elif sale_price.strip() and not base_price.strip():
            st.error("할인가를 입력하려면 기본 가격도 입력해주세요.")
        else:
            resolved_tone = (
                DEFAULT_COPY_TONE if selected_tone == "auto" else selected_tone
            )
            ad_input = build_ad_input(
                image_filename=uploaded_file.name,
                image_content_type=uploaded_file.type,
                store_name=store_name,
                product_name=product_name,
                base_price=base_price,
                sale_price=sale_price,
                features=features,
                promotion=promotion,
                tone=resolved_tone,
                copy_request=copy_request,
                call_to_action=call_to_action,
            )
            ad_input["campaign"]["focus_feature"] = features.strip()
            ad_input["campaign"]["generation_mode"] = generation_mode
            ad_input["campaign"]["promotion_copy_mode"] = promotion_copy_mode
            records: list[dict] = []
            visual_api_key = resolve_api_key(VISUAL_MODEL_CONFIG)
            copy_api_key = resolve_api_key(COPY_MODEL_CONFIG)
            missing_keys = [
                config.api_key_env
                for config, api_key in (
                    (VISUAL_MODEL_CONFIG, visual_api_key),
                    (COPY_MODEL_CONFIG, copy_api_key),
                )
                if not api_key
            ]
            prepared_input = ad_input
            if missing_keys:
                records.append(
                    {
                        "model_id": COPY_MODEL_CONFIG.model_id,
                        "model_label": COPY_MODEL_CONFIG.label,
                        "error": f"{', '.join(missing_keys)}가 설정되지 않았습니다.",
                    }
                )
            else:
                started_at = perf_counter()
                try:
                    with st.spinner(
                        "Gemini로 이미지를 확인하고 GPT로 광고 문구를 생성하고 있습니다..."
                    ):
                        pipeline_result = generate_app_copy(
                            ad_input,
                            image_bytes=uploaded_file.getvalue(),
                            image_mime_type=uploaded_file.type,
                            gemini_api_key=visual_api_key,
                            openai_api_key=copy_api_key,
                        )
                        prepared_input = pipeline_result.enriched_input
                        visual_result = pipeline_result.visual_result
                        result = pipeline_result.copy_result
                except CopyGenerationError as exc:
                    records.append(
                        {
                            "model_id": COPY_MODEL_CONFIG.model_id,
                            "model_label": COPY_MODEL_CONFIG.label,
                            "error": str(exc),
                        }
                    )
                except Exception:
                    records.append(
                        {
                            "model_id": COPY_MODEL_CONFIG.model_id,
                            "model_label": COPY_MODEL_CONFIG.label,
                            "error": (
                                "시각 정보 추출 또는 문구 생성 중 예상하지 못한 오류가 "
                                "발생했습니다."
                            ),
                        }
                    )
                else:
                    records.append(
                        {
                            "model_id": COPY_MODEL_CONFIG.model_id,
                            "model_label": COPY_MODEL_CONFIG.label,
                            "pipeline_version": PIPELINE_VERSION,
                            "prompt_version": result.prompt_version,
                            "visual_model_id": VISUAL_MODEL_CONFIG.model_id,
                            "visual_model_label": VISUAL_MODEL_CONFIG.label,
                            "visual_prompt_version": visual_result.prompt_version,
                            "visual_status": visual_result.status,
                            "visual_notice": visual_result.notice,
                            "visual_facts": [
                                dict(fact) for fact in visual_result.verified_facts
                            ],
                            "rejected_visual_observations": [
                                dict(fact)
                                for fact in visual_result.rejected_observations
                            ],
                            "excluded_visual_inferences": list(
                                visual_result.excluded_inferences
                            ),
                            "status": result.status,
                            "attempts": result.attempts,
                            "generation_mode": result.generation_mode,
                            "notice": result.notice,
                            "validation_issues": list(result.validation_issues),
                            "attempt_history": list(result.attempt_history),
                            "copies": result.copies,
                            "elapsed_seconds": perf_counter() - started_at,
                        }
                    )

            copy_state = {
                "image_key": current_image_key,
                "ad_input": prepared_input,
                "records": records,
                "text_position": copy_text_position,
            }
            st.session_state["copy_results"] = copy_state
            st.session_state.pop("selected_copy_result", None)
            st.session_state.pop("banner_result", None)

    if copy_state:
        st.success("검증된 시각 정보와 광고 문구 후보를 배너 제작용으로 준비했습니다.")
        with st.expander("광고 생성 요청 JSON 확인"):
            st.json(copy_state["ad_input"])

        st.subheader("3. 생성된 문구 확인")
        for record in copy_state["records"]:
            render_copy_record(record)

        available_copies: dict[str, dict] = {}
        available_copy_metadata: dict[str, dict] = {}
        copy_labels: dict[str, str] = {}
        for record in copy_state["records"]:
            for index, copy in enumerate(record.get("copies", []), start=1):
                option_key = f"{record['model_id']}::{index}"
                available_copies[option_key] = copy
                available_copy_metadata[option_key] = {
                    "copy_index": index,
                    "model_id": record["model_id"],
                    "model_label": record["model_label"],
                }
                body_label = f" / {copy['body']}" if copy.get("body") else ""
                copy_labels[option_key] = (
                    f"문구 후보 {index} · {copy['headline']}{body_label}"
                )

        if available_copies:
            st.subheader("4. 최종 문구 선택")
            saved_copy = st.session_state.get("selected_copy_result") or {}
            saved_copy_key = saved_copy.get("key")
            option_columns = st.columns(min(2, len(available_copies)), gap="large")
            for option_index, (option_key, copy) in enumerate(
                available_copies.items(), start=1
            ):
                with option_columns[(option_index - 1) % len(option_columns)]:
                    with st.container(border=True):
                        st.caption(f"문구 후보 {option_index}")
                        st.markdown(f"### {copy['headline']}")
                        if copy.get("body"):
                            st.write(copy["body"])
                        else:
                            st.caption("보조 문구 없이 제목만 사용")
                        is_selected = saved_copy_key == option_key
                        if st.button(
                            "선택됨" if is_selected else "이 문구 선택",
                            key=f"select_copy_card_{option_key}",
                            type="primary" if is_selected else "secondary",
                            disabled=is_selected,
                            width="stretch",
                        ):
                            st.session_state["selected_copy_result"] = {
                                "key": option_key,
                                "copy": copy,
                                "metadata": available_copy_metadata[option_key],
                            }
                            st.rerun()

            selected_copy_state = st.session_state.get("selected_copy_result") or {}
            if selected_copy_state.get("key") in available_copies:
                st.success(
                    f"선택한 문구: {selected_copy_state['copy']['headline']}"
                )
                st.button(
                    "선택한 문구로 결과 보기",
                    type="primary",
                    width="stretch",
                    on_click=set_workflow_step,
                    args=("results",),
                )
            else:
                st.caption("두 후보 중 사용할 문구를 선택해주세요.")
            return

            st.subheader("4. 1080×1080 SNS 배너 만들기")

            image_source_options = ["original"]
            if cutout_record:
                image_source_options.insert(0, "cutout")
            image_source_labels = {
                "cutout": "배경 제거 결과",
                "original": "원본 사진",
            }
            selected_image_source = st.radio(
                "배너에 사용할 상품 이미지",
                options=image_source_options,
                format_func=image_source_labels.get,
                horizontal=True,
                help=(
                    "배경 제거 결과가 제품을 잘랐거나 테두리가 부자연스러우면 "
                    "원본 사진을 선택할 수 있습니다."
                ),
            )
            if selected_image_source == "original":
                st.caption("원본 사진을 비율 변경 없이 배너의 제품 영역 안에 배치합니다.")

            selection_column, template_column = st.columns(2, gap="large")
            with selection_column:
                selected_copy_key = st.selectbox(
                    "배너에 사용할 광고 문구",
                    options=list(available_copies),
                    format_func=copy_labels.get,
                )
            with template_column:
                template_options = [
                    template.template_id for template in BANNER_TEMPLATES
                ]
                recommended_template_id = (
                    "promo" if promotion or sale_price else "cream"
                )
                brand_style_locked = st.checkbox(
                    "브랜드 스타일 고정",
                    key="brand_style_lock_enabled",
                    help=(
                        "선택한 템플릿의 대표 색상과 매장명 위치를 다음 배너에도 "
                        "그대로 사용합니다."
                    ),
                )
                saved_brand_style = st.session_state.get("brand_style_preset") or {}
                default_template_id = recommended_template_id
                if brand_style_locked and saved_brand_style.get("template_id") in template_options:
                    default_template_id = saved_brand_style["template_id"]
                selected_template_id = st.selectbox(
                    "배너 템플릿",
                    options=template_options,
                    index=template_options.index(default_template_id),
                    format_func=lambda value: (
                        f"{TEMPLATE_MAP[value].label} · "
                        f"{TEMPLATE_MAP[value].description}"
                    ),
                    help=(
                        "행사 정보가 있으면 프로모션 템플릿을 추천하고, 없으면 "
                        "기본 템플릿을 사용합니다. 원하는 디자인으로 바꿀 수 있습니다."
                    ),
                    key="selected_banner_template",
                )
                if brand_style_locked:
                    palette, store_position = BRAND_STYLE_DETAILS[selected_template_id]
                    st.session_state["brand_style_preset"] = {
                        "template_id": selected_template_id,
                        "palette": palette,
                        "store_position": store_position,
                    }
                    st.caption(
                        f"고정됨 · {palette} · 매장명 {store_position}"
                    )

            selection_signature = (
                f"{current_image_key}::{selected_copy_key}::{selected_template_id}::"
                f"{selected_image_source}"
            )
            generate_banner_clicked = st.button(
                "선택한 문구로 배너 생성",
                type="primary",
                width="stretch",
            )
            banner_record = st.session_state.get("banner_result")

            if generate_banner_clicked:
                ad_input = copy_state["ad_input"]
                try:
                    with st.spinner("상품 이미지와 선택한 문구를 배치하고 있습니다..."):
                        selected_copy = available_copies[selected_copy_key]
                        banner_copy = dict(selected_copy)
                        if selected_image_source == "cutout" and cutout_record:
                            product_source = Image.open(BytesIO(cutout_record["png_bytes"]))
                        else:
                            product_source = preview_image.convert("RGBA")
                        with product_source as product_image:
                            banner_promotion = promotion_badge_text(
                                ad_input["campaign"]["promotion"],
                                headline=banner_copy["headline"],
                                promotion_copy_mode=ad_input["campaign"].get(
                                    "promotion_copy_mode",
                                    BADGE_ONLY,
                                ),
                            )
                            banner = generate_banner(
                                product_image.copy(),
                                banner_copy,
                                store_name=ad_input["store"]["name"],
                                product_name=ad_input["product"]["name"],
                                base_price=ad_input["product"]["base_price"],
                                sale_price=ad_input["product"]["sale_price"],
                                promotion=banner_promotion,
                                call_to_action=ad_input["campaign"]["call_to_action"],
                                template_id=selected_template_id,
                            )
                        banner_png = banner_to_png_bytes(banner)
                        selected_metadata = available_copy_metadata[selected_copy_key]
                        export_package = build_export_package(
                            banner_png=banner_png,
                            copy=banner_copy,
                            copy_index=selected_metadata["copy_index"],
                            model_id=selected_metadata["model_id"],
                            model_label=selected_metadata["model_label"],
                            template_id=selected_template_id,
                            template_label=TEMPLATE_MAP[selected_template_id].label,
                            ad_input=ad_input,
                        )
                except Exception:
                    st.error("배너 생성 중 오류가 발생했습니다. 다른 템플릿으로 시도해주세요.")
                else:
                    banner_record = {
                        "signature": selection_signature,
                        "png_bytes": banner_png,
                        "template_label": TEMPLATE_MAP[selected_template_id].label,
                        "banner_filename": export_package.banner_filename,
                        "copy_filename": export_package.copy_filename,
                        "copy_text": export_package.copy_text,
                        "metadata_filename": export_package.metadata_filename,
                        "metadata_json": export_package.metadata_json,
                        "zip_filename": export_package.zip_filename,
                        "zip_bytes": export_package.zip_bytes,
                    }
                    st.session_state["banner_result"] = banner_record

            if banner_record and banner_record.get("signature") == selection_signature:
                st.image(
                    banner_record["png_bytes"],
                    caption=(
                        f"{banner_record['template_label']} · 1080×1080px 미리보기"
                    ),
                    width="stretch",
                )
                st.success("SNS 배너 생성 완료 · PNG, 문구, 생성 정보 다운로드 가능")
                st.caption("선택한 제목과 보조 문구를 검증된 결과 그대로 사용했습니다.")
                first_download_column, second_download_column = st.columns(2)
                with first_download_column:
                    st.download_button(
                        "배너 PNG 다운로드",
                        data=banner_record["png_bytes"],
                        file_name=banner_record["banner_filename"],
                        mime="image/png",
                        width="stretch",
                    )
                with second_download_column:
                    st.download_button(
                        "광고 문구 TXT 다운로드",
                        data=banner_record["copy_text"].encode("utf-8"),
                        file_name=banner_record["copy_filename"],
                        mime="text/plain; charset=utf-8",
                        width="stretch",
                    )

                metadata_column, package_column = st.columns(2)
                with metadata_column:
                    st.download_button(
                        "생성 정보 JSON 다운로드",
                        data=banner_record["metadata_json"].encode("utf-8"),
                        file_name=banner_record["metadata_filename"],
                        mime="application/json",
                        width="stretch",
                    )
                with package_column:
                    st.download_button(
                        "전체 결과 ZIP 다운로드",
                        data=banner_record["zip_bytes"],
                        file_name=banner_record["zip_filename"],
                        mime="application/zip",
                        type="primary",
                        width="stretch",
                    )
                with st.expander("ZIP에 포함되는 생성 정보 확인"):
                    st.code(banner_record["metadata_json"], language="json")

if __name__ == "__main__":
    main()
