# 가게 한컷

카페·디저트 소상공인을 위한 SNS 광고 문구·이미지 제작 서비스입니다.

## 프로젝트 개요

소규모 카페와 베이커리 운영자는 상품 사진을 가지고 있어도 광고 문구를 작성하고
SNS 이미지를 디자인하는 데 많은 시간과 비용을 사용합니다.

**가게 한컷**은 생성형 AI를 활용해 상품 사진 또는 간단한 스케치와 요청사항만으로
광고 문구와 광고용 이미지를 만들 수 있도록 돕습니다. 전문 디자인 도구가 익숙하지
않은 사용자도 하나의 Streamlit 앱에서 결과를 확인하고 내려받을 수 있습니다.

이번 프로젝트는 과제에서 제시한 **소상공인을 위한 광고 콘텐츠 생성 서비스**라는
주제를 카페·디저트 매장의 SNS 광고 제작 문제로 구체화했습니다. 구현 범위는 광고
문구, 상품 중심 광고 이미지와 1080×1080 SNS 배너이며 상세 페이지와 메뉴판 제작은
포함하지 않습니다.

## 최종 보고서

프로젝트 진행 과정, 기능별 결과 이미지, 성과, 한계점과 개선 계획을 정리한 최종
보고서입니다.

**[최종 보고서 PDF 다운로드](docs/report/store_onecut_final_report.pdf)**

## 주요 기능

| 기능 | 설명 |
|---|---|
| 광고 문구 추천 | 상품 이미지와 상품 특징을 바탕으로 제목과 보조 문구 후보를 생성합니다. |
| 상품 사진 꾸미기 | 입력한 상품은 유지하고 배경과 광고 연출을 구성합니다. |
| 사진 새롭게 바꾸기 | 입력 사진을 참고해 상품·배경·분위기·구도를 요청대로 변경합니다. |
| 스케치를 광고로 완성하기 | 스케치의 배치를 참고해 새로운 광고 이미지를 생성합니다. |

## 사용 기술

- Python, Streamlit
- Pillow, OpenCV, BiRefNet
- Gemini 3.5 Flash-Lite
- GPT-6 Luna, `gpt-image-2.5-sunburst`
- `black-forest-labs/FLUX.2-klein-4B`
- pytest

## 프로젝트 구조

GitHub 공개본은 최종 앱 실행 코드와 기능별 최종 실험 자료를 다음 구조로 정리합니다.

```bash
project/
├── app.py                         # 네 가지 기능을 제공하는 Streamlit 앱
├── src/
│   ├── features/
│   │   ├── ad_copy/
│   │   │   ├── pipeline.py        # 이미지 분석부터 문구 후보 생성까지의 최종 흐름
│   │   │   ├── workflow.py        # 확정된 광고 문구 생성·검증 워크플로우
│   │   │   ├── generator.py       # Gemini·GPT 모델 호출
│   │   │   ├── visual_analysis.py # 상품 이미지에서 확인 가능한 정보 추출
│   │   │   ├── prompt.py          # 확정된 최종 광고 문구 프롬프트
│   │   │   ├── validation.py      # 근거·금지어·길이·후보 유사도 검증
│   │   │   ├── recovery.py        # 검증에 실패한 후보만 부분 재생성
│   │   │   ├── promotion.py       # 가격·할인·행사 정보 처리 규칙
│   │   │   ├── models.py          # 문구 기능 입력·출력 자료형
│   │   │   └── internal/          # 최종 워크플로우 전용 내부 처리 모듈
│   │   ├── product_preserve/
│   │   │   ├── service.py         # 상품 보존 광고 이미지 생성 흐름
│   │   │   ├── renderer.py        # 상품과 배경·문구를 1080×1080로 합성
│   │   │   └── contract.py        # 상품 보존 규칙과 템플릿 설정
│   │   ├── reference_edit/
│   │   │   ├── service.py         # 레퍼런스 이미지 편집 API 연결
│   │   │   ├── prompt.py          # 확정된 이미지 편집 내부 프롬프트
│   │   │   └── validation.py      # 업로드 이미지와 요청사항 검증
│   │   └── sketch/
│   │       ├── service.py         # 스케치 기반 이미지 생성 흐름
│   │       ├── generator.py       # FLUX.2 Klein 로드와 추론
│   │       └── prompt.py          # 안내선·화살표·문구 공간 처리 규칙
│   └── shared/
│       ├── input_schema.py        # 공통 사용자 입력 구조와 검증
│       ├── background_remover.py  # 상품 이미지 배경 제거
│       ├── banner_generator.py    # 공통 광고 배너 템플릿 생성
│       ├── combined_banner_renderer.py # 상품 이미지와 문구 합성
│       └── export_bundle.py       # PNG·TXT·JSON·ZIP 결과 내보내기
├── data/
│   ├── ad_copy/                   # 광고 문구 추천 개발·테스트 자료
│   ├── product_preserve/          # 상품 사진 꾸미기 개발·테스트 자료
│   ├── reference_edit/            # 사진 새롭게 바꾸기 개발·테스트 자료
│   └── sketch/                    # 스케치 기반 이미지 개발·테스트 자료
│       ├── development/
│       │   ├── inputs/            # 개발용 원본 이미지
│       │   ├── requests.json      # 개발용 요청사항
│       │   └── outputs/           # 개발용 생성 결과와 평가 자료
│       └── test/
│           ├── inputs/            # 테스트용 원본 이미지
│           ├── requests.json      # 테스트용 요청사항
│           └── outputs/           # 테스트용 생성 결과와 평가 자료
├── docs/features/
│   ├── ad_copy/README.md          # 광고 문구 추천 작동 방식과 사용법
│   ├── product_preserve/README.md # 상품 사진 꾸미기 작동 방식과 사용법
│   ├── reference_edit/README.md   # 사진 새롭게 바꾸기 작동 방식과 사용법
│   └── sketch/README.md           # 스케치 광고 기능 작동 방식과 사용법
├── docs/report/
│   └── store_onecut_final_report.pdf # 프로젝트 결과와 평가를 정리한 최종 보고서
├── tests/                         # 최종 통합 패키지 회귀 테스트
├── notebooks/
│   └── feature4_flux2_klein_colab.ipynb # 출력 셀을 제거한 기능 4 노트북
├── assets/fonts/                  # 광고 배너용 한글 폰트
├── .streamlit/config.toml         # Streamlit 화면 설정
├── .env.example                   # API 키 환경변수 템플릿(실제 값 없음)
├── .gitignore                     # 키·캐시·로그 제외 규칙
├── requirements.txt               # Streamlit 앱 기본 의존성
├── requirements-feature4.txt      # FLUX 스케치 기능 추가 의존성
└── README.md
```

공개본의 `src`에는 버전별 파일을 나열하지 않고, 최종 동작에 필요한 코드만 역할별
모듈로 통합합니다. 개발 과정의 이전 프롬프트와 비교 실험 코드는 포함하지 않습니다.

## 기능별 사용 안내

- [광고 문구 추천](docs/features/ad_copy/README.md)
- [상품 사진 꾸미기](docs/features/product_preserve/README.md)
- [사진 새롭게 바꾸기](docs/features/reference_edit/README.md)
- [스케치를 광고로 완성하기](docs/features/sketch/README.md)

## 실행 방법

Python 3.11 또는 3.12를 권장합니다.

### 1. 저장소 내려받기

```bash
git clone https://github.com/202030312/codeit-AI12-part4-dh.git
cd codeit-AI12-part4-dh
```

### 2. 가상환경과 패키지 설치

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS 또는 Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env.example`을 `.env`로 복사한 뒤 본인의 API 키를 입력합니다.

```dotenv
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

API 키가 들어 있는 `.env`는 GitHub에 올리지 않습니다.

### 4. 앱 실행

```bash
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 열면 앱을 확인할 수 있습니다.

### 5. 테스트

```bash
python -m pytest -q
```

## 스케치 기능 실행

스케치 기능은 FLUX.2 Klein을 직접 실행하므로 별도 패키지와 L4급 GPU 환경을
권장합니다.

```bash
python -m pip install -r requirements-feature4.txt
python -m streamlit run app.py
```

## GitHub 공개 원칙

- GitHub에는 앱에서 사용하는 기능별 확정본만 올립니다.
- 기능별 최종 개발·테스트 원본, 요청사항 JSON, 생성 결과와 평가 자료를 함께 보존합니다.
- 중간 프롬프트 버전, ZIP 파일, 캐시, 로그와 임시 파일은 올리지 않습니다.
- 이후 버전 변경은 중복 파일 대신 Git 커밋과 태그로 기록합니다.
- 업로드 전 API 키와 100MB 이상 파일이 없는지 확인합니다.
- `.env`는 제외하고 값이 비어 있는 `.env.example`만 제공합니다.

최종 보고서는 `docs/report/store_onecut_final_report.pdf`에 포함되어 있습니다.
