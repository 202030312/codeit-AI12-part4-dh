"""API 모델을 사용해 SNS 광고 문구를 생성한다."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
from typing import Any, Literal, TypedDict


ProviderName = Literal["gemini", "openai", "huggingface"]

PRIMARY_MODEL_ID = "gemini-3.5-flash-lite"
COPY_PROMPT_VERSION = "gemini_copy_v0.4.1-vision"


@dataclass(frozen=True)
class ModelConfig:
    """비교 실험에 사용하는 모델의 실행 설정."""

    model_id: str
    label: str
    provider: ProviderName
    api_key_env: str


EXPERIMENT_MODELS = (
    ModelConfig(
        model_id=PRIMARY_MODEL_ID,
        label="Gemini 3.5 Flash-Lite",
        provider="gemini",
        api_key_env="GEMINI_API_KEY",
    ),
    ModelConfig(
        model_id="gemini-3.8-flash",
        label="Gemini 3.8 Flash",
        provider="gemini",
        api_key_env="GEMINI_API_KEY",
    ),
    ModelConfig(
        model_id="gpt-6-luna",
        label="GPT-6 Luna",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
    ),
    ModelConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        label="Qwen3-4B-Instruct-2507",
        provider="huggingface",
        api_key_env="HF_TOKEN",
    ),
)
MODEL_CONFIGS = {config.model_id: config for config in EXPERIMENT_MODELS}


class AdCopy(TypedDict):
    """화면과 향후 배너 생성 단계에서 사용하는 광고 문구 한 세트."""

    headline: str
    body: str
    hashtags: list[str]


COPY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "copies": {
            "type": "array",
            "description": "입력 정보량에 맞춰 작성한 SNS 광고 문구 1~3종",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "headline": {
                        "type": "string",
                        "minLength": 6,
                        "maxLength": 20,
                        "description": "공백 포함 6~20자인 완결된 배너용 핵심 문구",
                    },
                    "body": {
                        "type": "string",
                        "minLength": 0,
                        "maxLength": 34,
                        "description": (
                            "제목과 다른 입력 사실이 있을 때만 쓰는 34자 이하의 보조 설명. "
                            "불필요하면 빈 문자열을 반환함"
                        ),
                    },
                },
                "required": ["headline", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["copies"],
    "additionalProperties": False,
}


SYSTEM_INSTRUCTIONS = """당신은 한국의 카페·디저트 소상공인을 돕는 SNS 광고 카피라이터입니다.
입력으로 제공된 JSON의 정보만 사용해 한국어 광고 문구를 1~3종 작성하세요. 서로 다른 유효 문구가 한 종뿐이면 한 종만 반환하고, 개수를 맞추려고 문구를 억지로 늘리지 마세요.

[headline과 body 작성 규칙]
1. 사용자가 입력한 상품 정보와 특징만 근거로 작성합니다. 입력에 없는 원재료, 맛·향·식감, 제조 방식, 신선도, 건강 효능을 추가하지 않습니다.
2. headline은 가장 중요한 특징 한 가지를 짧고 자연스럽게 강조합니다. 입력에 특징이 여러 개 있어도 headline에서 전부 설명하지 않고, body에 사용할 특징을 남겨둡니다.
3. body는 headline에서 사용하지 않은 특징 한 가지를 구체적으로 설명합니다. 같은 재료명을 다시 사용할 수는 있지만, headline의 내용을 서술어만 바꾸어 반복하지 않습니다. '조합 → 어우러짐', '향 → 향의 매력'처럼 같은 의미를 바꾸어 쓰는 것도 반복으로 봅니다.
4. features 배열의 항목 개수와 실제 정보 개수를 동일하게 보지 않습니다. 하나의 문자열 안에도 맛, 향, 식감, 색상, 구성, 형태처럼 서로 독립적으로 설명할 수 있는 사실이 두 개 이상 포함될 수 있습니다. 입력 JSON의 fact_policy는 앱이 features를 의미 단위로 정리한 결과이므로 우선 적용합니다. distinct_fact_count가 1이면 headline에 사용하고 body는 빈 문자열로 반환합니다. distinct_fact_count가 2 이상이고 body_required가 true이면 가장 중요한 사실 하나를 headline에 사용하고, headline에서 사용하지 않은 다른 사실 하나를 body에 사용합니다. 이때 body를 빈 문자열로 반환하지 않습니다.
5. 입력 특징이 'A와 B의 조합'처럼 두 대상을 연결하는 하나의 관계형 특징이라면 두 대상이 있어도 하나의 사실로 처리합니다. A와 B를 모두 유지하고, 관계 표현은 입력 의미를 바꾸지 않는 범위에서 '조합', '조화', '어우러짐'으로 표현할 수 있습니다. 두 대상을 '두 가지 메뉴', '특별한 만남' 같은 추상적인 표현으로 대체하지 않으며 body는 빈 문자열로 반환합니다.
6. body는 필수가 아닙니다. headline과 다른 구체적인 입력 사실이 있을 때만 공백과 문장부호를 포함해 34자 이하의 한 문장으로 작성합니다. 최소 글자 수는 없으며, 길이를 맞추기 위한 불필요한 수식어를 넣지 않습니다.
7. body를 작성할 때는 특징을 직접 설명하는 부드러운 해요체를 사용합니다. '~해요.', '~이에요.', '~느껴져요.' 등을 내용에 맞게 사용합니다. 목적격 조사 '을/를' 뒤에 '느껴져요'를 연결하지 않으며, 주격 조사 '이/가 느껴져요' 또는 목적격 조사 '을/를 느낄 수 있어요'처럼 자연스럽게 씁니다. 자연스러운 문장을 확실히 만들 수 없다면 body는 빈 문자열로 반환합니다. '즐겨보세요.', '느껴보세요.', '만나보세요.' 등 소비자에게 행동을 요청하는 표현은 사용하지 않습니다.
8. 특징의 종류에 맞는 표현을 선택합니다. 맛·향·식감은 감각적으로 설명하고, 색상·구성·형태는 그 내용을 사실대로 설명합니다. '여러 색을 느껴보세요'처럼 특징과 어울리지 않는 서술어를 사용하지 않습니다. 감각적 특징이 입력되지 않았다면 구성·색상·형태를 설명할 수 있습니다.
9. '풍미', '매력', '가득', '조합' 같은 단어만으로 특징 설명을 끝내지 않습니다. 입력에 구체적인 감각 표현이 있다면 그 표현을 우선 활용합니다.
10. product.name의 전체 문자열은 headline과 body에 사용하지 않습니다. 상품명은 별도 라벨에 표시되므로 headline은 product.features의 표현을 중심으로 작성합니다. 특징에 실제로 포함된 원재료명은 사용할 수 있지만 product.name 전체와 동일한 명칭은 사용하지 않습니다. 두 영역을 합쳐 입력 특징의 핵심 단어를 최소 한 개 직접 포함합니다.
11. 방문·주문·할인·증정 안내는 별도 영역에 표시되므로 headline과 body에 추가하지 않습니다.
12. 쉼표, 줄바꿈, 세미콜론, 글머리표로 구분된 특징은 각각 보존합니다. 접속 표현으로 이어진 하나의 문자열도 실제로 독립된 사실이 여러 개인지 의미를 기준으로 판단합니다.
13. 해시태그는 앱이 별도로 생성하므로 작성하지 않습니다. 출력에는 headline과 body만 포함합니다.
14. 출력 전에 입력 근거, 실제 사실 수, body 글자 수, 자연스러운 한국어, headline과 body의 의미 중복을 확인하고 수정합니다. 검토 과정은 출력하지 않으며 기존 출력 형식을 유지합니다.

[공통 표현 및 안전 규칙]
- 입력 JSON에서 확인할 수 있는 정보만 사용합니다. 입력에 없는 사실, 가격, 할인, 행사, 기간, 혜택을 만들지 않습니다.
- campaign.copy_request는 문구의 말투, 길이, 강조 순서를 조절하는 선택 지시입니다. 상품 사실의 근거가 아니므로 copy_request에만 적힌 맛·향·식감·재료·효능·가격·행사 내용을 문구에 추가하지 않습니다.
- copy_request가 product.features와 충돌하면 product.features를 우선하고 충돌하는 요청은 무시합니다. copy_request가 비어 있으면 기존 headline과 body 규칙만 적용합니다.
- 상품 이미지가 함께 제공되면 색상, 형태, 개수, 포장처럼 화면에서 명확하게 확인되는 정보만 보조 맥락으로 사용합니다. 이미지의 배경, 조명, 소품, 촬영 장소는 상품 사실로 사용하지 않습니다.
- 이미지에서 맛, 향, 식감, 온도, 원재료, 원산지, 제조 방식, 신선도, 효능을 추측하지 않습니다. 이미지와 product 정보가 충돌하면 사용자가 입력한 product 정보를 우선하며, 확실하지 않은 시각 정보는 문구에서 생략합니다.
- 한국 소상공인의 SNS 광고에 어울리는 친근하고 명확한 표현을 사용하며, 과장되거나 지나치게 고급스러운 표현은 피합니다.
- 입력에 없는 원산지, 재료, 효능, 수상 이력, 기간, 할인율 등의 사실을 만들지 않습니다.
- null인 항목은 문구에서 자연스럽게 생략합니다.
- 선택된 광고 분위기를 유지하되 여러 문구를 반환한다면 표현과 강조점은 서로 다르게 합니다.
- 광고 분위기 값은 문체를 정하는 용도로만 사용하고 headline이나 body에 그대로 복사하지 않습니다. 분위기 형용사를 맛·향·식감·조합 같은 제품 속성에 직접 붙이지 않습니다.
- headline은 공백 포함 6~20자로 작성합니다. 짧아도 자연스럽고 뜻이 완결되면 억지로 늘리지 않습니다.
- headline과 body 안에 줄바꿈 문자를 넣지 않습니다.
- headline 끝에는 마침표, 느낌표, 물음표 같은 문장부호를 붙이지 않습니다.
- body를 '합니다', '입니다', '~다' 같은 문어체 종결로 끝내지 않습니다.
- 매장명은 배너에 별도로 표시되므로 headline과 body에 매장명을 반복하지 않습니다.
- 프로모션은 배너의 행사 배지에 별도로 표시되므로 headline과 body에 할인 금액, 할인율, 시간, 증정 조건을 반복하지 않습니다.
- 가격은 배너에 별도로 표시되므로 headline과 body에 반복하지 않습니다.
- headline은 상품명만 바꾸면 다른 상품에도 그대로 쓸 수 있는 지나치게 일반적인 표현을 피합니다.
- headline은 그 자체로 뜻이 끝나는 완결된 문구로 씁니다. '가득한', '마시는', '담은'처럼 뒤에 명사가 더 필요한 말로 끝내지 않습니다.
- '디저트에요'가 아니라 '디저트예요'처럼 맞춤법에 맞는 자연스러운 한국어를 사용합니다.
- '향이 입안에 퍼져보세요', '맛을 맛보세요'처럼 목적어와 서술어가 어색하거나 의미가 중복되는 표현을 사용하지 않습니다.
- 입력 특징에 '특징인' 같은 메모형 표현이 있으면 그대로 복사하지 말고 자연스러운 광고 문장으로 고칩니다. 예: 'X 향이 특징인 상품'은 'X 향'을 핵심으로 씁니다.
- 'A와 B의 조합'처럼 두 대상을 잇는 특징은 한쪽을 임의로 빼지 않습니다. 하나의 상품을 이루는 재료라면 입력에 적힌 A와 B를 원재료명으로 자연스럽게 사용합니다.
- 조사 앞에 쉼표를 붙인 '의,', '을,', '를,' 같은 표현을 만들지 않습니다.
- 특징의 종류와 맞지 않는 서술어를 사용하지 않으며, body는 소비자에게 행동을 요청하는 문장이 아니라 상품 특징을 직접 설명하는 문장으로 씁니다.
- headline과 body의 내용어는 입력의 상품 특징 표현을 우선 사용합니다. 입력 밖의 중립 표현도 문장을 자연스럽게 잇는 데 필요한 최소 범위에서만 사용합니다.
- 입력에 같은 뜻이 없으면 '매력적인, 특별한, 노릇하게, 독특한, 환상의, 새로운, 깊은, 풍부한, 기분 좋은' 같은 수식어도 추가하지 않습니다.
- 같은 문장 안에서 '향 ... 향', '풍미 ... 풍미', '식감 ... 식감', '맛 ... 맛'처럼 핵심 단어를 반복하지 않습니다.
- '풍미 가득한 맛', '빵 결의 특징', '느껴지는 깔끔한 풍미'처럼 번역투이거나 의미가 겹치는 표현을 사용하지 않습니다.
- 입력의 감각 범주를 바꾸지 않습니다. 식감은 식감으로, 향은 향으로, 풍미는 풍미로 표현하고 임의로 모두 '맛'이라고 바꾸지 않습니다.
- 입력에 근거가 없다면 '오늘 구운', '갓 구운', '매일 마셔도 질리지 않는', '누구나 좋아하는', '최고의', '꼭 먹어야 하는' 같은 주장을 사용하지 않습니다.
- 상품명이나 사진만 보고 맛·향·온도·재료·식감을 추측하지 않습니다. 특히 입력 특징에 같은 의미가 없으면 '촉촉한', '시원한/시원하게', '청량감', '톡 쏘는', '차가운', '따뜻한', '부드러운', '달콤한', '고소한', '바삭한', '쫄깃한'을 쓰지 않습니다.
- 상품명에 온도나 조리 상태를 연상시키는 단어가 있어도 입력 특징에 같은 의미가 없다면 온도·청량감·식감을 추가하지 않습니다.
- 입력 특징을 자연스럽게 활용하되, 근거를 늘리기 위해 새로운 감각 형용사를 덧붙이지 않습니다.

출력 직전에는 각 문구를 다시 읽고 입력에 없는 사실이나 수식어, 어색한 조사·수량어·목적어와 서술어, 메모용 단어, 조사가 빠진 명사 나열, 별도 표시되는 매장명·가격·행사 문구의 반복을 제거합니다. headline과 body에서 product.name 전체 문자열을 다시 찾아 제거합니다. 위 점검을 마친 최종 JSON만 반환합니다.
"""


class CopyGenerationError(RuntimeError):
    """모델 응답을 광고 문구 1~3종으로 해석할 수 없을 때 발생한다."""


def split_product_features(value: str | None) -> list[str]:
    """사용자가 적은 특징을 구분 기호 기준으로 보존해 분리한다."""
    if not value:
        return []
    parts = re.split(r"(?:\r?\n|[,;；]|[•·])", value)
    features: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^\s*[-*]\s*", "", part).strip()
        cleaned = " ".join(cleaned.split())
        meta_match = re.fullmatch(r"(.+?)(?:이|가) 특징인 .+", cleaned)
        if meta_match:
            cleaned = meta_match.group(1).strip()
        if cleaned and cleaned not in features:
            features.append(cleaned)
    return features


def split_distinct_feature_facts(value: str | None) -> list[str]:
    """입력 문자열을 의미상 독립적으로 설명할 수 있는 사실 단위로 나눈다.

    단순히 재료가 둘이라는 이유로 사실을 둘로 세지 않는다. 특히
    ``A와 B의 조합``은 관계 전체가 하나의 사실이다. 반면 향·식감처럼
    서로 다른 속성이나 겉·속처럼 독립된 대조 속성은 각각 보존한다.
    """

    feature_items = split_product_features(value)
    facts: list[str] = []
    category_terms = (
        "맛",
        "향",
        "풍미",
        "식감",
        "색",
        "색상",
        "구성",
        "형태",
        "모양",
        "겉",
        "속",
        "윗면",
        "결",
    )

    def add_fact(fact: str) -> None:
        cleaned = " ".join(fact.split()).strip(" ,")
        if cleaned and cleaned not in facts:
            facts.append(cleaned)

    def normalize_connected_predicate(predicate: str) -> str:
        predicate = predicate.strip()
        if predicate.endswith("하"):
            return predicate[:-1] + "함"
        if predicate and not predicate.endswith(
            ("맛", "향", "풍미", "식감", "색", "색상", "구성", "형태", "모양", "결", "함")
        ):
            return predicate + "함"
        return predicate

    def looks_independent(part: str) -> bool:
        normalized = part.strip()
        return normalized.endswith("함") or any(
            term in normalized for term in category_terms
        ) or bool(
            re.search(
                r"(?:한|운|로운|스러운|있는|없는|짙은|진한|연한)\s+[0-9A-Za-z가-힣]+$",
                normalized,
            )
        )

    for item in feature_items:
        # 두 대상을 잇는 관계 자체가 정보인 경우에는 재료 수와 무관하게 한
        # 사실로 둔다. 조화·어우러짐도 같은 관계형 입력으로 취급한다.
        if re.search(r"(?:와|과)\s*.+(?:조합|조화|어우러)", item):
            add_fact(item)
            continue

        # '겉은 바삭하고 속은 촉촉한 식감'과 '담백하고 쫄깃한 식감'처럼
        # 접속 어미로 이어진 독립 속성을 보존한다.
        connected = re.fullmatch(r"(.+?)(?:하고|하며)\s+(.+)", item)
        if connected:
            left = normalize_connected_predicate(connected.group(1))
            right = connected.group(2).strip()
            if looks_independent(left) and looks_independent(right):
                add_fact(left)
                add_fact(right)
                continue

        # '진한 향과 부드러운 식감'처럼 양쪽이 각각 설명 가능한 속성일
        # 때만 나눈다. 단순 원재료 나열은 그대로 한 사실로 남긴다.
        paired = re.fullmatch(r"(.+?)(?:와|과)\s+(.+)", item)
        if paired:
            left, right = paired.group(1).strip(), paired.group(2).strip()
            if looks_independent(left) and looks_independent(right):
                add_fact(left)
                add_fact(right)
                continue

        add_fact(item)

    return facts


def build_deterministic_hashtags(ad_input: dict[str, Any]) -> list[str]:
    """모델과 분리해 상품명·매장명으로 안전한 해시태그를 만든다."""
    values = (
        ad_input.get("product", {}).get("name"),
        ad_input.get("store", {}).get("name"),
    )
    hashtags: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"[^0-9A-Za-z가-힣]", "", value)
        hashtag = f"#{normalized}" if normalized else ""
        if hashtag and hashtag not in hashtags:
            hashtags.append(hashtag)
    return hashtags or ["#상품"]


def _build_prompt(
    ad_input: dict[str, Any],
    retry_feedback: str | None = None,
    *,
    image_provided: bool = False,
    prompt_version: str = COPY_PROMPT_VERSION,
) -> str:
    product = ad_input.get("product", {})
    raw_features = product.get("features")
    raw_feature_text = raw_features if isinstance(raw_features, str) else None
    features = split_product_features(raw_feature_text)
    distinct_facts = split_distinct_feature_facts(raw_feature_text)
    # 파일명과 MIME 형식은 저장·식별에만 사용한다. 모델이 파일명의 busy,
    # windowlight 같은 실험 조건을 상품 특징으로 오해하지 않도록 아예 제외한다.
    copy_context = {
        "store": {"name": ad_input.get("store", {}).get("name")},
        "product": {
            "name": product.get("name"),
            "base_price": product.get("base_price", product.get("price")),
            "sale_price": product.get("sale_price"),
            "features": features or None,
        },
        "campaign": {
            "promotion": ad_input.get("campaign", {}).get("promotion"),
            "tone": ad_input.get("campaign", {}).get("tone"),
            "copy_request": ad_input.get("campaign", {}).get("copy_request"),
        },
        "fact_policy": {
            "distinct_facts": distinct_facts,
            "distinct_fact_count": len(distinct_facts),
            "body_required": len(distinct_facts) >= 2,
        },
        "image_context": {
            "provided": image_provided,
            "role": (
                "명확하게 보이는 색상·형태·개수·포장 확인용 보조 맥락"
                if image_provided
                else "제공되지 않음"
            ),
        },
    }
    prompt = (
        f"문구 프롬프트 버전: {prompt_version}\n"
        "다음 광고 문구용 입력 JSON을 바탕으로 유효한 문구를 1~3종 작성하세요. "
        "파일명과 파일 경로는 제공되지 않으며 추측해서도 안 됩니다.\n"
        + json.dumps(copy_context, ensure_ascii=False, indent=2)
        + "\n\n상품 사실의 허용 근거는 product.features에 실제로 적힌 값과, 제공된 "
        "이미지에서 명확하게 확인되는 색상·형태·개수·포장뿐입니다. 두 근거가 "
        "충돌하면 product.features를 우선합니다. product.name은 상품 식별용이고 "
        "campaign.tone과 campaign.copy_request는 표현 제어용일 뿐 상품 사실의 "
        "근거가 아닙니다. 상품명·이미지·문구 요청사항에서 맛·향·식감·온도·재료·"
        "효능을 추론하지 마세요. 이미지의 배경·조명·소품·촬영 장소도 상품 특징으로 "
        "사용하지 마세요."
    )
    if retry_feedback:
        prompt += (
            "\n\n이전 응답은 아래 이유로 사용할 수 없었습니다. "
            "문제를 수정하되 입력 JSON에 없는 사실은 추가하지 마세요.\n"
            f"- {retry_feedback}"
        )
    return prompt


def _create_gemini_client(api_key: str, timeout_ms: int = 120_000) -> Any:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise CopyGenerationError(
            "Google Gen AI 패키지가 설치되지 않았습니다. requirements.txt를 설치해주세요."
        ) from exc

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _create_openai_client(
    api_key: str,
    timeout_seconds: float = 120.0,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise CopyGenerationError(
            "OpenAI 패키지가 설치되지 않았습니다. requirements.txt를 설치해주세요."
        ) from exc

    return OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=1,
    )


def _create_huggingface_client(api_key: str, timeout_seconds: int = 90) -> Any:
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise CopyGenerationError(
            "Hugging Face Hub 패키지가 설치되지 않았습니다. requirements.txt를 설치해주세요."
        ) from exc

    return InferenceClient(provider="auto", api_key=api_key, timeout=timeout_seconds)


def resolve_api_key(config: ModelConfig) -> str:
    """환경 변수 또는 Hugging Face 로그인 캐시에서 API 토큰을 찾는다."""
    import os

    api_key = os.getenv(config.api_key_env, "").strip()
    if api_key or config.provider != "huggingface":
        return api_key

    try:
        from huggingface_hub import get_token
    except ImportError:
        return ""
    return (get_token() or "").strip()


def _generate_with_gemini(
    *,
    prompt: str,
    model_id: str,
    api_key: str,
    client: Any | None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    response_schema: dict[str, Any] = COPY_RESPONSE_SCHEMA,
    thinking_level: str | None = None,
) -> str:
    gemini_client = client or _create_gemini_client(api_key)
    text_input = system_instructions + "\n" + prompt
    interaction_input: str | list[dict[str, str]] = text_input
    if image_bytes is not None:
        interaction_input = [
            {"type": "text", "text": text_input},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("ascii"),
                "mime_type": image_mime_type or "image/png",
            },
        ]
    generation_config: dict[str, Any] = {
        "temperature": 0.2,
        "top_p": 0.8,
        "max_output_tokens": 700,
    }
    if thinking_level:
        generation_config["thinking_level"] = thinking_level
    interaction = gemini_client.interactions.create(
        model=model_id,
        input=interaction_input,
        generation_config=generation_config,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": response_schema,
        },
    )
    return getattr(interaction, "output_text", "") or ""


def _generate_with_gemini_generate_content(
    *,
    prompt: str,
    model_id: str,
    api_key: str,
    client: Any | None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    response_schema: dict[str, Any] = COPY_RESPONSE_SCHEMA,
    thinking_level: str | None = None,
) -> str:
    """일반 멀티모달 생성 API로 동일한 구조화 문구를 생성한다."""
    try:
        from google.genai import types
    except ImportError as exc:
        raise CopyGenerationError(
            "Google Gen AI 패키지가 설치되지 않았습니다. requirements.txt를 설치해주세요."
        ) from exc

    gemini_client = client or _create_gemini_client(api_key)
    parts = [types.Part.from_text(text=system_instructions + "\n" + prompt)]
    if image_bytes is not None:
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=image_mime_type or "image/png",
            )
        )
    thinking_config = (
        types.ThinkingConfig(thinking_level=thinking_level)
        if thinking_level
        else None
    )
    response = gemini_client.models.generate_content(
        model=model_id,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=700,
            response_mime_type="application/json",
            response_json_schema=response_schema,
            thinking_config=thinking_config,
        ),
    )
    return getattr(response, "text", "") or ""


def _first_complete_json_object(output_text: str) -> str:
    """응답 뒤에 설명이나 중복 출력이 붙어도 첫 완전한 JSON 객체만 보존한다."""

    stripped = output_text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return stripped
    except json.JSONDecodeError:
        pass

    for start, character in enumerate(stripped):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(stripped)):
            current = stripped[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : end + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return candidate
                    break
    return stripped


def _generate_with_openai(
    *,
    prompt: str,
    model_id: str,
    api_key: str,
    client: Any | None,
    system_instructions: str = SYSTEM_INSTRUCTIONS,
    response_schema: dict[str, Any] = COPY_RESPONSE_SCHEMA,
    max_output_tokens: int = 1200,
) -> tuple[str, dict[str, int]]:
    """Responses API와 Structured Outputs로 JSON 문구를 생성한다."""

    openai_client = client or _create_openai_client(api_key)
    response = openai_client.responses.create(
        model=model_id,
        instructions=system_instructions,
        input=prompt,
        reasoning={"effort": "none"},
        max_output_tokens=max_output_tokens,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "banner_copy_candidates",
                "strict": True,
                "schema": response_schema,
            }
        },
    )
    status = str(getattr(response, "status", "") or "")
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) or "unknown"
        raise CopyGenerationError(f"OpenAI 응답이 완료되지 않았습니다: {reason}")

    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    usage_payload = {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(
            getattr(input_details, "cached_tokens", 0) or 0
        ),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(
            getattr(output_details, "reasoning_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    output_text = getattr(response, "output_text", "") or ""
    return _first_complete_json_object(output_text), usage_payload


def _generate_with_huggingface(
    *,
    prompt: str,
    model_id: str,
    api_key: str,
    client: Any | None,
) -> str:
    huggingface_client = client or _create_huggingface_client(api_key)
    completion = huggingface_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sns_ad_copy_variants",
                "strict": True,
                "schema": COPY_RESPONSE_SCHEMA,
            },
        },
        temperature=0.7,
        max_tokens=900,
    )
    return completion.choices[0].message.content or ""


def _parse_copies(output_text: str, ad_input: dict[str, Any]) -> list[AdCopy]:
    try:
        payload = json.loads(output_text)
        copies = payload["copies"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CopyGenerationError("AI 응답을 광고 문구 형식으로 읽을 수 없습니다.") from exc

    if not isinstance(copies, list) or not 1 <= len(copies) <= 3:
        raise CopyGenerationError("AI가 광고 문구를 1~3종으로 반환하지 않았습니다.")

    required_keys = {"headline", "body"}
    hashtags = build_deterministic_hashtags(ad_input)
    parsed_copies: list[AdCopy] = []
    for copy in copies:
        if not isinstance(copy, dict) or set(copy) != required_keys:
            raise CopyGenerationError("AI 광고 문구의 항목 구성이 올바르지 않습니다.")
        if not isinstance(copy["headline"], str) or not copy["headline"].strip():
            raise CopyGenerationError("AI 광고 문구의 제목이 비어 있습니다.")
        if not isinstance(copy["body"], str):
            raise CopyGenerationError("AI 광고 문구의 본문 형식이 올바르지 않습니다.")
        parsed_copies.append(
            {
                "headline": copy["headline"],
                "body": copy["body"],
                "hashtags": hashtags.copy(),
            }
        )

    return parsed_copies


def generate_ad_copies(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    retry_feedback: str | None = None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> list[AdCopy]:
    """선택한 API 모델로 구조화된 광고 문구 1~3종을 생성한다."""
    config = MODEL_CONFIGS.get(model_id)
    if config is None:
        raise CopyGenerationError(f"지원하지 않는 실험 모델입니다: {model_id}")
    if not api_key.strip():
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")
    if image_bytes is not None and config.provider != "gemini":
        raise CopyGenerationError("상품 이미지 입력은 Gemini 모델에서만 지원합니다.")
    if image_bytes is not None and image_mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }:
        raise CopyGenerationError("지원하지 않는 상품 이미지 형식입니다.")

    prompt = _build_prompt(
        ad_input,
        retry_feedback,
        image_provided=image_bytes is not None,
    )
    if config.provider == "gemini":
        output_text = _generate_with_gemini(
            prompt=prompt,
            model_id=model_id,
            api_key=api_key,
            client=client,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )
    elif config.provider == "openai":
        output_text, _ = _generate_with_openai(
            prompt=prompt,
            model_id=model_id,
            api_key=api_key,
            client=client,
        )
    else:
        output_text = _generate_with_huggingface(
            prompt=prompt,
            model_id=model_id,
            api_key=api_key,
            client=client,
        )

    if not output_text:
        raise CopyGenerationError("AI가 광고 문구를 반환하지 않았습니다.")
    return _parse_copies(output_text, ad_input)
