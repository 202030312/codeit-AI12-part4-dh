import base64
import json
from types import SimpleNamespace
import unittest

from src.features.ad_copy.generator import (
    COPY_PROMPT_VERSION,
    COPY_RESPONSE_SCHEMA,
    EXPERIMENT_MODELS,
    PRIMARY_MODEL_ID,
    SYSTEM_INSTRUCTIONS,
    CopyGenerationError,
    _first_complete_json_object,
    generate_ad_copies,
    split_distinct_feature_facts,
    split_product_features,
)


MODEL_COPIES = [
    {
        "headline": "딸기 향이 피어나는 순간",
        "body": "생딸기의 산뜻한 풍미를 천천히 즐겨보세요.",
    },
    {
        "headline": "부드러운 여운을 담은 오늘",
        "body": "수제 크림의 부드러운 결을 편안히 느껴보세요.",
    },
    {
        "headline": "산뜻함과 부드러움의 조화",
        "body": "생딸기와 수제 크림의 조화를 만나보세요.",
    },
]

EXPECTED_COPIES = [
    {
        **copy,
        "hashtags": ["#딸기크림라떼", "#오늘의작은카페"],
    }
    for copy in MODEL_COPIES
]


class FakeInteractions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeChatCompletions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=self.output_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            output_text=self.output_text,
            status="completed",
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=40,
                total_tokens=160,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )


class GenerateAdCopiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ad_input = {
            "schema_version": "1.1",
            "image": {"file_name": "latte.png", "content_type": "image/png"},
            "store": {"name": "오늘의 작은 카페"},
            "product": {
                "name": "딸기 크림 라떼",
                "base_price": "6,500원",
                "sale_price": "5,500원",
                "features": "생딸기 사용, 수제 크림",
            },
            "campaign": {
                "promotion": "이번 주말 10% 할인",
                "tone": "감성적인",
                "copy_request": "생딸기 풍미를 먼저 강조해 주세요.",
            },
        }
        self.output_text = json.dumps({"copies": MODEL_COPIES}, ensure_ascii=False)

    def test_experiment_has_expected_models(self) -> None:
        self.assertEqual(
            [config.model_id for config in EXPERIMENT_MODELS],
            [
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash",
                "gpt-6-luna",
                "Qwen/Qwen3-4B-Instruct-2507",
            ],
        )

    def test_system_prompt_has_small_business_copy_rules(self) -> None:
        self.assertIn("입력에 없는 사실, 가격, 할인, 행사", SYSTEM_INSTRUCTIONS)
        self.assertIn("한국 소상공인의 SNS 광고", SYSTEM_INSTRUCTIONS)
        self.assertIn("headline은 가장 중요한 특징", SYSTEM_INSTRUCTIONS)
        self.assertIn("행동을 요청하는 표현은 사용하지 않습니다", SYSTEM_INSTRUCTIONS)
        self.assertIn("디저트예요", SYSTEM_INSTRUCTIONS)
        self.assertIn("오늘 구운", SYSTEM_INSTRUCTIONS)
        self.assertIn("촉촉한", SYSTEM_INSTRUCTIONS)
        self.assertIn("상품명이나 사진만 보고", SYSTEM_INSTRUCTIONS)
        self.assertEqual(PRIMARY_MODEL_ID, "gemini-3.5-flash-lite")
        self.assertEqual(COPY_PROMPT_VERSION, "gemini_copy_v0.4.1-vision")
        self.assertIn("campaign.copy_request", SYSTEM_INSTRUCTIONS)
        self.assertIn("상품 사실의 근거가 아니므로", SYSTEM_INSTRUCTIONS)
        self.assertIn("상품 이미지가 함께 제공되면", SYSTEM_INSTRUCTIONS)
        self.assertIn("이미지에서 맛, 향, 식감", SYSTEM_INSTRUCTIONS)
        self.assertIn("1~3종", SYSTEM_INSTRUCTIONS)
        self.assertIn("body는 빈 문자열", SYSTEM_INSTRUCTIONS)
        self.assertIn("A와 B의 조합", SYSTEM_INSTRUCTIONS)
        self.assertIn("product.name의 전체 문자열", SYSTEM_INSTRUCTIONS)
        self.assertIn("을/를' 뒤에 '느껴져요", SYSTEM_INSTRUCTIONS)
        self.assertIn("특징인", SYSTEM_INSTRUCTIONS)
        self.assertIn("A와 B의 조합", SYSTEM_INSTRUCTIONS)
        self.assertIn("노릇하게", SYSTEM_INSTRUCTIONS)
        self.assertIn("하나의 상품을 이루는 재료", SYSTEM_INSTRUCTIONS)
        self.assertIn("실제 정보 개수", SYSTEM_INSTRUCTIONS)
        self.assertIn("해시태그는 앱이 별도로 생성", SYSTEM_INSTRUCTIONS)

    def test_model_schema_excludes_app_generated_hashtags(self) -> None:
        item_properties = COPY_RESPONSE_SCHEMA["properties"]["copies"]["items"][
            "properties"
        ]
        self.assertEqual(set(item_properties), {"headline", "body"})

    def test_gemini_models_request_structured_output(self) -> None:
        for model_id in ("gemini-3.5-flash-lite", "gemini-3.8-flash"):
            with self.subTest(model_id=model_id):
                interactions = FakeInteractions(self.output_text)
                client = SimpleNamespace(interactions=interactions)

                result = generate_ad_copies(
                    self.ad_input,
                    model_id=model_id,
                    api_key="test-key",
                    client=client,
                )

                self.assertEqual(result, EXPECTED_COPIES)
                self.assertEqual(interactions.request["model"], model_id)
                self.assertEqual(
                    interactions.request["response_format"]["mime_type"],
                    "application/json",
                )
                self.assertEqual(
                    interactions.request["generation_config"]["temperature"],
                    0.2,
                )
                self.assertIn("딸기 크림 라떼", interactions.request["input"])
                self.assertNotIn("latte.png", interactions.request["input"])
                self.assertNotIn("file_name", interactions.request["input"])
                self.assertIn("base_price", interactions.request["input"])
                self.assertIn('"distinct_fact_count": 2', interactions.request["input"])
                self.assertIn('"body_required": true', interactions.request["input"])
                self.assertIn("생딸기 풍미를 먼저 강조", interactions.request["input"])
                self.assertIn("표현 제어용일 뿐 상품 사실의 근거가 아닙니다", interactions.request["input"])

    def test_gemini_request_includes_product_image_when_provided(self) -> None:
        interactions = FakeInteractions(self.output_text)
        client = SimpleNamespace(interactions=interactions)
        image_bytes = b"test-image-bytes"

        result = generate_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
            image_bytes=image_bytes,
            image_mime_type="image/png",
        )

        self.assertEqual(result, EXPECTED_COPIES)
        request_input = interactions.request["input"]
        self.assertIsInstance(request_input, list)
        self.assertEqual(request_input[0]["type"], "text")
        self.assertIn('"provided": true', request_input[0]["text"])
        self.assertIn("색상·형태·개수·포장", request_input[0]["text"])
        self.assertEqual(request_input[1]["type"], "image")
        self.assertEqual(
            request_input[1]["data"],
            base64.b64encode(image_bytes).decode("ascii"),
        )
        self.assertEqual(request_input[1]["mime_type"], "image/png")

    def test_rejects_image_for_non_gemini_provider(self) -> None:
        with self.assertRaisesRegex(CopyGenerationError, "Gemini"):
            generate_ad_copies(
                self.ad_input,
                model_id="Qwen/Qwen3-4B-Instruct-2507",
                api_key="test-key",
                image_bytes=b"test-image-bytes",
                image_mime_type="image/png",
            )

    def test_qwen_model_requests_strict_json_schema(self) -> None:
        completions = FakeChatCompletions(self.output_text)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result = generate_ad_copies(
            self.ad_input,
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result, EXPECTED_COPIES)
        self.assertEqual(
            completions.request["model"], "Qwen/Qwen3-4B-Instruct-2507"
        )
        self.assertTrue(
            completions.request["response_format"]["json_schema"]["strict"]
        )
        self.assertEqual(completions.request["max_tokens"], 900)

    def test_openai_model_uses_responses_structured_output(self) -> None:
        responses = FakeResponses(self.output_text)
        client = SimpleNamespace(responses=responses)

        result = generate_ad_copies(
            self.ad_input,
            model_id="gpt-6-luna",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result, EXPECTED_COPIES)
        self.assertEqual(responses.request["model"], "gpt-6-luna")
        self.assertEqual(responses.request["reasoning"], {"effort": "none"})
        self.assertFalse(responses.request["store"])
        self.assertTrue(responses.request["text"]["format"]["strict"])
        self.assertEqual(
            responses.request["text"]["format"]["schema"],
            COPY_RESPONSE_SCHEMA,
        )
        self.assertEqual(responses.request["max_output_tokens"], 1200)

    def test_openai_output_keeps_first_complete_json_object(self) -> None:
        first = json.dumps({"copies": MODEL_COPIES}, ensure_ascii=False)
        duplicated = first + "\n불필요한 설명\n" + first

        self.assertEqual(_first_complete_json_object(duplicated), first)

    def test_accepts_one_to_three_copies(self) -> None:
        interactions = FakeInteractions(
            json.dumps({"copies": MODEL_COPIES[:2]}, ensure_ascii=False)
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result, EXPECTED_COPIES[:2])

    def test_accepts_blank_optional_body(self) -> None:
        copies = [{**MODEL_COPIES[0], "body": ""}]
        interactions = FakeInteractions(json.dumps({"copies": copies}, ensure_ascii=False))
        client = SimpleNamespace(interactions=interactions)

        result = generate_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result[0]["body"], "")

    def test_rejects_zero_or_more_than_three_copies(self) -> None:
        for copies in ([], MODEL_COPIES + [MODEL_COPIES[0]]):
            with self.subTest(count=len(copies)):
                interactions = FakeInteractions(
                    json.dumps({"copies": copies}, ensure_ascii=False)
                )
                client = SimpleNamespace(interactions=interactions)
                with self.assertRaises(CopyGenerationError):
                    generate_ad_copies(
                        self.ad_input,
                        model_id="gemini-3.5-flash-lite",
                        api_key="test-key",
                        client=client,
                    )

    def test_splits_and_preserves_multiple_features(self) -> None:
        self.assertEqual(
            split_product_features("바삭한 겉면, 촉촉한 속\n- 로즈마리 향"),
            ["바삭한 겉면", "촉촉한 속", "로즈마리 향"],
        )

    def test_splits_distinct_facts_inside_one_feature_item(self) -> None:
        examples = (
            ("진한 향", ["진한 향"]),
            ("A와 B의 조합", ["A와 B의 조합"]),
            ("진한 향과 부드러운 식감", ["진한 향", "부드러운 식감"]),
            ("겉은 바삭하고 속은 촉촉함", ["겉은 바삭함", "속은 촉촉함"]),
            ("담백하고 쫄깃한 식감", ["담백함", "쫄깃한 식감"]),
            (
                "바삭한 캐러멜 윗면과 부드러운 크림",
                ["바삭한 캐러멜 윗면", "부드러운 크림"],
            ),
        )
        for raw, expected in examples:
            with self.subTest(raw=raw):
                self.assertEqual(split_distinct_feature_facts(raw), expected)

    def test_rejects_blank_api_key_before_calling_client(self) -> None:
        interactions = FakeInteractions(self.output_text)
        client = SimpleNamespace(interactions=interactions)

        with self.assertRaises(CopyGenerationError):
            generate_ad_copies(
                self.ad_input,
                model_id="gemini-3.5-flash-lite",
                api_key=" ",
                client=client,
            )
        self.assertIsNone(interactions.request)

    def test_rejects_unknown_model(self) -> None:
        with self.assertRaises(CopyGenerationError):
            generate_ad_copies(
                self.ad_input,
                model_id="unknown-model",
                api_key="test-key",
            )


if __name__ == "__main__":
    unittest.main()
