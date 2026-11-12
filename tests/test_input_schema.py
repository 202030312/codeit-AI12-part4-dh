import json
import unittest

from src.shared.input_schema import build_ad_input


class BuildAdInputTests(unittest.TestCase):
    def test_builds_fixed_json_structure_and_trims_text(self) -> None:
        result = build_ad_input(
            image_filename=" strawberry.png ",
            image_content_type=" image/png ",
            store_name=" 오늘의 작은 카페 ",
            product_name=" 딸기 크림 라떼 ",
            base_price=" 6,500원 ",
            sale_price=" 5,500원 ",
            features=" 생딸기 사용 ",
            promotion=" 주말 10% 할인 ",
            tone=" 감성적인 ",
            copy_request=" 딸기 풍미를 짧게 강조해 주세요. ",
            call_to_action=" 매장에서 만나보세요. ",
        )

        self.assertEqual(
            result,
            {
                "schema_version": "1.2",
                "image": {
                    "file_name": "strawberry.png",
                    "content_type": "image/png",
                },
                "store": {"name": "오늘의 작은 카페"},
                "product": {
                    "name": "딸기 크림 라떼",
                    "base_price": "6,500원",
                    "sale_price": "5,500원",
                    "features": "생딸기 사용",
                },
                "campaign": {
                    "promotion": "주말 10% 할인",
                    "tone": "감성적인",
                    "copy_request": "딸기 풍미를 짧게 강조해 주세요.",
                    "call_to_action": "매장에서 만나보세요.",
                },
            },
        )
        self.assertEqual(json.loads(json.dumps(result, ensure_ascii=False)), result)

    def test_converts_blank_optional_fields_to_null(self) -> None:
        result = build_ad_input(
            image_filename="product.jpg",
            image_content_type=None,
            store_name=" ",
            product_name="크루아상",
            base_price="",
            sale_price="",
            features="바삭한 식감",
            promotion="\t",
            tone="고급스러운",
        )

        self.assertIsNone(result["image"]["content_type"])
        self.assertIsNone(result["store"]["name"])
        self.assertIsNone(result["product"]["base_price"])
        self.assertIsNone(result["product"]["sale_price"])
        self.assertEqual(result["product"]["features"], "바삭한 식감")
        self.assertIsNone(result["campaign"]["promotion"])
        self.assertIsNone(result["campaign"]["copy_request"])
        self.assertIsNone(result["campaign"]["call_to_action"])

    def test_rejects_missing_required_fields(self) -> None:
        common_values = {
            "image_filename": "product.jpg",
            "image_content_type": "image/jpeg",
            "store_name": "카페",
            "product_name": "라떼",
            "base_price": None,
            "sale_price": None,
            "features": "담백한 풍미",
            "promotion": None,
            "tone": "친근한",
        }

        for field_name in ("image_filename", "product_name", "features", "tone"):
            with self.subTest(field_name=field_name):
                values = common_values | {field_name: " "}
                with self.assertRaises(ValueError):
                    build_ad_input(**values)

    def test_rejects_sale_price_without_base_price(self) -> None:
        with self.assertRaisesRegex(ValueError, "기본 가격"):
            build_ad_input(
                image_filename="product.jpg",
                image_content_type="image/jpeg",
                store_name="카페",
                product_name="라떼",
                base_price=None,
                sale_price="5,000원",
                features="바삭한 식감",
                promotion=None,
                tone="친근한",
            )

    def test_accepts_legacy_price_as_base_price(self) -> None:
        result = build_ad_input(
            image_filename="product.jpg",
            image_content_type="image/jpeg",
            store_name="카페",
            product_name="라떼",
            price="6,000원",
            features="깔끔한 커피 풍미",
            promotion=None,
            tone="친근한",
        )

        self.assertEqual(result["product"]["base_price"], "6,000원")
        self.assertIsNone(result["product"]["sale_price"])


if __name__ == "__main__":
    unittest.main()
