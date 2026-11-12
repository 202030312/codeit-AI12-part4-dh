import json
from types import SimpleNamespace
import unittest

from src.features.ad_copy.validation import (
    build_fallback_copies,
    generate_safe_ad_copies,
    validate_ad_copies,
)


VALID_COPIES = [
    {
        "headline": "딸기 풍미를 담은 한 잔",
        "body": "상큼한 딸기 풍미가 자연스럽게 느껴져요.",
        "hashtags": ["#테스트카페", "#딸기라떼"],
    },
    {
        "headline": "상큼한 풍미가 깨우는 오후",
        "body": "상큼한 딸기 풍미가 입안에 남아요.",
        "hashtags": ["#딸기", "#딸기라떼"],
    },
    {
        "headline": "상큼한 딸기 향의 여운",
        "body": "상큼한 딸기 풍미가 그대로 전해져요.",
        "hashtags": ["#테스트카페"],
    },
]


def _model_payload(copies):
    """Gemini가 반환하는 headline/body 전용 응답으로 변환한다."""
    return [
        {"headline": copy["headline"], "body": copy["body"]}
        for copy in copies
    ]


def _expected_generated(copies):
    """앱이 상품명·매장명으로 해시태그를 붙인 최종 문구이다."""
    return [
        {
            "headline": copy["headline"],
            "body": copy["body"],
            "hashtags": ["#딸기라떼", "#테스트카페"],
        }
        for copy in copies
    ]


class SequenceInteractions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(output_text=response)


class CopyQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ad_input = {
            "schema_version": "1.1",
            "image": {
                "file_name": "strawberry_latte_clean.png",
                "content_type": "image/png",
            },
            "store": {"name": "테스트 카페"},
            "product": {
                "name": "딸기 라떼",
                "base_price": "6,000원",
                "sale_price": None,
                "features": "상큼한 딸기 풍미",
            },
            "campaign": {
                "promotion": None,
                "tone": "따뜻하고 감성적",
                "copy_request": None,
            },
        }

    def test_accepts_copies_grounded_in_input(self) -> None:
        self.assertEqual(validate_ad_copies(VALID_COPIES, self.ad_input), [])

    def test_rejects_unsupported_claims_numbers_and_hashtags(self) -> None:
        unsafe = [dict(copy) for copy in VALID_COPIES]
        unsafe[0] = {
            "headline": "합리적인 가격의 건강한 한 잔",
            "body": "피로를 달래는 딸기 라떼를 20% 할인으로 만나보세요.",
            "hashtags": ["#디저트"],
        }

        issues = " ".join(validate_ad_copies(unsafe, self.ad_input))

        self.assertIn("20", issues)
        self.assertIn("합리적", issues)
        self.assertIn("건강", issues)
        self.assertIn("피로", issues)
        self.assertIn("할인", issues)
        self.assertIn("#디저트", issues)

    def test_copy_request_is_not_treated_as_product_fact_evidence(self) -> None:
        self.ad_input["campaign"]["copy_request"] = (
            "건강 효능과 달콤한 맛을 강조해 주세요."
        )
        requested_but_unsupported = [dict(copy) for copy in VALID_COPIES]
        requested_but_unsupported[0] = {
            "headline": "건강하고 달콤한 한 잔",
            "body": "달콤한 맛이 그대로 느껴져요.",
            "hashtags": ["#딸기라떼", "#테스트카페"],
        }

        issues = " ".join(
            validate_ad_copies(requested_but_unsupported, self.ad_input)
        )

        self.assertIn("건강", issues)
        self.assertIn("달콤한", issues)

    def test_rejects_headline_and_body_that_repeat_one_of_two_facts(self) -> None:
        self.ad_input["product"]["features"] = "담백하고 쫄깃한 식감"
        repeated = [
            {
                "headline": "담백함이 느껴져요",
                "body": "담백한 맛을 느낄 수 있어요.",
                "hashtags": ["#딸기라떼", "#테스트카페"],
            }
        ]

        issues = " ".join(validate_ad_copies(repeated, self.ad_input))

        self.assertIn("서로 다른 입력 특징", issues)

    def test_rejects_copy_that_will_not_fit_banner_or_repeats_labels(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "테스트 카페에서 평일 오전 500원 할인으로 즐기는 딸기 라떼",
            "body": "딸기 라떼 " * 50,
            "hashtags": ["#딸기라떼"],
        }
        self.ad_input["campaign"]["promotion"] = "평일 오전 500원 할인"

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("20자", issues)
        self.assertIn("34자", issues)
        self.assertIn("매장명", issues)
        self.assertIn("행사 배지", issues)

    def test_product_name_repetition_is_not_a_hard_failure(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "브라우니 아이스크림",
            "body": "달콤한 조합의 깊은 풍미를 즐겨보세요.",
            "hashtags": ["#브라우니아이스크림"],
        }
        self.ad_input["product"] = {
            "name": "브라우니 아이스크림",
            "base_price": "8,000원",
            "sale_price": None,
            "features": "브라우니와 아이스크림의 조합",
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertNotIn("상품명이 반복", issues)

    def test_rejects_headline_punctuation_and_unsupported_superlative(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "극상의 달콤함!",
            "body": "상큼한 딸기 풍미를 천천히 즐겨보세요.",
            "hashtags": ["#딸기"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("문장부호", issues)
        self.assertIn("극상", issues)

    def test_accepts_hashtag_after_removing_korean_particle(self) -> None:
        copies = [
            {
                "headline": "부드러운 시나몬 향의 여운",
                "body": "시나몬 향과 부드러운 빵 결이 느껴져요.",
                "hashtags": ["#시나몬향"],
            }
        ]
        self.ad_input["product"] = {
            "name": "시나몬롤",
            "base_price": "5,000원",
            "sale_price": None,
            "features": "시나몬 향과 부드러운 빵 결",
        }

        self.assertEqual(validate_ad_copies(copies, self.ad_input), [])

    def test_retries_invalid_copy_and_returns_corrected_result(self) -> None:
        unsafe_copy = {
            "headline": "합리적인 가격",
            "body": "피로를 달래는 특별한 풍미를 느껴보세요.",
            "hashtags": ["#디저트"],
        }
        unsafe = [dict(unsafe_copy) for _ in range(3)]
        interactions = SequenceInteractions(
            [
                json.dumps({"copies": _model_payload(unsafe)}, ensure_ascii=False),
                json.dumps(
                    {"copies": _model_payload(VALID_COPIES)},
                    ensure_ascii=False,
                ),
            ]
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_safe_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result.status, "retried")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.copies, _expected_generated(VALID_COPIES))
        self.assertTrue(result.validation_issues)
        self.assertIn("이전 응답", interactions.requests[1]["input"])

    def test_uses_fallback_after_two_model_errors(self) -> None:
        interactions = SequenceInteractions(
            [RuntimeError("temporary error"), RuntimeError("temporary error")]
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_safe_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.copies), 1)
        self.assertEqual(validate_ad_copies(result.copies, self.ad_input), [])

    def test_fallback_uses_only_input_values(self) -> None:
        copies = build_fallback_copies(self.ad_input)

        self.assertEqual(len(copies), 1)
        self.assertIn("상큼한 딸기 풍미", copies[0]["headline"])
        self.assertEqual(copies[0]["body"], "")
        self.assertEqual(validate_ad_copies(copies, self.ad_input), [])

    def test_rejects_inconsistent_body_ending_and_repeated_labels(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "따뜻한 감성을 담은 한 잔",
            "body": "테스트 카페의 딸기 라떼는 6,000원입니다.",
            "hashtags": ["#테스트카페", "#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("해요체", issues)
        self.assertIn("매장명이 반복", issues)
        self.assertNotIn("상품명이 반복", issues)
        self.assertIn("가격이 반복", issues)

    def test_requires_period_after_polite_ending(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0]["body"] = invalid[0]["body"].removesuffix(".")

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("해요체", issues)

    def test_accepts_natural_polite_body_ending(self) -> None:
        copies = [dict(copy) for copy in VALID_COPIES]
        copies[0]["body"] = "생딸기의 산뜻한 풍미가 은은하게 느껴져요."

        issues = " ".join(validate_ad_copies(copies, self.ad_input))

        self.assertNotIn("해요체", issues)

    def test_rejects_body_that_requests_consumer_action(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0]["body"] = "상큼한 딸기 풍미를 직접 느껴보세요."

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("행동을 요청", issues)

    def test_generic_body_is_soft_but_action_request_is_still_rejected(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0]["body"] = "풍미를 즐겨보세요."

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertNotIn("지나치게 추상적", issues)
        self.assertIn("행동을 요청", issues)

    def test_rejects_known_grammar_and_redundancy_errors(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "매력적인 식감의 디저트에요",
            "body": "달콤한 맛을 맛보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("디저트에요", issues)
        self.assertIn("맛을 맛보세요", issues)

    def test_rejects_headline_that_ends_without_a_predicate(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0]["headline"] = "깔끔한 커피 풍미가 매력"

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("서술어 없이", issues)

    def test_rejects_meta_wording_and_incomplete_pairing(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "토닉의 조합",
            "body": "카다멈 향이 특징인 번을 만나보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("두 대상", issues)
        self.assertIn("특징인", issues)

    def test_rejects_ingredients_described_as_two_menus(self) -> None:
        ad_input = {
            **self.ad_input,
            "product": {
                **self.ad_input["product"],
                "name": "브라우니 아이스크림",
                "features": "브라우니와 아이스크림의 조합",
            },
        }
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "꾸덕함과 차가움의 조화",
            "body": "두 메뉴의 조합을 천천히 즐겨보세요.",
            "hashtags": ["#브라우니아이스크림"],
        }

        issues = " ".join(validate_ad_copies(invalid, ad_input))

        self.assertIn("두 가지 메뉴", issues)

    def test_allows_input_ingredient_names_without_full_product_label(self) -> None:
        ad_input = {
            **self.ad_input,
            "product": {
                **self.ad_input["product"],
                "name": "브라우니 아이스크림",
                "features": "꾸덕한 브라우니와 차가운 아이스크림의 조합",
            },
        }
        copies = [dict(copy) for copy in VALID_COPIES]
        copies[0] = {
            "headline": "꾸덕함과 차가움의 조화",
            "body": "브라우니와 아이스크림이 조화로워요.",
            "hashtags": ["#브라우니아이스크림"],
        }

        issues = " ".join(validate_ad_copies(copies, ad_input))

        self.assertNotIn("상품명이 반복", issues)

    def test_rejects_unsupported_qualifier_and_repeated_feature_noun(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "독특한 향의 순간",
            "body": "향이 퍼지는 카다멈 향을 즐겨보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("독특", issues)
        self.assertIn("핵심 단어 '향'", issues)

    def test_rejects_tone_value_used_as_product_copy(self) -> None:
        self.ad_input["campaign"]["tone"] = "친근한"
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "친근한 딸기 풍미",
            "body": "달콤한 향을 천천히 즐겨보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("광고 분위기 값", issues)

    def test_fallback_rewrites_feature_note_as_natural_copy(self) -> None:
        ad_input = {
            **self.ad_input,
            "product": {
                **self.ad_input["product"],
                "name": "카다멈 번",
                "features": "카다멈 향이 특징인 번",
            },
        }

        copies = build_fallback_copies(ad_input)

        self.assertTrue(all("특징인" not in copy["headline"] for copy in copies))
        self.assertTrue(all("카다멈 향" in copy["headline"] for copy in copies))
        self.assertTrue(all(copy["body"] == "" for copy in copies))
        self.assertEqual(validate_ad_copies(copies, ad_input), [])

    def test_generic_headline_is_soft_but_unsupported_claim_is_rejected(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "활기찬 오후를 채우는 맛",
            "body": "오늘 구운 풍미를 천천히 즐겨보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertNotIn("지나치게 일반적", issues)
        self.assertIn("오늘 구운", issues)

    def test_rejects_copy_without_actual_menu_feature(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "기분 좋은 순간을 채우는 시간",
            "body": "특별한 매력을 편안하게 직접 즐겨보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("실제 메뉴 특징", issues)

    def test_rejects_unsupported_sensory_inference(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0] = {
            "headline": "청량감이 살아나는 순간",
            "body": "상큼한 딸기 풍미를 시원하게 즐겨보세요.",
            "hashtags": ["#딸기라떼"],
        }

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("청량감", issues)
        self.assertIn("시원한", issues)

    def test_rejects_incomplete_headline(self) -> None:
        invalid = [dict(copy) for copy in VALID_COPIES]
        invalid[0]["headline"] = "입안 가득 전해지는"

        issues = " ".join(validate_ad_copies(invalid, self.ad_input))

        self.assertIn("완결되지 않은", issues)

    def test_keeps_valid_model_candidates_when_others_fail(self) -> None:
        partly_invalid = [dict(copy) for copy in VALID_COPIES]
        partly_invalid[0] = {
            "headline": "시원하게 마시는",
            "body": "상큼한 딸기 풍미를 편안하게 느껴보세요.",
            "hashtags": ["#딸기라떼"],
        }
        interactions = SequenceInteractions(
            [
                json.dumps(
                    {"copies": _model_payload(partly_invalid)},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"copies": _model_payload(partly_invalid)},
                    ensure_ascii=False,
                ),
            ]
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_safe_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result.status, "generated")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.model_copy_count, 2)
        self.assertEqual(result.fallback_copy_count, 0)
        self.assertEqual(validate_ad_copies(result.copies, self.ad_input), [])

    def test_accepts_empty_body_when_headline_contains_feature(self) -> None:
        copies = [
            {
                "headline": "상큼한 딸기 풍미",
                "body": "",
                "hashtags": ["#딸기라떼"],
            }
        ]

        self.assertEqual(validate_ad_copies(copies, self.ad_input), [])

    def test_accepts_relation_synonym_when_both_operands_are_preserved(self) -> None:
        self.ad_input["product"] = {
            "name": "조합 디저트",
            "base_price": "8,000원",
            "sale_price": None,
            "features": "재료하나와 재료둘의 조합",
        }
        copies = [
            {
                "headline": "재료하나와 재료둘의 조화",
                "body": "",
                "hashtags": ["#조합디저트"],
            }
        ]

        self.assertEqual(validate_ad_copies(copies, self.ad_input), [])

    def test_accepts_relation_as_meeting_when_both_operands_are_preserved(self) -> None:
        self.ad_input["product"] = {
            "name": "조합 디저트",
            "base_price": "8,000원",
            "sale_price": None,
            "features": "재료하나와 재료둘의 조합",
        }
        copies = [
            {
                "headline": "재료하나와 재료둘의 만남",
                "body": "",
                "hashtags": ["#조합디저트"],
            }
        ]

        self.assertEqual(validate_ad_copies(copies, self.ad_input), [])

    def test_rejects_relation_synonym_when_an_operand_is_missing(self) -> None:
        self.ad_input["product"] = {
            "name": "조합 디저트",
            "base_price": "8,000원",
            "sale_price": None,
            "features": "재료하나와 재료둘의 조합",
        }
        copies = [
            {
                "headline": "재료하나의 조화 그대로",
                "body": "",
                "hashtags": ["#조합디저트"],
            }
        ]

        issues = " ".join(validate_ad_copies(copies, self.ad_input))
        self.assertIn("실제 메뉴 특징", issues)

    def test_drops_invalid_optional_body_when_headline_is_safe(self) -> None:
        body_only_invalid = [
            {
                "headline": "상큼한 딸기 풍미",
                "body": "은은한 향을 느껴져요.",
                "hashtags": ["#딸기라떼"],
            }
        ]
        interactions = SequenceInteractions(
            [
                json.dumps(
                    {"copies": _model_payload(body_only_invalid)},
                    ensure_ascii=False,
                )
            ]
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_safe_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result.status, "generated")
        self.assertEqual(result.copies[0]["body"], "")
        self.assertIn("본문", result.notice or "")

    def test_fallback_splits_two_features_between_headline_and_body(self) -> None:
        self.ad_input["product"]["features"] = "바삭한 겉면, 촉촉한 속"

        copies = build_fallback_copies(self.ad_input)

        self.assertEqual(len(copies), 1)
        self.assertIn("바삭한 겉면", copies[0]["headline"])
        self.assertIn("촉촉한 속", copies[0]["body"])

    def test_fallback_splits_two_facts_inside_one_feature_item(self) -> None:
        self.ad_input["product"]["features"] = "진한 향과 부드러운 식감"

        copies = build_fallback_copies(self.ad_input)

        self.assertIn("진한 향", copies[0]["headline"])
        self.assertIn("부드러운 식감", copies[0]["body"])

    def test_fallback_turns_connected_adjective_into_natural_headline(self) -> None:
        self.ad_input["product"]["features"] = "겉은 바삭하고 속은 촉촉한 식감"
        crisp_copy = build_fallback_copies(self.ad_input)[0]
        self.assertEqual(crisp_copy["headline"], "겉은 바삭해요")

        self.ad_input["product"]["features"] = "담백하고 쫄깃한 식감"
        mild_copy = build_fallback_copies(self.ad_input)[0]
        self.assertEqual(mild_copy["headline"], "담백함이 느껴져요")

    def test_repairs_single_candidate_that_repeats_full_product_name(self) -> None:
        repeated = [
            {
                "headline": "딸기 라떼의 상큼한 풍미",
                "body": "",
                "hashtags": ["#ignored"],
            }
        ]
        interactions = SequenceInteractions(
            [
                json.dumps(
                    {"copies": _model_payload(repeated)},
                    ensure_ascii=False,
                )
            ]
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_safe_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result.status, "mixed")
        self.assertEqual(result.copies[0]["headline"], "상큼한 딸기 풍미")
        self.assertNotIn("딸기 라떼", result.copies[0]["headline"])
        self.assertIn("상품명이 반복된 제목", result.notice or "")

    def test_completes_blank_body_when_input_has_two_distinct_facts(self) -> None:
        self.ad_input["product"]["features"] = "진한 향과 부드러운 식감"
        generated = [
            {
                "headline": "진한 향과 부드러운 식감",
                "body": "",
                "hashtags": ["#ignored"],
            }
        ]
        interactions = SequenceInteractions(
            [
                json.dumps(
                    {"copies": _model_payload(generated)},
                    ensure_ascii=False,
                )
            ]
        )
        client = SimpleNamespace(interactions=interactions)

        result = generate_safe_ad_copies(
            self.ad_input,
            model_id="gemini-3.5-flash-lite",
            api_key="test-key",
            client=client,
        )

        self.assertEqual(result.status, "mixed")
        self.assertIn("진한 향", result.copies[0]["headline"])
        self.assertIn("부드러운 식감", result.copies[0]["body"])
        self.assertEqual(result.model_copy_count, 0)
        self.assertEqual(result.fallback_copy_count, 1)
        self.assertIn("제목과 본문으로 분리", result.notice or "")


if __name__ == "__main__":
    unittest.main()
