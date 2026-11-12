from __future__ import annotations

from src.features.ad_copy.workflow import (
    COPY_PROMPT_VERSION,
    _fallback_options,
    build_banner_copy_prompt,
    collect_banner_copy_issues,
)


def _input(
    name: str,
    features: str,
    *,
    tone: str = "따뜻하고 감성적으로",
    request: str = "출근길에 가볍게 즐기는 간식 상황을 강조해 주세요.",
) -> dict:
    return {
        "product": {"name": name, "features": features},
        "campaign": {
            "tone": tone,
            "copy_request": request,
            "promotion": None,
            "promotion_copy_mode": "badge_only",
            "generation_mode": "headline_and_subcopy",
            "verified_visual_facts": [],
        },
    }


def test_prompt_requires_product_fact_request_context_and_distinct_roles() -> None:
    prompt = build_banner_copy_prompt(
        _input("크림 브리오슈 번", "부드러운 브리오슈와 크림의 조합")
    )

    assert COPY_PROMPT_VERSION == "gpt-copy-v1.10.18-request-aware-partial-recovery"
    assert "추가 요청에 타깃·이용 상황·강조 메시지가 있으면" in prompt
    assert "같은 의미의 자연스러운 변형" in prompt
    assert '"candidate_roles"' in prompt


def test_semantic_fact_paraphrase_is_accepted_across_headline_and_body() -> None:
    ad_input = _input(
        "크림 브리오슈 번",
        "부드러운 브리오슈와 크림의 조합",
    )
    copies = [
        {
            "headline": "출근길의 가벼운 한입",
            "body": "부드러운 브리오슈와 크림",
            "hashtags": [],
        },
        {
            "headline": "가볍게 즐기는 간식 시간",
            "body": "브리오슈와 크림의 부드러운 조화",
            "hashtags": [],
        },
    ]

    assert collect_banner_copy_issues(copies, ad_input) == []


def test_context_hook_may_use_body_for_single_verified_fact() -> None:
    ad_input = _input(
        "흑임자 라떼",
        "고소한 흑임자 풍미",
        tone="차분하고 고급스럽게",
        request="잠시 쉬어 가는 시간의 분위기를 강조해 주세요.",
    )
    copies = [
        {
            "headline": "잠시 쉬어 가는 시간",
            "body": "고소한 흑임자 풍미",
            "hashtags": [],
        },
        {
            "headline": "천천히 음미하는 시간",
            "body": "고소한 흑임자 풍미",
            "hashtags": [],
        },
    ]

    assert collect_banner_copy_issues(copies, ad_input) == []


def test_incomplete_particle_ending_is_rejected() -> None:
    ad_input = _input(
        "소금빵",
        "고소한 버터 풍미와 담백한 소금 맛",
        request="퇴근길 간식 상황을 강조해 주세요.",
    )
    copies = [
        {
            "headline": "고소한 버터 풍미",
            "body": "담백한 소금 맛으로 퇴근길 간식에",
            "hashtags": [],
        },
        {
            "headline": "퇴근길에 더하는 작은 여유",
            "body": "고소한 버터 풍미 · 담백한 소금 맛",
            "hashtags": [],
        },
    ]

    codes = {issue.code for issue in collect_banner_copy_issues(copies, ad_input)}
    assert "INCOMPLETE_EXPRESSION" in codes


def test_fallback_uses_allowed_request_context_instead_of_only_repeating_fact() -> None:
    ad_input = _input(
        "흑임자 라떼",
        "고소한 흑임자 풍미",
        tone="차분하고 고급스럽게",
        request="잠시 쉬어 가는 시간의 분위기를 강조해 주세요.",
    )

    copies = _fallback_options(ad_input)

    assert len(copies) == 2
    assert any("쉬어" in copy["headline"] or "음미" in copy["headline"] for copy in copies)
    assert all("흑임자" in copy["body"] for copy in copies)
