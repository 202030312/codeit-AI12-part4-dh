"""사용자 요청 맥락과 의미 기반 사실 검증을 강화한 문구 파이프라인 v1.10.18."""

from __future__ import annotations

import json
import re
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    build_deterministic_hashtags,
)
from src.features.ad_copy.validation import _fact_is_reflected
from src.features.ad_copy.promotion import (
    BADGE_ONLY,
    headline_is_exact_promotion,
    headline_mentions_promotion,
    normalize_promotion_copy_mode,
)
from src.features.ad_copy.models import (
    BannerCopyGenerationResult,
    CopyInputError,
    CopyValidationIssue,
    analyze_copy_input,
)
import src.features.ad_copy.internal.product_rules as fact_workflow
import src.features.ad_copy.internal.caption_rules as natural_workflow
import src.features.ad_copy.recovery as previous


COPY_PROMPT_VERSION = "gpt-copy-v1.10.18-request-aware-partial-recovery"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = previous.INTERNAL_CANDIDATE_COUNT
FINAL_OPTION_COUNT = previous.FINAL_OPTION_COUNT

_DANGLING_ENDING_PATTERN = re.compile(
    r"(?:으로|에서|에게|까지|부터|위한|담아|더해|곁들여|향해|함께|[에와과을를의])$"
)


def _provider_copy_prompt_version(provider: str) -> str:
    return f"{provider}_copy_v1.10.18-request-aware-partial-recovery"


def _facts(ad_input: dict[str, Any]) -> list[str]:
    return previous._facts(ad_input)


def _normalize(value: str) -> str:
    return previous._normalize(value)


def _fact_sets(copy: AdCopy, facts: list[str]) -> tuple[set[int], set[int]]:
    headline_facts = {
        index
        for index, fact in enumerate(facts)
        if _fact_is_reflected(copy["headline"], fact)
    }
    body_facts = {
        index
        for index, fact in enumerate(facts)
        if _fact_is_reflected(copy["body"], fact)
    }
    return headline_facts, body_facts


def _has_incomplete_ending(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).rstrip(" .!?。！？")
    if not normalized:
        return False
    return bool(_DANGLING_ENDING_PATTERN.search(normalized))


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    """제목과 본문 전체에서 의미가 보존되면 허용하고 미완성 표현은 차단한다."""

    prepared = natural_workflow._copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    product = prepared.get("product", {})
    menu_name = str(product.get("name") or "").strip()
    feature = " ".join(str(product.get("features") or "").split())
    facts = _facts(prepared)
    campaign = prepared.get("campaign", {})
    promotion = str(campaign.get("promotion") or "").strip()
    promotion_mode = normalize_promotion_copy_mode(
        campaign.get("promotion_copy_mode")
    )
    issues: list[CopyValidationIssue] = []

    if len(copies) != FINAL_OPTION_COUNT:
        issues.append(
            CopyValidationIssue(
                "OPTION_COUNT", f"최종 문구는 {FINAL_OPTION_COUNT}개여야 합니다."
            )
        )

    seen: set[tuple[str, str]] = set()
    for index, copy in enumerate(copies, start=1):
        headline = " ".join(str(copy.get("headline") or "").split())
        body = " ".join(str(copy.get("body") or "").split())
        if not headline:
            issues.append(
                CopyValidationIssue(
                    "MISSING_HEADLINE",
                    f"선택지 {index} 제목이 비어 있습니다.",
                    index,
                    "headline",
                )
            )
        elif len(headline) > profile.limits.headline:
            issues.append(
                CopyValidationIssue(
                    "HEADLINE_LENGTH",
                    f"선택지 {index} 제목이 길이 제한을 넘습니다.",
                    index,
                    "headline",
                )
            )
        if len(body) > profile.limits.subcopy:
            issues.append(
                CopyValidationIssue(
                    "SUBCOPY_LENGTH",
                    f"선택지 {index} 보조 문구가 길이 제한을 넘습니다.",
                    index,
                    "subcopy",
                )
            )
        if natural_workflow.headline_repeats_menu_name(headline, menu_name):
            issues.append(
                CopyValidationIssue(
                    "MENU_NAME_REPEAT",
                    f"선택지 {index} 제목이 별도 메뉴명을 반복합니다.",
                    index,
                    "headline",
                )
            )

        for field, value in (("headline", headline), ("subcopy", body)):
            if natural_workflow.has_caption_leak(value):
                issues.append(
                    CopyValidationIssue(
                        "IMAGE_CAPTION_LEAK",
                        f"선택지 {index} {field}에 사진 판독형 표현이 있습니다.",
                        index,
                        field,
                    )
                )
            if natural_workflow.has_awkward_expression(value) or _has_incomplete_ending(
                value
            ):
                issues.append(
                    CopyValidationIssue(
                        "INCOMPLETE_EXPRESSION",
                        f"선택지 {index} {field}이 완결되지 않은 표현입니다.",
                        index,
                        field,
                    )
                )

        headline_facts, body_facts = _fact_sets(
            {"headline": headline, "body": body, "hashtags": []}, facts
        )
        reflected_facts = headline_facts | body_facts
        if feature and not reflected_facts:
            issues.append(
                CopyValidationIssue(
                    "PRODUCT_FACT_NOT_PRESERVED",
                    f"선택지 {index}에 확인된 상품 특징이 보존되지 않았습니다.",
                    index,
                )
            )
        if len(facts) >= 2:
            missing = set(range(len(facts))) - reflected_facts
            if missing:
                issues.append(
                    CopyValidationIssue(
                        "PAIR_MISSING_FACTS",
                        f"선택지 {index}에 독립적인 상품 특징이 빠졌습니다.",
                        index,
                    )
                )
        if headline_facts & body_facts:
            issues.append(
                CopyValidationIssue(
                    "SAME_CLAIM_REPEATED",
                    f"선택지 {index}의 제목과 보조 문구가 같은 특징을 반복합니다.",
                    index,
                )
            )
        if len(facts) == 1 and body and headline_facts and body_facts:
            issues.append(
                CopyValidationIssue(
                    "UNNECESSARY_SUBCOPY",
                    f"선택지 {index}이 하나의 특징을 보조 문구로 반복합니다.",
                    index,
                    "subcopy",
                )
            )
        combined = f"{headline} {body}".strip()
        if natural_workflow._unsupported_sensory_claim(combined, feature):
            issues.append(
                CopyValidationIssue(
                    "UNSUPPORTED_SENSORY_CLAIM",
                    f"선택지 {index}에 입력에 없는 감각 표현이 있습니다.",
                    index,
                )
            )
        if promotion_mode == BADGE_ONLY and (
            headline_mentions_promotion(headline)
            or headline_mentions_promotion(body)
        ):
            issues.append(
                CopyValidationIssue(
                    "PROMOTION_IN_COPY",
                    f"선택지 {index}에 배지 전용 프로모션이 반복됩니다.",
                    index,
                )
            )
        if (
            promotion
            and promotion_mode != BADGE_ONLY
            and headline_mentions_promotion(headline)
            and not headline_is_exact_promotion(headline, promotion)
        ):
            issues.append(
                CopyValidationIssue(
                    "PROMOTION_REWRITTEN",
                    f"선택지 {index} 프로모션 조건이 원문과 다릅니다.",
                    index,
                    "headline",
                )
            )

        key = (_normalize(headline), _normalize(body))
        if key in seen:
            issues.append(
                CopyValidationIssue(
                    "DUPLICATE_COPY",
                    f"선택지 {index}가 다른 선택지와 중복됩니다.",
                    index,
                )
            )
        seen.add(key)
    return issues


def build_banner_copy_prompt(
    ad_input: dict[str, Any],
    *,
    retry_context: dict[str, Any] | None = None,
) -> str:
    prepared = natural_workflow._copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    product = prepared.get("product", {})
    campaign = prepared.get("campaign", {})
    facts = _facts(prepared)
    payload: dict[str, Any] = {
        "prompt_version": COPY_PROMPT_VERSION,
        "product_name_rendered_separately": str(product.get("name") or ""),
        "verified_product_facts": facts,
        "copy_eligible_visual_facts": list(
            campaign.get("copy_eligible_visual_facts") or []
        ),
        "requested_tone": str(campaign.get("tone") or ""),
        "additional_request": str(campaign.get("copy_request") or ""),
        "generation_mode": profile.generation_mode,
        "headline_limit": profile.limits.headline,
        "subcopy_limit": profile.limits.subcopy,
        "candidate_count": INTERNAL_CANDIDATE_COUNT,
        "message_cardinality": len(facts),
        "rules": [
            "각 후보는 제목과 body 전체에서 확인된 상품 특징을 하나 이상 의미상 보존한다.",
            "상품 특징은 원문 복사가 아니라 같은 의미의 자연스러운 변형으로 표현해도 된다.",
            "추가 요청에 타깃·이용 상황·강조 메시지가 있으면 각 후보에 그중 하나 이상을 반영한다.",
            "스타일과 톤은 어휘와 문장 리듬만 조절하며 새로운 맛·재료·효능의 근거로 사용하지 않는다.",
            "상황 훅을 제목에 쓴 경우 확인된 상품 특징은 body에 배치할 수 있다.",
            "제목과 body가 같은 사실을 반복하지 않는다.",
            "조사·연결어·관형형으로 끝나는 미완성 표현을 만들지 않는다.",
            "상품명 전체는 별도 슬롯에 표시되므로 제목에 반복하지 않는다.",
            "4개 후보는 시작 어절과 관점을 달리하고 동일 문장을 복제하지 않는다.",
            "입력에 없는 맛·향·식감·재료·효과·상황을 만들지 않는다.",
        ],
        "candidate_roles": [
            "상품 특징 중심의 간결한 표현",
            "요청 상황 또는 타깃을 활용한 훅과 상품 근거",
            "선택한 톤을 분명히 살린 표현",
            "같은 사실을 다른 문장 구조로 표현한 대안",
        ],
    }
    if retry_context:
        payload["retry_context"] = retry_context
    return json.dumps(payload, ensure_ascii=False, indent=2)


_GENERATION_SCHEMA = previous._GENERATION_SCHEMA

_SYSTEM_INSTRUCTIONS = """
너는 카페·디저트·베이커리 소상공인을 위한 한국어 광고 카피 편집자다.
상품명은 별도 영역에 표시된다. 각 후보는 확인된 상품 특징을 의미상 보존하면서,
사용자가 적은 타깃·이용 상황·강조 메시지가 있으면 그 맥락을 실제 문구에 반영한다.
스타일과 톤은 표현 방식일 뿐 새로운 상품 사실의 근거가 아니다. 제목과 보조 문구는
같은 내용을 반복하지 않고, 조사나 연결어로 끝나는 미완성 문장을 쓰지 않는다.
4개 후보는 서로 다른 시작과 관점을 사용한다. JSON 스키마와 일치하는 결과만 반환한다.
""".strip()

_JUDGE_INSTRUCTIONS = """
너는 한국어 광고 카피 QC 감수자다. 각 후보가 자연스럽고 짧은 광고 문구로 쓸 수 있는지,
확인된 상품 특징을 벗어나지 않는지, 사용자가 요청한 타깃·상황·강조 메시지와 선택한
톤을 실제로 반영했는지 판정한다. 요청 맥락이 있는데 특징만 그대로 옮긴 설명형 문구는
ad_value를 2 이하로 평가한다. 조사·연결어로 끝나는 미완성 표현도 naturalness를 3 이하로
평가한다. 상품명 반복과 사진 판독형 문장은 탈락으로 판정한다. JSON만 반환한다.
""".strip()


def _candidate(item: dict[str, Any], hashtags: list[str]) -> AdCopy:
    return previous._candidate(item, hashtags)


def _candidate_issues(
    copy: AdCopy, ad_input: dict[str, Any]
) -> list[CopyValidationIssue]:
    placeholder: AdCopy = {
        "headline": "임시 후보 구분 문구",
        "body": "",
        "hashtags": [],
    }
    return [
        issue
        for issue in collect_banner_copy_issues([copy, placeholder], ad_input)
        if issue.option_index == 1
    ]


def _request_hooks(ad_input: dict[str, Any]) -> list[str]:
    campaign = ad_input.get("campaign", {})
    context = " ".join(
        str(value or "")
        for value in (campaign.get("tone"), campaign.get("copy_request"))
    )
    profile = analyze_copy_input(ad_input, image_provided=False)
    menu_name = str(ad_input.get("product", {}).get("name") or "")
    values: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.split())
        if (
            4 <= len(cleaned) <= profile.limits.headline
            and not natural_workflow.headline_repeats_menu_name(cleaned, menu_name)
            and not natural_workflow.has_caption_leak(cleaned)
            and not natural_workflow.has_awkward_expression(cleaned)
            and not _has_incomplete_ending(cleaned)
            and cleaned not in values
        ):
            values.append(cleaned)

    patterns = (
        (("출근길",), "출근길의 가벼운 한입"),
        (("퇴근길",), "퇴근길에 더하는 작은 여유"),
        (("기념일",), "작은 기념일을 화사하게"),
        (("쉬어", "휴식"), "잠시 쉬어 가는 시간"),
        (("티타임",), "오후 티타임의 작은 즐거움"),
        (("늦은 저녁", "저녁"), "늦은 저녁의 여유"),
        (("비 오는", "비오는"), "비 오는 날의 한 잔"),
        (("무더운", "여름"), "무더운 오후의 기분 전환"),
        (("아침",), "아침을 여는 작은 즐거움"),
        (("기분 전환",), "기분 전환이 필요한 순간"),
        (("간식",), "가볍게 즐기는 간식 시간"),
    )
    for terms, value in patterns:
        if any(term in context for term in terms):
            add(value)
    for value in natural_workflow._fallback_hooks(ad_input):
        add(value)
    return values


def _fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    prepared = natural_workflow._copy_input_without_visual_caption(ad_input)
    hashtags = build_deterministic_hashtags(prepared)
    facts = _facts(prepared)
    profile = analyze_copy_input(prepared, image_provided=False)
    menu_name = str(prepared.get("product", {}).get("name") or "")
    values: list[AdCopy] = []

    def add(headline: str, body: str = "") -> None:
        if profile.generation_mode == "headline_only":
            body = ""
        copy = _candidate({"headline": headline, "body": body}, hashtags)
        if copy not in values and not _candidate_issues(copy, prepared):
            values.append(copy)

    request = str(prepared.get("campaign", {}).get("copy_request") or "")
    hooks = _request_hooks(prepared)
    if facts and profile.generation_mode == "headline_and_subcopy":
        combined_fact = " · ".join(facts)
        for hook in hooks:
            add(hook, combined_fact)

    if len(facts) >= 2:
        for left_index, left in enumerate(facts):
            for right_index, right in enumerate(facts):
                if left_index != right_index:
                    add(left, right)
    elif facts:
        fact = facts[0]
        relation = re.fullmatch(r"\s*(.+?)(?:와|과)\s*(.+?)의\s*조합\s*", fact)
        if relation and natural_workflow.headline_repeats_menu_name(fact, menu_name):
            if "대비" in request:
                add("두 요소가 만드는 대비", fact)
                add("서로 다른 두 매력의 대비", fact)
            else:
                add("두 재료의 분명한 조합", fact)
                add("서로 다른 두 재료의 만남", fact)
        if (
            len(fact) <= profile.limits.headline
            and not natural_workflow.headline_repeats_menu_name(fact, menu_name)
        ):
            add(fact)
            if fact.endswith("향"):
                add(
                    f"{fact}{fact_workflow.ad_value_base._subject_particle(fact)} 남기는 여운"
                )
        for value in fact_workflow._compound_fact_title_variants(fact):
            if not natural_workflow.headline_repeats_menu_name(value, menu_name):
                add(value)
        for value in fact_workflow._headline_only_fallback_candidates(prepared):
            if not natural_workflow.headline_repeats_menu_name(value, menu_name):
                add(value)

    if len(values) < FINAL_OPTION_COUNT and facts:
        for hook in hooks:
            add(hook, facts[0])
    if len(values) < FINAL_OPTION_COUNT:
        raise CopyGenerationError(
            "요청 맥락과 상품 사실을 보존하는 fallback 두 개를 만들지 못했습니다."
        )
    return values[:FINAL_OPTION_COUNT]


def _judge_candidates(
    candidates: list[AdCopy],
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None,
) -> tuple[set[int], list[dict[str, Any]], dict[str, int]]:
    prompt = json.dumps(
        {
            "product_name_rendered_separately": ad_input.get("product", {}).get(
                "name"
            ),
            "verified_product_feature": ad_input.get("product", {}).get(
                "features"
            ),
            "requested_tone": ad_input.get("campaign", {}).get("tone"),
            "additional_request": ad_input.get("campaign", {}).get(
                "copy_request"
            ),
            "candidates": [
                {
                    "index": index,
                    "headline": copy["headline"],
                    "subcopy": copy["body"],
                }
                for index, copy in enumerate(candidates, start=1)
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    raw, usage = natural_workflow._call_model(
        prompt=prompt,
        model_id=model_id,
        api_key=api_key,
        client=client,
        schema=natural_workflow._JUDGE_SCHEMA,
        instructions=_JUDGE_INSTRUCTIONS,
    )
    payload = json.loads(raw)
    evaluations = list(payload.get("evaluations") or [])
    passed = {
        int(item["index"])
        for item in evaluations
        if item.get("meaningful") is True
        and int(item.get("naturalness", 0)) >= 4
        and int(item.get("ad_value", 0)) >= 3
        and item.get("grounded") is True
        and item.get("menu_name_repeated") is False
        and item.get("descriptive_caption") is False
    }
    return passed, evaluations, usage


def _append_distinct(values: list[AdCopy], copy: AdCopy) -> None:
    key = (_normalize(copy["headline"]), _normalize(copy["body"]))
    if all(
        (_normalize(item["headline"]), _normalize(item["body"])) != key
        for item in values
    ):
        values.append(copy)


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    prepared = natural_workflow._copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider not in {"openai", "gemini"}:
        raise CopyGenerationError(
            "문구 생성은 OpenAI 또는 Gemini 모델만 지원합니다."
        )
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    hashtags = build_deterministic_hashtags(prepared)
    history: list[dict[str, Any]] = []
    accepted: list[AdCopy] = []
    first_issues: tuple[str, ...] = ()
    retry_context: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        raw = ""
        usage: dict[str, int] = {}
        try:
            prompt = build_banner_copy_prompt(
                prepared,
                retry_context=retry_context,
            )
            raw, usage = natural_workflow._call_model(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                schema=_GENERATION_SCHEMA,
                instructions=_SYSTEM_INSTRUCTIONS,
            )
            payload = json.loads(raw)
            generated = [
                _candidate(item, hashtags) for item in payload.get("options", [])
            ]
            diagnostics: list[dict[str, Any]] = []
            locally_valid: list[tuple[int, AdCopy]] = []
            for index, copy in enumerate(generated, start=1):
                issues = _candidate_issues(copy, prepared)
                diagnostics.append(
                    {
                        "index": index,
                        "headline": copy["headline"],
                        "body": copy["body"],
                        "issues": [issue.code for issue in issues],
                    }
                )
                if not issues:
                    locally_valid.append((index, copy))

            judge_passed, judge_values, judge_usage = _judge_candidates(
                generated,
                prepared,
                model_id=model_id,
                api_key=api_key,
                client=client,
            )
            valid = [
                copy for index, copy in locally_valid if index in judge_passed
            ]
            for copy in valid:
                _append_distinct(accepted, copy)
            history.append(
                {
                    "attempt": attempt,
                    "raw_image_sent": False,
                    "visual_caption_sent_to_copy_model": False,
                    "filtered_visual_facts_sent": list(
                        prepared.get("campaign", {}).get(
                            "copy_eligible_visual_facts"
                        )
                        or []
                    ),
                    "message_cardinality": len(_facts(prepared)),
                    "raw_output": raw,
                    "usage": usage,
                    "local_qc": diagnostics,
                    "naturalness_judge": judge_values,
                    "judge_usage": judge_usage,
                    "qualified_count": len(valid),
                    "accepted_total": len(accepted),
                }
            )
            if len(accepted) >= FINAL_OPTION_COUNT:
                selected = accepted[:FINAL_OPTION_COUNT]
                if not collect_banner_copy_issues(selected, prepared):
                    return BannerCopyGenerationResult(
                        copies=selected,
                        generation_mode=(
                            "headline_and_subcopy"
                            if any(copy["body"] for copy in selected)
                            else "headline_only"
                        ),
                        status="generated" if attempt == 1 else "retried",
                        attempts=attempt,
                        notice=(
                            "상품 사실과 요청 맥락을 함께 반영하고 의미 기반 사실성·"
                            "자연스러움 검증을 통과한 후보만 선택했습니다."
                        ),
                        validation_issues=first_issues,
                        attempt_history=tuple(history),
                        prompt_version=_provider_copy_prompt_version(
                            config.provider
                        ),
                    )
            issue_codes = [
                code for item in diagnostics for code in item["issues"]
            ]
            rejected_reasons = [
                str(item.get("reason") or "")
                for item in judge_values
                if int(item.get("index", 0)) not in judge_passed
            ]
            if not first_issues:
                first_issues = tuple(issue_codes or ("JUDGE_REJECTED",))
            retry_context = {
                "accepted_candidates": [
                    {"headline": item["headline"], "body": item["body"]}
                    for item in accepted
                ],
                "local_issue_codes": issue_codes,
                "judge_rejection_reasons": rejected_reasons,
                "instruction": (
                    "통과한 후보는 반복하지 말고 부족한 후보만 다른 시작과 문장 "
                    "구조로 다시 작성한다."
                ),
            }
        except Exception as exc:
            history.append(
                {
                    "attempt": attempt,
                    "raw_image_sent": False,
                    "visual_caption_sent_to_copy_model": False,
                    "issues": [f"{type(exc).__name__}: {exc}"],
                    "raw_output": raw,
                    "usage": usage,
                    "qualified_count": 0,
                    "accepted_total": len(accepted),
                }
            )
            if not first_issues:
                first_issues = (f"{type(exc).__name__}: {exc}",)

    fallback = _fallback_options(prepared)
    if accepted:
        mixed = accepted[:1]
        for copy in fallback:
            _append_distinct(mixed, copy)
            if len(mixed) == FINAL_OPTION_COUNT:
                break
        if len(mixed) == FINAL_OPTION_COUNT and not collect_banner_copy_issues(
            mixed, prepared
        ):
            return BannerCopyGenerationResult(
                copies=mixed,
                generation_mode=(
                    "headline_and_subcopy"
                    if any(copy["body"] for copy in mixed)
                    else "headline_only"
                ),
                status="mixed",
                attempts=max_attempts,
                notice=(
                    "검증을 통과한 모델 문구는 유지하고 부족한 후보만 요청 맥락을 "
                    "반영한 안전 문구로 보완했습니다."
                ),
                validation_issues=first_issues,
                attempt_history=tuple(history),
                prompt_version=_provider_copy_prompt_version(config.provider),
            )

    if collect_banner_copy_issues(fallback, prepared):
        raise CopyGenerationError("fallback 문구가 최종 QC를 통과하지 못했습니다.")
    return BannerCopyGenerationResult(
        copies=fallback,
        generation_mode=(
            "headline_and_subcopy"
            if any(copy["body"] for copy in fallback)
            else "headline_only"
        ),
        status="fallback",
        attempts=max_attempts,
        notice=(
            "모델 후보가 최종 검증을 통과하지 못해 상품 사실과 허용된 요청 "
            "맥락을 결합한 fallback을 사용했습니다."
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=_provider_copy_prompt_version(config.provider),
    )


advertising_safe_visual_facts = natural_workflow.advertising_safe_visual_facts
attach_visual_context = natural_workflow.attach_visual_context
extract_verified_visual_facts = natural_workflow.extract_verified_visual_facts
