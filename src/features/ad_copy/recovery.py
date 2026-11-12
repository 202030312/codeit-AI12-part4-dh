"""인력 메시지 개수에 따라 제목·보조 문구를 일반화하는 문구 파이프라인 v1.10.17."""

from __future__ import annotations

import json
import re
from typing import Any

from src.features.ad_copy.generator import (
    MODEL_CONFIGS,
    AdCopy,
    CopyGenerationError,
    build_deterministic_hashtags,
    split_distinct_feature_facts,
)
from src.features.ad_copy.validation import _fact_is_reflected, _has_feature_grounding
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
import src.features.ad_copy.internal.caption_rules as previous


COPY_PROMPT_VERSION = "gpt-copy-v1.10.17-message-cardinality-batch-ready"
VISUAL_PROMPT_VERSION = previous.VISUAL_PROMPT_VERSION
INTERNAL_CANDIDATE_COUNT = 4
FINAL_OPTION_COUNT = 2


def _provider_copy_prompt_version(provider: str) -> str:
    return f"{provider}_copy_v1.10.17-message-cardinality-batch-ready"


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _facts(ad_input: dict[str, Any]) -> list[str]:
    feature = " ".join(
        str(ad_input.get("product", {}).get("features") or "").split()
    )
    return [" ".join(item.split()) for item in split_distinct_feature_facts(feature) if item.strip()]


def _fact_sets(copy: AdCopy, facts: list[str]) -> tuple[set[int], set[int]]:
    headline_facts = {
        index for index, fact in enumerate(facts)
        if _fact_is_reflected(copy["headline"], fact)
    }
    body_facts = {
        index for index, fact in enumerate(facts)
        if _fact_is_reflected(copy["body"], fact)
    }
    return headline_facts, body_facts


def collect_banner_copy_issues(
    copies: list[AdCopy],
    ad_input: dict[str, Any],
) -> list[CopyValidationIssue]:
    prepared = previous._copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    product = prepared.get("product", {})
    menu_name = str(product.get("name") or "").strip()
    feature = " ".join(str(product.get("features") or "").split())
    facts = _facts(prepared)
    campaign = prepared.get("campaign", {})
    promotion = str(campaign.get("promotion") or "").strip()
    promotion_mode = normalize_promotion_copy_mode(campaign.get("promotion_copy_mode"))
    issues: list[CopyValidationIssue] = []

    if len(copies) != FINAL_OPTION_COUNT:
        issues.append(CopyValidationIssue(
            "OPTION_COUNT", f"최종 문구는 {FINAL_OPTION_COUNT}개여야 합니다."
        ))
    seen: set[tuple[str, str]] = set()
    for index, copy in enumerate(copies, start=1):
        headline = " ".join(str(copy.get("headline") or "").split())
        body = " ".join(str(copy.get("body") or "").split())
        if not headline:
            issues.append(CopyValidationIssue(
                "MISSING_HEADLINE", f"선택지 {index} 제목이 비어 있습니다.", index, "headline"
            ))
        elif len(headline) > profile.limits.headline:
            issues.append(CopyValidationIssue(
                "HEADLINE_LENGTH", f"선택지 {index} 제목이 길이 제한을 넘습니다.", index, "headline"
            ))
        if len(body) > profile.limits.subcopy:
            issues.append(CopyValidationIssue(
                "SUBCOPY_LENGTH", f"선택지 {index} 보조 문구가 길이 제한을 넘습니다.", index, "subcopy"
            ))
        if previous.headline_repeats_menu_name(headline, menu_name):
            issues.append(CopyValidationIssue(
                "MENU_NAME_REPEAT", f"선택지 {index} 제목이 별도 메뉴명을 반복합니다.", index, "headline"
            ))
        for field, value in (("headline", headline), ("subcopy", body)):
            if previous.has_caption_leak(value):
                issues.append(CopyValidationIssue(
                    "IMAGE_CAPTION_LEAK", f"선택지 {index} {field}에 사진 판독형 표현이 있습니다.", index, field
                ))
            if previous.has_awkward_expression(value):
                issues.append(CopyValidationIssue(
                    "AWKWARD_EXPRESSION", f"선택지 {index} {field}이 자연스럽지 않습니다.", index, field
                ))

        combined = f"{headline} {body}".strip()
        if feature and not _has_feature_grounding(combined, feature, menu_name):
            issues.append(CopyValidationIssue(
                "PRODUCT_FACT_NOT_PRESERVED", f"선택지 {index}에 확인된 상품 특징이 보존되지 않았습니다.", index
            ))
        headline_facts, body_facts = _fact_sets(
            {"headline": headline, "body": body, "hashtags": []}, facts
        )
        if len(facts) >= 2:
            missing = set(range(len(facts))) - headline_facts - body_facts
            if missing:
                issues.append(CopyValidationIssue(
                    "PAIR_MISSING_FACTS", f"선택지 {index}에 독립적인 상품 특징이 빠졌습니다.", index
                ))
            if not headline_facts or not body_facts:
                issues.append(CopyValidationIssue(
                    "MESSAGE_ROLE_NOT_SPLIT", f"선택지 {index}의 두 특징이 제목과 보조 문구로 나뉘지 않았습니다.", index
                ))
        if headline_facts & body_facts:
            issues.append(CopyValidationIssue(
                "SAME_CLAIM_REPEATED", f"선택지 {index}의 제목과 보조 문구가 같은 특징을 반복합니다.", index
            ))
        if len(facts) == 1 and body and headline_facts and body_facts:
            issues.append(CopyValidationIssue(
                "UNNECESSARY_SUBCOPY", f"선택지 {index}이 하나의 특징을 보조 문구로 반복합니다.", index, "subcopy"
            ))
        if previous._unsupported_sensory_claim(headline, feature):
            issues.append(CopyValidationIssue(
                "UNSUPPORTED_SENSORY_HEADLINE", f"선택지 {index} 제목에 입력에 없는 감각 표현이 있습니다.", index, "headline"
            ))
        if promotion_mode == BADGE_ONLY and (
            headline_mentions_promotion(headline) or headline_mentions_promotion(body)
        ):
            issues.append(CopyValidationIssue(
                "PROMOTION_IN_COPY", f"선택지 {index}에 배지 전용 프로모션이 반복됩니다.", index
            ))
        if promotion and promotion_mode != BADGE_ONLY and headline_mentions_promotion(headline):
            if not headline_is_exact_promotion(headline, promotion):
                issues.append(CopyValidationIssue(
                    "PROMOTION_REWRITTEN", f"선택지 {index} 프로모션 조건이 원문과 다릅니다.", index, "headline"
                ))
        key = (_normalize(headline), _normalize(body))
        if key in seen:
            issues.append(CopyValidationIssue(
                "DUPLICATE_COPY", f"선택지 {index}가 다른 선택지와 중복됩니다.", index
            ))
        seen.add(key)
    return issues


def build_banner_copy_prompt(ad_input: dict[str, Any]) -> str:
    prepared = previous._copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    product = prepared.get("product", {})
    campaign = prepared.get("campaign", {})
    facts = _facts(prepared)
    payload = {
        "prompt_version": COPY_PROMPT_VERSION,
        "product_name_rendered_separately": str(product.get("name") or ""),
        "verified_product_facts": facts,
        "copy_eligible_visual_facts": list(campaign.get("copy_eligible_visual_facts") or []),
        "requested_tone": str(campaign.get("tone") or ""),
        "copy_request": str(campaign.get("copy_request") or ""),
        "headline_limit": profile.limits.headline,
        "subcopy_limit": profile.limits.subcopy,
        "candidate_count": INTERNAL_CANDIDATE_COUNT,
        "message_cardinality": len(facts),
        "rules": [
            "상품명 전체는 별도 슬롯에 표시되므로 제목에 반복하지 않는다.",
            "독립 특징이 1개이고 제목에 쓸 수 있으면 특징 중심 제목만 작성하고 body는 빈 문자열로 둔다.",
            "독립 특징이 1개이지만 그 특징을 제목에 쓰면 상품명 전체가 반복되는 경우에만 톤·요청 기반 훅을 제목으로, 특징을 body로 나눈다.",
            "독립 특징이 2개 이상이면 가장 강한 특징을 제목에, 다른 특징을 body에 배정하고 같은 주장을 반복하지 않는다.",
            "copy_eligible_visual_facts는 선택적 참고 정보이며 보이는·확인된·구성·상단·표면 같은 판독형 표현으로 복사하지 않는다.",
            "순간·시간·즐거움·매력·조화 같은 범용 추상어보다 확인된 풍미·향·식감·조합을 우선한다.",
            "입력에 없는 맛·향·식감·재료·효과·상황을 만들지 않는다.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "minItems": INTERNAL_CANDIDATE_COUNT,
            "maxItems": INTERNAL_CANDIDATE_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["headline", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["options"],
    "additionalProperties": False,
}

_SYSTEM_INSTRUCTIONS = """
너는 카페 광고의 한국어 카피 편집자다. 상품명은 별도로 크게 표시된다.
확인된 메시지가 하나면 제목을 특징 중심으로 작성하고 불필요한 body를 만들지 마라.
독립 메시지가 두 개 이상일 때만 제목과 body로 나누고 같은 주장을 반복하지 마라.
사진 판독문, 입력에 없는 감각 속성, 상품명 전체 반복, 상투적인 추상어를 피하라.
JSON 스키마와 일치하는 결과만 반환하라.
""".strip()


def _candidate(item: dict[str, Any], hashtags: list[str]) -> AdCopy:
    return {
        "headline": " ".join(str(item.get("headline") or "").strip(" ,.!?").split()),
        "body": " ".join(str(item.get("body") or "").strip(" .").split()),
        "hashtags": hashtags.copy(),
    }


def _candidate_issues(copy: AdCopy, ad_input: dict[str, Any]) -> list[CopyValidationIssue]:
    placeholder: AdCopy = {
        "headline": "임시 후보 구분 문구",
        "body": "",
        "hashtags": [],
    }
    return [
        issue for issue in collect_banner_copy_issues([copy, placeholder], ad_input)
        if issue.option_index == 1
    ]


def _fallback_options(ad_input: dict[str, Any]) -> list[AdCopy]:
    prepared = previous._copy_input_without_visual_caption(ad_input)
    hashtags = build_deterministic_hashtags(prepared)
    facts = _facts(prepared)
    profile = analyze_copy_input(prepared, image_provided=False)
    menu_name = str(prepared.get("product", {}).get("name") or "")
    values: list[AdCopy] = []

    def add(headline: str, body: str = "") -> None:
        copy = _candidate({"headline": headline, "body": body}, hashtags)
        if (
            copy not in values
            and not _candidate_issues(copy, prepared)
        ):
            values.append(copy)

    if len(facts) >= 2:
        for left_index, left in enumerate(facts):
            for right_index, right in enumerate(facts):
                if left_index != right_index:
                    add(left, right)
    elif facts:
        fact = facts[0]
        request = str(prepared.get("campaign", {}).get("copy_request") or "")
        relation = re.fullmatch(r"\s*(.+?)(?:와|과)\s*(.+?)의\s*조합\s*", fact)
        if relation and previous.headline_repeats_menu_name(fact, menu_name):
            if "대비" in request:
                add("두 요소가 만드는 대비", fact)
                add("서로 다른 두 매력의 대비", fact)
            else:
                add("두 재료의 분명한 조합", fact)
                add("서로 다른 두 재료의 만남", fact)
        if (
            len(fact) <= profile.limits.headline
            and not previous.headline_repeats_menu_name(fact, menu_name)
        ):
            add(fact)
            if fact.endswith("향"):
                add(f"{fact}{fact_workflow.ad_value_base._subject_particle(fact)} 남기는 여운")
        reduced = re.sub(
            r"(?:이|가)\s*특징인\s+[0-9A-Za-z가-힣]+$",
            "",
            fact,
        ).strip()
        if reduced and reduced != fact and not previous.headline_repeats_menu_name(reduced, menu_name):
            add(reduced)
            add(f"{reduced}{fact_workflow.ad_value_base._subject_particle(reduced)} 남기는 여운")
        for value in fact_workflow._compound_fact_title_variants(fact):
            if not previous.headline_repeats_menu_name(value, menu_name):
                add(value)
        for value in fact_workflow._headline_only_fallback_candidates(prepared):
            if not previous.headline_repeats_menu_name(value, menu_name):
                add(value)
        if previous.headline_repeats_menu_name(fact, menu_name):
            for hook in previous._fallback_hooks(prepared):
                add(hook, fact)
    if len(values) < FINAL_OPTION_COUNT:
        for hook in previous._fallback_hooks(prepared):
            add(hook, facts[0] if facts else "")
    if len(values) < FINAL_OPTION_COUNT:
        raise CopyGenerationError("메시지 개수 규칙을 통과하는 fallback 두 개를 만들지 못했습니다.")
    return values[:FINAL_OPTION_COUNT]


def generate_safe_banner_copy_options(
    ad_input: dict[str, Any],
    *,
    model_id: str,
    api_key: str,
    client: Any | None = None,
    max_attempts: int = 2,
) -> BannerCopyGenerationResult:
    prepared = previous._copy_input_without_visual_caption(ad_input)
    profile = analyze_copy_input(prepared, image_provided=False)
    if profile.input_issues:
        raise CopyInputError(" ".join(profile.input_issues))
    config = MODEL_CONFIGS.get(model_id)
    if config is None or config.provider not in {"openai", "gemini"}:
        raise CopyGenerationError("문구 생성은 OpenAI 또는 Gemini 모델만 지원합니다.")
    if not api_key.strip() and client is None:
        raise CopyGenerationError(f"{config.api_key_env}가 설정되지 않았습니다.")

    hashtags = build_deterministic_hashtags(prepared)
    history: list[dict[str, Any]] = []
    first_issues: tuple[str, ...] = ()
    for attempt in range(1, max_attempts + 1):
        raw = ""
        usage: dict[str, int] = {}
        try:
            prompt = build_banner_copy_prompt(prepared)
            if attempt > 1:
                prompt += "\n재작성: 이전 후보와 다르게 쓰고 메시지 개수·중복·상품명 분리를 다시 검사한다."
            raw, usage = previous._call_model(
                prompt=prompt,
                model_id=model_id,
                api_key=api_key,
                client=client,
                schema=_GENERATION_SCHEMA,
                instructions=_SYSTEM_INSTRUCTIONS,
            )
            payload = json.loads(raw)
            generated = [_candidate(item, hashtags) for item in payload.get("options", [])]
            diagnostics = []
            locally_valid: list[tuple[int, AdCopy]] = []
            for index, copy in enumerate(generated, start=1):
                issues = _candidate_issues(copy, prepared)
                diagnostics.append({
                    "index": index,
                    "headline": copy["headline"],
                    "body": copy["body"],
                    "issues": [issue.code for issue in issues],
                })
                if not issues:
                    locally_valid.append((index, copy))
            judge_passed, judge_values, judge_usage = previous._judge_candidates(
                generated,
                prepared,
                model_id=model_id,
                api_key=api_key,
                client=client,
            )
            valid = [copy for index, copy in locally_valid if index in judge_passed]
            history.append({
                "attempt": attempt,
                "raw_image_sent": False,
                "visual_caption_sent_to_copy_model": False,
                "filtered_visual_facts_sent": list(
                    prepared.get("campaign", {}).get("copy_eligible_visual_facts") or []
                ),
                "message_cardinality": len(_facts(prepared)),
                "raw_output": raw,
                "usage": usage,
                "local_qc": diagnostics,
                "naturalness_judge": judge_values,
                "judge_usage": judge_usage,
                "qualified_count": len(valid),
            })
            if len(valid) >= FINAL_OPTION_COUNT:
                selected = valid[:FINAL_OPTION_COUNT]
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
                            "확인된 메시지 개수에 따라 제목·보조 문구를 구성하고 "
                            "메뉴명 중복·사진 판독·자연스러움 QC를 통과한 후보만 선택했습니다."
                        ),
                        validation_issues=first_issues,
                        attempt_history=tuple(history),
                        prompt_version=_provider_copy_prompt_version(config.provider),
                    )
            if not first_issues:
                first_issues = tuple(code for item in diagnostics for code in item["issues"])
        except Exception as exc:
            history.append({
                "attempt": attempt,
                "raw_image_sent": False,
                "visual_caption_sent_to_copy_model": False,
                "issues": [f"{type(exc).__name__}: {exc}"],
                "raw_output": raw,
                "usage": usage,
                "qualified_count": 0,
            })
            if not first_issues:
                first_issues = (f"{type(exc).__name__}: {exc}",)

    fallback = _fallback_options(prepared)
    if collect_banner_copy_issues(fallback, prepared):
        raise CopyGenerationError("fallback 문구가 최종 QC를 통과하지 못했습니다.")
    return BannerCopyGenerationResult(
        copies=fallback,
        generation_mode=(
            "headline_and_subcopy" if any(copy["body"] for copy in fallback) else "headline_only"
        ),
        status="fallback",
        attempts=max_attempts,
        notice=(
            "모델 후보가 메시지 개수 규칙을 통과하지 못해 확인된 상품 특징 "
            "기반 fallback을 사용했습니다."
        ),
        validation_issues=first_issues,
        attempt_history=tuple(history),
        prompt_version=_provider_copy_prompt_version(config.provider),
    )


advertising_safe_visual_facts = previous.advertising_safe_visual_facts
attach_visual_context = previous.attach_visual_context
extract_verified_visual_facts = previous.extract_verified_visual_facts

