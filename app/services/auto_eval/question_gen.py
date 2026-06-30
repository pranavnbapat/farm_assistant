from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.clients.vllm_client import generate_once
from app.config import get_settings

from .models import LocalizedQuestion, PlannedBaseQuestion

S = get_settings()
logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Question generator returned empty output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1).strip())
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _question_prompt(planned: PlannedBaseQuestion, languages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prompt for ONE base question localized into every requested language."""
    system = (
        "You generate neutral benchmark questions for an agricultural chatbot arena. "
        "Return strict JSON only. Do not include markdown. "
        "Agriculture questions must be realistic EU agriculture, forestry, advisory, policy, crop, livestock, soil, climate, machinery, or farm-management questions. "
        "Non-agriculture questions must be clearly outside agriculture so a scoped farm assistant should identify the mismatch or answer cautiously. "
        "Write one English source question, then a faithful localized version for every requested EU language. "
        "Do not add answers."
    )
    user = (
        "Create one localized benchmark question.\n"
        f"base_question_id: {planned.base_question_id}\n"
        f"topic_category: {planned.topic_category}\n"
        f"Languages JSON: {json.dumps(languages, ensure_ascii=False)}\n\n"
        "Return exactly this JSON shape: {\n"
        f"  \"base_question_id\": \"{planned.base_question_id}\",\n"
        f"  \"topic_category\": \"{planned.topic_category}\",\n"
        "  \"source_language\": \"en\",\n"
        "  \"source_question\": \"...\",\n"
        "  \"localizations\": [{\"language\": \"en\", \"language_name\": \"English\", \"question\": \"...\"}]\n"
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_one(payload: Any, planned: PlannedBaseQuestion, languages: list[dict[str, str]]) -> list[LocalizedQuestion]:
    """Parse one base-question payload, tolerating partially-missing localizations."""
    expected_codes = {language["code"] for language in languages}
    if not isinstance(payload, dict):
        raise ValueError("base question output must be a JSON object")

    topic = str(payload.get("topic_category") or planned.topic_category).strip()
    if topic not in {"agriculture", "non_agriculture"}:
        topic = planned.topic_category
    source_question = str(payload.get("source_question") or "").strip()
    source_language = str(payload.get("source_language") or "en").strip() or "en"
    localizations = payload.get("localizations")
    if not isinstance(localizations, list):
        raise ValueError("base question output is missing a localizations list")

    localized: list[LocalizedQuestion] = []
    seen: set[str] = set()
    for loc in localizations:
        if not isinstance(loc, dict):
            continue
        code = str(loc.get("language") or "").strip().lower()
        if code not in expected_codes or code in seen:
            continue
        question = str(loc.get("question") or "").strip()
        if not question:
            continue
        seen.add(code)
        localized.append(LocalizedQuestion(
            base_question_id=planned.base_question_id,
            topic_category=topic,
            source_language=source_language,
            source_question=source_question or question,
            language=code,
            language_name=str(loc.get("language_name") or code).strip(),
            question=question,
        ))

    missing = expected_codes - seen
    if missing:
        # Tolerate partial coverage: a few missing/garbled localizations (Maltese, Irish…)
        # should not discard the whole base question or abort the cycle.
        logger.warning(
            "Question %s missing %d/%d localizations: %s",
            planned.base_question_id, len(missing), len(expected_codes), ", ".join(sorted(missing)),
        )
    return localized


async def _generate_one(planned: PlannedBaseQuestion, languages: list[dict[str, str]]) -> list[LocalizedQuestion]:
    raw = await generate_once(
        "",
        temperature=0.7,
        max_tokens=getattr(S, "AUTOMATION_QUESTION_MAX_TOKENS", 3000),
        messages=_question_prompt(planned, languages),
    )
    return _parse_one(_extract_json(raw), planned, languages)


async def generate_localized_questions(
    plan: list[PlannedBaseQuestion],
    languages: list[dict[str, str]],
) -> list[LocalizedQuestion]:
    # One Qwen3 call per base question (bounded token budget, parallelized). A failed or
    # partial base question is skipped rather than aborting the whole cycle.
    results = await asyncio.gather(
        *(_generate_one(planned, languages) for planned in plan),
        return_exceptions=True,
    )
    localized: list[LocalizedQuestion] = []
    for planned, result in zip(plan, results):
        if isinstance(result, BaseException):
            logger.warning("Question generation failed for %s: %s", planned.base_question_id, result)
            continue
        localized.extend(result)
    if not localized:
        raise RuntimeError("Question generation produced no usable localized questions")
    return localized
