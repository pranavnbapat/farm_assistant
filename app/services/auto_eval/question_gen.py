from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from app.clients.vllm_client import generate_once
from app.config import get_settings

from .models import LocalizedQuestion, PlannedBaseQuestion

S = get_settings()
logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> Any:
    # Tolerant extraction: strict=False allows control chars; brace-scan survives any
    # preamble/thinking the model prints around a small one-field JSON object.
    text = (raw or "").strip()
    if not text:
        raise ValueError("Generator returned empty output")
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1).strip(), strict=False)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1], strict=False)
        raise


def _question_field(raw: str) -> str:
    """Pull the `question` string out of a tiny {"question": "..."} object."""
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("empty question field")
    return question


async def _call_json(messages: list[dict[str, str]], *, max_tokens: int) -> str:
    """One generation call with 429 backoff + re-generate on malformed/empty JSON.

    Each call asks for a single short string, so the JSON is tiny and rarely malformed —
    a stark contrast to asking one call for all 24 languages at once.
    """
    retries = getattr(S, "AUTOMATION_QUESTION_RETRIES", 4)
    delay = getattr(S, "AUTOMATION_QUESTION_RETRY_BASE_DELAY", 3.0)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await generate_once("", temperature=0.7, max_tokens=max_tokens, messages=messages)
        except httpx.HTTPStatusError as error:
            last_error = error
            code = error.response.status_code if error.response is not None else None
            if code == 429 and attempt < retries:
                logger.warning("question-gen 429; retry %d/%d after %.1fs", attempt + 1, retries, delay)
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("generation produced no output")


def _source_prompt(planned: PlannedBaseQuestion) -> list[dict[str, str]]:
    if planned.topic_category == "agriculture":
        scope = (
            "a realistic EU agriculture, forestry, advisory, policy, crop, livestock, soil, "
            "climate, machinery, or farm-management question"
        )
    else:
        scope = (
            "a question clearly OUTSIDE agriculture (so a scoped farm assistant should "
            "recognise the mismatch or answer cautiously)"
        )
    system = (
        "You generate neutral benchmark questions for an agricultural chatbot arena. "
        "Return strict JSON only — no markdown, no extra text."
    )
    user = (
        f"Write ONE {scope}, in English. One sentence, no answer.\n"
        "Return exactly this JSON: {\"question\": \"...\"}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _translate_prompt(question_en: str, language_name: str) -> list[dict[str, str]]:
    system = (
        f"You are a professional translator. Translate the user's question into {language_name}, "
        "preserving meaning and any technical/agricultural terms. "
        "Return strict JSON only — no markdown, no notes."
    )
    user = f"Question (English): {question_en}\nReturn exactly this JSON: {{\"question\": \"<translation in {language_name}>\"}}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _generate_source(planned: PlannedBaseQuestion) -> str:
    last_error: Exception | None = None
    for attempt in range((getattr(S, "AUTOMATION_QUESTION_RETRIES", 4)) + 1):
        try:
            raw = await _call_json(_source_prompt(planned), max_tokens=400)
            return _question_field(raw)
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            await asyncio.sleep(0.5)
    raise last_error or RuntimeError("source generation failed")


async def _translate(question_en: str, language_name: str) -> str:
    last_error: Exception | None = None
    for attempt in range((getattr(S, "AUTOMATION_QUESTION_RETRIES", 4)) + 1):
        try:
            raw = await _call_json(_translate_prompt(question_en, language_name), max_tokens=600)
            return _question_field(raw)
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            await asyncio.sleep(0.5)
    raise last_error or RuntimeError("translation failed")


async def generate_localized_questions(
    plan: list[PlannedBaseQuestion],
    languages: list[dict[str, str]],
) -> list[LocalizedQuestion]:
    """For each base question: generate the English question once, then translate it into
    each language with its own tiny call — strictly serial (no concurrent bursts at the
    rate-limited vLLM). Per-language failures are skipped, not fatal.
    """
    localized: list[LocalizedQuestion] = []
    for planned in plan:
        try:
            source_en = await _generate_source(planned)
        except Exception as error:
            logger.warning("Source question generation failed for %s: %s", planned.base_question_id, error)
            continue
        for language in languages:
            code = language["code"]
            name = language["name"]
            try:
                question = source_en if code == "en" else await _translate(source_en, name)
                if not question.strip():
                    raise ValueError("empty translation")
            except Exception as error:
                logger.warning("Translate %s -> %s failed: %s", planned.base_question_id, code, str(error)[:120])
                continue
            localized.append(LocalizedQuestion(
                base_question_id=planned.base_question_id,
                topic_category=planned.topic_category,
                source_language="en",
                source_question=source_en,
                language=code,
                language_name=name,
                question=question.strip(),
            ))
    if not localized:
        raise RuntimeError("Question generation produced no usable localized questions")
    return localized
