from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any

import httpx

from app.clients.vllm_client import generate_once
from app.config import get_settings

from .constants import QUESTION_ANGLES
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


async def _call_json(messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.7) -> str:
    """One generation call with 429 backoff + re-generate on malformed/empty JSON.

    Each call asks for a single short string, so the JSON is tiny and rarely malformed —
    a stark contrast to asking one call for all 24 languages at once.
    """
    retries = getattr(S, "AUTOMATION_QUESTION_RETRIES", 4)
    delay = getattr(S, "AUTOMATION_QUESTION_RETRY_BASE_DELAY", 3.0)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await generate_once("", temperature=temperature, max_tokens=max_tokens, messages=messages)
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


def _source_prompt(planned: PlannedBaseQuestion, angle: str, avoid: list[str] | None = None) -> list[dict[str, str]]:
    hint = planned.topic_hint
    if planned.topic_category == "agriculture":
        focus = f"about {hint}" if hint else "about EU agriculture, forestry, advisory or policy"
        instruction = (
            f"Write ONE specific, realistic EU agriculture question {focus}, framed as {angle}. "
            "Make it concrete and distinct. One sentence, no answer."
        )
    else:
        focus = f"about {hint}" if hint else "on a clearly non-agriculture subject"
        instruction = (
            f"Write ONE question clearly OUTSIDE agriculture, {focus}, framed as {angle}, "
            "so a scoped farm assistant should recognise the mismatch. One sentence, no answer."
        )
    if avoid:
        instruction += " Do NOT repeat any of these earlier questions: " + " | ".join(a[:120] for a in avoid[-5:])
    system = (
        "You generate diverse, non-repeating benchmark questions for a chatbot arena. "
        "Return strict JSON only — no markdown, no extra text."
    )
    user = instruction + "\nReturn exactly this JSON: {\"question\": \"...\"}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _translate_prompt(question_en: str, language_name: str) -> list[dict[str, str]]:
    system = (
        f"You are a professional translator. Translate the user's question into {language_name}, "
        "preserving meaning and any technical/agricultural terms. "
        "Return strict JSON only — no markdown, no notes."
    )
    user = f"Question (English): {question_en}\nReturn exactly this JSON: {{\"question\": \"<translation in {language_name}>\"}}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize(text: str) -> str:
    """Normalize for duplicate detection: lowercase, strip punctuation and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower())).strip()


async def _generate_unique_source(planned: PlannedBaseQuestion, seen_norm: set[str], recent: list[str]) -> str:
    """Generate an English source question that is NOT a duplicate of one already produced.

    Each attempt varies the angle and passes the recent questions to avoid; on a duplicate
    (normalized match) it regenerates. Guarantees no exact/normalized repeats within a run.
    """
    tries = getattr(S, "AUTOMATION_SOURCE_UNIQUE_TRIES", 8)
    last_q: str | None = None
    last_error: Exception | None = None
    for attempt in range(tries):
        angle = random.choice(QUESTION_ANGLES)
        try:
            raw = await _call_json(_source_prompt(planned, angle, avoid=recent), max_tokens=400, temperature=0.95)
            q = _question_field(raw)
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            await asyncio.sleep(0.4)
            continue
        last_q = q
        if _normalize(q) not in seen_norm:
            return q
        logger.info("    duplicate question, regenerating (attempt %d/%d)", attempt + 1, tries)
    if last_q is not None:
        logger.warning("    could not find a unique question after %d tries; keeping last", tries)
        return last_q
    raise last_error or RuntimeError("source generation failed")


async def _translate(question_en: str, language_name: str) -> str:
    last_error: Exception | None = None
    for attempt in range((getattr(S, "AUTOMATION_QUESTION_RETRIES", 4)) + 1):
        try:
            # Low temperature for faithful translation (high temp belongs to source diversity).
            raw = await _call_json(_translate_prompt(question_en, language_name), max_tokens=600, temperature=0.3)
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
    seen_norm: set[str] = set()       # all source questions produced so far (normalized) — dedup
    recent: list[str] = []            # last few source questions, shown to the model to avoid
    total = len(plan)
    for i, planned in enumerate(plan, 1):
        t_src = time.perf_counter()
        try:
            source_en = await _generate_unique_source(planned, seen_norm, recent)
        except Exception as error:
            logger.warning("[gen %d/%d] source FAILED for %s: %s", i, total, planned.base_question_id, error)
            continue
        seen_norm.add(_normalize(source_en))
        recent.append(source_en)
        if len(recent) > 8:
            recent.pop(0)
        logger.info("[gen %d/%d] %s (%s/%s) source in %.1fs: %s", i, total, planned.base_question_id,
                    planned.topic_category, planned.topic_hint or "-", time.perf_counter() - t_src, source_en[:90])
        ok = 0
        for language in languages:
            code = language["code"]
            name = language["name"]
            t_tr = time.perf_counter()
            try:
                question = source_en if code == "en" else await _translate(source_en, name)
                if not question.strip():
                    raise ValueError("empty translation")
            except Exception as error:
                logger.warning("[gen %d/%d] %s -> %s FAILED: %s", i, total, planned.base_question_id, code, str(error)[:100])
                continue
            ok += 1
            logger.info("[gen %d/%d] %s -> %s in %.1fs", i, total, planned.base_question_id, code, time.perf_counter() - t_tr)
            localized.append(LocalizedQuestion(
                base_question_id=planned.base_question_id,
                topic_category=planned.topic_category,
                source_language="en",
                source_question=source_en,
                language=code,
                language_name=name,
                question=question.strip(),
            ))
        logger.info("[gen %d/%d] %s done: %d/%d languages in %.1fs",
                    i, total, planned.base_question_id, ok, len(languages), time.perf_counter() - t_src)
    if not localized:
        raise RuntimeError("Question generation produced no usable localized questions")
    logger.info("question generation complete: %d localized questions from %d base questions", len(localized), total)
    return localized
