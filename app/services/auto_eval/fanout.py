from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import get_settings

from .constants import DEFAULT_VARIANTS
from .models import LocalizedQuestion, VariantAnswer

S = get_settings()


def _base_url(backend: str) -> str:
    if backend == "um_qwen3":
        return getattr(S, "FARM_ASSISTANT_UM_QWEN3_URL", "https://farm-assistant.nexavion.com").rstrip("/")
    if backend == "eurollm":
        return getattr(S, "FARM_ASSISTANT_EUROLLM_URL", "http://127.0.0.1:18005").rstrip("/")
    if backend == "euf_chatbot_tnods":
        return getattr(S, "EUF_CHATBOT_API_URL", "https://farm-assistant.tnods.nl").rstrip("/")
    raise ValueError(f"Unsupported answer backend: {backend}")


def _auth_headers(access_token: str, refresh_token: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {access_token}"}
    if refresh_token:
        headers["X-Refresh-Token"] = refresh_token
    return headers


def _experiment_headers(batch_id: str, question: LocalizedQuestion, variant: dict[str, str]) -> dict[str, str]:
    return {
        "X-Farm-Experiment-Id": getattr(S, "AUTOMATION_EXPERIMENT_ID", "automated_eval"),
        "X-Farm-Experiment-Session-Id": batch_id,
        "X-Farm-Experiment-Variant-Id": variant["id"],
        "X-Farm-Experiment-Backend": variant["backend"],
        "X-Farm-Experiment-Rag-Profile": "",
        "X-Farm-Auto-Eval-Language": question.language,
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len((text or "").split()))


def _variant_metadata(variant: dict[str, str]) -> dict[str, Any]:
    backend = variant["backend"]
    model_name = {
        "um_qwen3": getattr(S, "FARM_ASSISTANT_UM_QWEN3_MODEL_NAME", "qwen3-30b-a3b-awq"),
        "euf_chatbot_tnods": getattr(S, "EUF_CHATBOT_MODEL_NAME", "azure_ai/mistral-small-2503"),
        "eurollm": getattr(S, "FARM_ASSISTANT_EUROLLM_MODEL_NAME", "utter-project/EuroLLM-9B-Instruct"),
    }.get(backend, backend)
    return {
        "variant_id": variant["id"],
        "backend_key": backend,
        "display_name_internal": variant.get("label", variant["id"]),
        "model_name": model_name,
        "automation_source": "farm_assistant.auto_eval",
    }


def _extract_sse_events(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line.strip() and data_lines:
            events.append((event_name, "\n".join(data_lines)))
            event_name = "message"
            data_lines = []
    if data_lines:
        events.append((event_name, "\n".join(data_lines)))
    return events


def _parse_stream_response(text: str) -> tuple[str, list[dict[str, Any]], str | None, dict[str, Any]]:
    tokens: list[str] = []
    sources: list[dict[str, Any]] = []
    grounding_mode = None
    meta: dict[str, Any] = {}
    for event, raw_data in _extract_sse_events(text):
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = raw_data
        if event in {"token", "answer_token"}:
            if isinstance(payload, dict):
                tokens.append(str(payload.get("token") or payload.get("response") or ""))
            else:
                tokens.append(str(payload))
        elif event == "grounding" and isinstance(payload, dict):
            grounding_mode = payload.get("mode") or grounding_mode
        elif event == "sources" and isinstance(payload, list):
            sources = payload
        elif event == "stats" and isinstance(payload, dict):
            meta.update(payload)
        elif event in {"final", "done"} and isinstance(payload, dict):
            message = (
                payload.get("assistant_message")
                or payload.get("answer")
                or payload.get("response")
                or payload.get("text")
            )
            if message and not tokens:
                tokens.append(str(message))
            if isinstance(payload.get("sources"), list):
                sources = payload["sources"]
            grounding_mode = payload.get("grounding_mode") or grounding_mode
            if isinstance(payload.get("meta"), dict):
                meta.update(payload["meta"])
        elif event == "app_error":
            raise RuntimeError(str(payload.get("message") if isinstance(payload, dict) else payload))
    return "".join(tokens).strip(), sources, grounding_mode, meta


async def _create_session(client: httpx.AsyncClient, base_url: str, headers: dict[str, str]) -> str:
    response = await client.post(
        f"{base_url}/chatbot/api/chats",
        headers={**headers, "Content-Type": "application/json"},
        json={"title": "", "metadata": {"evaluation_mode": "automated_arena", "ephemeral": True}},
    )
    response.raise_for_status()
    payload = response.json()
    session_id = payload.get("session_uuid") or payload.get("id")
    if not session_id:
        raise RuntimeError("Farm Assistant did not return a session id")
    return str(session_id)


async def _delete_session(client: httpx.AsyncClient, base_url: str, session_id: str, headers: dict[str, str]) -> None:
    try:
        await client.delete(f"{base_url}/chatbot/api/chats/{session_id}", headers=headers)
    except Exception:
        pass


async def _farm_assistant_answer(
    client: httpx.AsyncClient,
    question: LocalizedQuestion,
    variant: dict[str, str],
    access_token: str,
    refresh_token: str | None,
    batch_id: str,
) -> VariantAnswer:
    backend = variant["backend"]
    base_url = _base_url(backend)
    headers = {**_auth_headers(access_token, refresh_token), **_experiment_headers(batch_id, question, variant)}
    started = time.monotonic()
    session_id = None
    try:
        if backend == "um_qwen3":
            path = "/chatbot/api/chats/message/stream"
        else:
            session_id = await _create_session(client, base_url, headers)
            path = f"/chatbot/api/chats/{session_id}/message/stream"
        query = urlencode({"q": question.question, "pause_personalization": "true"})
        response = await client.get(f"{base_url}{path}?{query}", headers=headers)
        response.raise_for_status()
        message, sources, grounding_mode, meta = _parse_stream_response(response.text)
        if not message:
            raise RuntimeError("Backend returned an empty response")
        latency_ms = int((time.monotonic() - started) * 1000)
        return VariantAnswer(
            variant_id=variant["id"],
            backend=backend,
            assistant_message=message,
            latency_ms=latency_ms,
            sources=sources,
            grounding_mode=grounding_mode,
            variant_metadata=_variant_metadata(variant),
            runtime_metadata={
                **meta,
                "request_endpoint": path,
                "request_method": "GET",
                "session_mode": "non_session" if backend == "um_qwen3" else "session",
                "question_length_chars": len(question.question),
                "question_length_tokens_est": _estimate_tokens(question.question),
                "answer_length_chars": len(message),
                "answer_length_tokens_est": _estimate_tokens(message),
                "sources_count": len(sources),
                "stream_completed": True,
            },
        )
    except Exception as error:
        return VariantAnswer(
            variant_id=variant["id"],
            backend=backend,
            assistant_message="This response could not be generated.",
            latency_ms=int((time.monotonic() - started) * 1000),
            error=str(error)[:1000],
            variant_metadata=_variant_metadata(variant),
            runtime_metadata={"stream_completed": False, "error": str(error)[:500]},
        )
    finally:
        if session_id:
            await _delete_session(client, base_url, session_id, headers)


def _extract_user_facing_content(payload: Any) -> str:
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("user_facing_content") or message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


async def _mistral_answer(client: httpx.AsyncClient, question: LocalizedQuestion, variant: dict[str, str]) -> VariantAnswer:
    base_url = _base_url("euf_chatbot_tnods")
    api_key = getattr(S, "EUF_CHATBOT_API_KEY", "")
    uuid = getattr(S, "EUF_CHATBOT_ARENA_UUID", "45b75f62-3fa3-4b18-8593-1411f110a98e")
    started = time.monotonic()
    if not api_key:
        return VariantAnswer(
            variant_id=variant["id"], backend=variant["backend"], assistant_message="This response could not be generated.",
            error="EUF_CHATBOT_API_KEY is not configured", variant_metadata=_variant_metadata(variant),
        )
    headers = {"Accept": "application/json", "X-API-Key": api_key, "uuid": uuid}
    try:
        create = await client.post(f"{base_url}/chatbot/api/chats", params={"description": "automated_arena"}, headers=headers)
        create.raise_for_status()
        chat_id = create.json().get("id")
        if not chat_id:
            raise RuntimeError("EUF Chatbot did not return a chat id")
        response = await client.post(
            f"{base_url}/chatbot/api/chats/{chat_id}/message",
            headers={**headers, "Content-Type": "application/json"},
            json={"message": question.question},
        )
        response.raise_for_status()
        message = _extract_user_facing_content(response.json())
        if not message:
            raise RuntimeError("EUF Chatbot returned an empty response")
        return VariantAnswer(
            variant_id=variant["id"], backend=variant["backend"], assistant_message=message,
            latency_ms=int((time.monotonic() - started) * 1000), variant_metadata=_variant_metadata(variant),
            runtime_metadata={
                "request_endpoint": "/chatbot/api/chats/{chat_id}/message",
                "request_method": "POST",
                "session_mode": "external_json",
                "question_length_chars": len(question.question),
                "question_length_tokens_est": _estimate_tokens(question.question),
                "answer_length_chars": len(message),
                "answer_length_tokens_est": _estimate_tokens(message),
                "sources_count": 0,
                "stream_completed": True,
            },
        )
    except Exception as error:
        return VariantAnswer(
            variant_id=variant["id"], backend=variant["backend"], assistant_message="This response could not be generated.",
            latency_ms=int((time.monotonic() - started) * 1000), error=str(error)[:1000],
            variant_metadata=_variant_metadata(variant), runtime_metadata={"stream_completed": False, "error": str(error)[:500]},
        )


async def answer_question(
    question: LocalizedQuestion,
    access_token: str,
    refresh_token: str | None,
    batch_id: str,
    variants: list[dict[str, str]] | None = None,
) -> list[VariantAnswer]:
    timeout = httpx.Timeout(getattr(S, "AUTOMATION_REQUEST_TIMEOUT", 180.0))
    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL, trust_env=False) as client:
        tasks = []
        for variant in variants or DEFAULT_VARIANTS:
            if variant["backend"] == "euf_chatbot_tnods":
                tasks.append(_mistral_answer(client, question, variant))
            else:
                tasks.append(_farm_assistant_answer(client, question, variant, access_token, refresh_token, batch_id))
        return await asyncio.gather(*tasks)


def assign_labels(answers: list[VariantAnswer], seed: int | None = None) -> list[VariantAnswer]:
    shuffled = list(answers)
    random.Random(seed).shuffle(shuffled)
    for index, answer in enumerate(shuffled):
        answer.label = chr(ord("A") + index)
    return shuffled
