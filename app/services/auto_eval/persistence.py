from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings

from .models import JudgeResult, LocalizedQuestion, VariantAnswer

S = get_settings()


class ServiceTokens(dict):
    @property
    def access_token(self) -> str:
        return str(self.get("access_token") or "")

    @property
    def refresh_token(self) -> str:
        return str(self.get("refresh_token") or "")


def _backend_url() -> str:
    if not S.CHAT_BACKEND_URL:
        raise RuntimeError("CHAT_BACKEND_URL is not configured")
    return S.CHAT_BACKEND_URL.rstrip("/")


def _auth_headers(tokens: ServiceTokens) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {tokens.access_token}"}
    if tokens.refresh_token:
        headers["X-Refresh-Token"] = tokens.refresh_token
    return headers


async def service_login() -> ServiceTokens:
    email = getattr(S, "AUTOMATION_SERVICE_EMAIL", "")
    password = getattr(S, "AUTOMATION_SERVICE_PASSWORD", "")
    if not email or not password:
        raise RuntimeError("AUTOMATION_SERVICE_EMAIL and AUTOMATION_SERVICE_PASSWORD are required")
    async with httpx.AsyncClient(timeout=30.0, verify=S.VERIFY_SSL, trust_env=False) as client:
        response = await client.post(f"{_backend_url()}/fastapi/login/", json={"email": email, "password": password})
        response.raise_for_status()
        payload = response.json()
    if not payload.get("access_token"):
        raise RuntimeError("Service login did not return an access token")
    return ServiceTokens(payload)


def _answer_payload(answer: VariantAnswer) -> dict[str, Any]:
    return {
        "label": answer.label,
        "variant_id": answer.variant_id,
        "backend": answer.backend,
        "assistant_message": answer.assistant_message,
        "latency_ms": answer.latency_ms,
        "sources": answer.sources,
        "grounding_mode": answer.grounding_mode,
        "error": answer.error,
        "variant_metadata": answer.variant_metadata,
        "runtime_metadata": answer.runtime_metadata,
    }


async def persist_comparison_run(
    tokens: ServiceTokens,
    batch_id: str,
    question: LocalizedQuestion,
    answers: list[VariantAnswer],
) -> str:
    payload = {
        "question": question.question,
        "compare_session_id": batch_id,
        "experiment_id": getattr(S, "AUTOMATION_EXPERIMENT_ID", "automated_eval"),
        "question_metadata": {
            "generated_by": "qwen3",
            "batch_id": batch_id,
            "base_question_id": question.base_question_id,
            "source_language": question.source_language,
            "source_question": question.source_question,
            "language": question.language,
            "language_name": question.language_name,
            "domain": question.topic_category,
            "is_agriculture": question.topic_category == "agriculture",
            "topic_ratio": getattr(S, "AUTOMATION_TOPIC_RATIO", "3:1"),
        },
        "answers": [_answer_payload(answer) for answer in answers],
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(getattr(S, "AUTOMATION_REQUEST_TIMEOUT", 180.0)), verify=S.VERIFY_SSL, trust_env=False) as client:
        response = await client.post(
            f"{_backend_url()}/chat/experiments/automated/runs/",
            headers={**_auth_headers(tokens), "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    run_id = data.get("comparison_run_id") or data.get("session_uuid")
    if not run_id:
        raise RuntimeError("Comparison persistence did not return comparison_run_id")
    return str(run_id)


async def fetch_runs(tokens: ServiceTokens, experiment_id: str, limit: int = 500) -> list[dict[str, Any]]:
    """Read back persisted comparison runs (newest first) for the judge phase.

    Each result includes `answers` (label + assistant_message + backend) and
    `llm_evaluation_providers` (providers already judged), so callers can skip
    runs that are fully evaluated.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(getattr(S, "AUTOMATION_REQUEST_TIMEOUT", 180.0)), verify=S.VERIFY_SSL, trust_env=False) as client:
        response = await client.get(
            f"{_backend_url()}/chat/experiments/automated/",
            headers=_auth_headers(tokens),
            params={"experiment_id": experiment_id, "limit": max(1, min(int(limit), 500))},
        )
        response.raise_for_status()
        data = response.json()
    return data.get("results") or []


async def persist_judge_result(tokens: ServiceTokens, comparison_run_id: str, batch_id: str, result: JudgeResult) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(getattr(S, "AUTOMATION_REQUEST_TIMEOUT", 180.0)), verify=S.VERIFY_SSL, trust_env=False) as client:
        response = await client.post(
            f"{_backend_url()}/chat/experiments/automated/evaluation/",
            headers={**_auth_headers(tokens), "Content-Type": "application/json"},
            json=result.to_persistence_payload(comparison_run_id, batch_id),
        )
        response.raise_for_status()
