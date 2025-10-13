# app/clients/ollama_client.py

import httpx, json

from typing import Dict, Any

from app.config import get_settings

S = get_settings()

def build_gen_payload(prompt: str, temperature: float, num_predict: int) -> Dict[str, Any]:
    return {
        "model": S.LLM_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": S.NUM_CTX
        },
        "stream": False
    }

def build_stream_payload(prompt: str, temperature: float, num_predict: int, context: list[int] | None = None) -> Dict[str, Any]:
    p = {
        "model": S.LLM_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": S.NUM_CTX
        },
        "stream": True
    }
    return _attach_context(p, context)

async def generate_once(prompt: str, temperature: float, num_predict: int) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=300, write=1800)) as client:
        r = await client.post(f"{S.OLLAMA_URL}/api/generate",
                              json=build_gen_payload(prompt, temperature, num_predict))
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

async def stream_generate(prompt: str, temperature: float, num_predict: int, context: list[int] | None = None):
    timeout = httpx.Timeout(connect=30.0, read=3600.0, write=300.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{S.OLLAMA_URL}/api/generate",
            json=build_stream_payload(prompt, temperature, num_predict, context),
        ) as r:
            r.raise_for_status()
            # NDJSON stream from Ollama; use async iterator for httpx.AsyncClient
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def _attach_context(payload: dict, context: list[int] | None) -> dict:
    if context:
        payload["context"] = context  # reuse KV cache from previous turn
    return payload

