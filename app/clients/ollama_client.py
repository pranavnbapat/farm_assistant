# app/clients/ollama_client.py

import httpx, json

from typing import Dict, Any

from app.config import get_settings

S = get_settings()
_transport = httpx.AsyncHTTPTransport(http2=False, retries=0)

def build_gen_payload(prompt: str, temperature: float, num_predict: int, model: str | None = None, num_ctx: int | None = None) -> Dict[str, Any]:
    return {
        "model": (model or S.LLM_MODEL),
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": (num_ctx or S.NUM_CTX),
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.2,
            "keep_alive": "30m",
        },
        "stream": False
    }

def build_stream_payload(prompt: str, temperature: float, num_predict: int,
                         context: list[int] | None = None,
                         model: str | None = None,
                         num_ctx: int | None = None) -> Dict[str, Any]:
    p = {
        "model": (model or S.LLM_MODEL),
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": (num_ctx or S.NUM_CTX),
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.2,
            "keep_alive": "30m",
        },
        "stream": True
    }
    return _attach_context(p, context)

async def generate_once(prompt: str, temperature: float, num_predict: int,
                        model: str | None = None, num_ctx: int | None = None) -> str:
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=1800.0, pool=None)
    async with httpx.AsyncClient(base_url=S.OLLAMA_URL, timeout=timeout, verify=S.VERIFY_SSL, transport=_transport,
                                 headers={"Connection": "keep-alive"}, trust_env=False) as client:
        r = await client.post(f"{S.OLLAMA_URL}/api/generate",
                              json=build_gen_payload(prompt, temperature, num_predict, model=model, num_ctx=num_ctx))
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

async def stream_generate(prompt: str, temperature: float, num_predict: int,
                          context: list[int] | None = None,
                          model: str | None = None,
                          num_ctx: int | None = None):
    timeout = httpx.Timeout(connect=30.0, read=3600.0, write=300.0, pool=None)
    async with httpx.AsyncClient(base_url=S.OLLAMA_URL, timeout=timeout, verify=S.VERIFY_SSL, transport=_transport,
                                 headers={"Connection": "keep-alive"}, trust_env=False) as client:
        async with client.stream(
                "POST",
                f"{S.OLLAMA_URL}/api/generate",
                json=build_stream_payload(prompt, temperature, num_predict, context, model=model, num_ctx=num_ctx),
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

