# app/clients/vllm_client.py

import httpx
import json
import logging

from typing import Dict, Any, AsyncGenerator

from app.config import get_settings


logger = logging.getLogger("farm-assistant.vllm")
S = get_settings()
logger.info(f"vLLM client initialized with URL: {S.VLLM_URL}, Model: {S.VLLM_MODEL}")
_transport = httpx.AsyncHTTPTransport(http2=False, retries=0)


def _build_headers() -> Dict[str, str]:
    """Build headers with API key if available."""
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    if S.VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {S.VLLM_API_KEY}"
    return headers


def _build_messages(prompt: str) -> list[Dict[str, str]]:
    """Build OpenAI-compatible messages format."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]


def build_gen_payload(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None = None
) -> Dict[str, Any]:
    """Build payload for non-streaming generation."""
    payload: Dict[str, Any] = {
        "model": model or S.VLLM_MODEL,
        "messages": _build_messages(prompt),
        "temperature": temperature,
        "top_p": 0.9,
    }
    # Only add max_tokens if it's positive (vLLM default is auto)
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    return payload


def build_stream_payload(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None = None
) -> Dict[str, Any]:
    """Build payload for streaming generation."""
    payload = build_gen_payload(prompt, temperature, max_tokens, model)
    payload["stream"] = True
    return payload


async def generate_once(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None = None,
    num_ctx: int | None = None  # Compatibility with Ollama interface, not used in vLLM
) -> str:
    """
    Non-streaming one-shot generation.
    Compatible with ollama_client.generate_once interface.
    """
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=1800.0, pool=None)
    url = f"{S.VLLM_URL}/v1/chat/completions"
    
    async with httpx.AsyncClient(
        timeout=timeout,
        verify=S.VERIFY_SSL,
        transport=_transport,
        headers=_build_headers(),
        trust_env=False
    ) as client:
        r = await client.post(
            url,
            json=build_gen_payload(prompt, temperature, max_tokens, model)
        )
        r.raise_for_status()
        data = r.json()
        
        # Extract content from OpenAI response format
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            return message.get("content", "").strip()
        return ""


async def stream_generate(
    prompt: str,
    temperature: float,
    max_tokens: int,
    context: list[int] | None = None,  # Compatibility with Ollama, not used in vLLM
    model: str | None = None,
    num_ctx: int | None = None  # Compatibility with Ollama, not used in vLLM
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming generation using Server-Sent Events.
    Compatible with ollama_client.stream_generate interface.
    Yields dicts with 'response' for tokens and 'done' when complete.
    """
    timeout = httpx.Timeout(connect=30.0, read=3600.0, write=300.0, pool=None)
    url = f"{S.VLLM_URL}/v1/chat/completions"
    
    logger.info(f"Starting vLLM stream to {url}, model: {model or S.VLLM_MODEL}")
    
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            verify=S.VERIFY_SSL,
            transport=_transport,
            headers=_build_headers(),
            trust_env=False
        ) as client:
            payload = build_stream_payload(prompt, temperature, max_tokens, model)
            logger.debug(f"Request payload: {json.dumps(payload)[:500]}...")
            
            async with client.stream(
                "POST",
                url,
                json=payload,
            ) as r:
                logger.info(f"vLLM response status: {r.status_code}")
                r.raise_for_status()
                
                # vLLM/OpenAI SSE format: data: {...}
                token_count = 0
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    # Check for stream end marker
                    if data_str.strip() == "[DONE]":
                        logger.info(f"Stream completed, tokens received: {token_count}")
                        break
                    
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE data: {e}, data: {data_str[:100]}")
                        continue
                    
                    # Extract delta content from OpenAI streaming format
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            token_count += 1
                            yield {"response": content}
                
                # Yield completion marker with stats (similar to Ollama format)
                yield {
                    "done": True,
                    "done_reason": "stop",
                    "response": "",
                }
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to vLLM at {url}: {e}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM HTTP error {e.response.status_code}: {e.response.text[:500]}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in vLLM streaming: {e}")
        raise
