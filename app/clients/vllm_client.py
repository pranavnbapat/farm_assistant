# app/clients/vllm_client.py

import base64
import httpx
import json
import logging

from functools import lru_cache
from typing import Dict, Any, AsyncGenerator, Optional, List

from app.config import get_settings


logger = logging.getLogger("farm-assistant.vllm")
S = get_settings()
logger.info(f"vLLM client initialized with URL: {S.VLLM_URL}, Model: {S.VLLM_MODEL}")
_transport = httpx.AsyncHTTPTransport(http2=False, retries=0)


def _build_headers(api_key: str | None = None) -> Dict[str, str]:
    """Build headers with API key if available."""
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    resolved_api_key = api_key if api_key is not None else S.VLLM_API_KEY
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    return headers


def _wrap_prompt_as_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Fallback for one-shot helpers that pass a single prompt string
    (turn-strategy router, query normalizer, title generator, summarise).
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Anthropic provider (additive, opt-in via LLM_PROVIDER=anthropic).
#
# Everything below only runs when the instance is explicitly configured for
# Anthropic. The default "vllm" path (Qwen3 / EuroLLM / OpenAI-compatible) never
# touches this code, never imports the `anthropic` SDK, and is unchanged.
# ---------------------------------------------------------------------------

def _provider() -> str:
    return (getattr(S, "LLM_PROVIDER", "") or "vllm").strip().lower()


@lru_cache(maxsize=1)
def _anthropic_client():
    # Lazy import: only the Anthropic instance ever imports the SDK, so the
    # vLLM/Qwen instance neither needs the package installed nor pays any cost.
    from anthropic import AsyncAnthropic
    return AsyncAnthropic(api_key=S.ANTHROPIC_API_KEY)


def _anthropic_max_tokens(max_tokens: int) -> int:
    # The Messages API requires a positive max_tokens. The vLLM path uses -1 to
    # mean "auto", so fall back to a configured default in that case.
    if isinstance(max_tokens, int) and max_tokens > 0:
        return max_tokens
    return getattr(S, "ANTHROPIC_MAX_TOKENS", 1024) or 1024


def _anthropic_model(model: str | None) -> str:
    # The pipeline passes its configured vLLM/OpenAI model name (e.g.
    # "qwen3-30b-a3b-awq") into generation calls; that is meaningless to the
    # Anthropic API and 404s. Honor an explicit claude-* override, otherwise
    # always use the instance's ANTHROPIC_MODEL.
    if model and model.lower().startswith("claude"):
        return model
    return S.ANTHROPIC_MODEL


def _configured_top_p() -> float | None:
    try:
        top_p = float(getattr(S, "TOP_P", 0.9))
    except (TypeError, ValueError):
        return 0.9
    if 0 < top_p <= 1:
        return top_p
    return None


def _split_system_and_messages(
    prompt: str,
    messages: Optional[List[Dict[str, str]]],
) -> tuple[Optional[str], List[Dict[str, str]]]:
    """
    Convert the OpenAI-style message list (which carries the system prompt as a
    leading {"role": "system"} entry) into Anthropic's shape: a top-level
    `system` string plus a user/assistant-only `messages` list.
    """
    source = messages if messages is not None else _wrap_prompt_as_messages(prompt)

    system_parts: List[str] = []
    chat: List[Dict[str, str]] = []
    for m in source:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue
        chat.append({"role": role, "content": content})

    if not chat:
        chat = [{"role": "user", "content": prompt or ""}]

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, chat


async def _anthropic_generate_once(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None,
    messages: Optional[List[Dict[str, str]]],
) -> str:
    system_text, chat = _split_system_and_messages(prompt, messages)
    # Anthropic rejects sending both `temperature` and `top_p` ("cannot both be
    # specified for this model"). We drive sampling with `temperature` (the app's
    # configured knob), so we deliberately omit `top_p` here.
    kwargs: Dict[str, Any] = {
        "model": _anthropic_model(model),
        "max_tokens": _anthropic_max_tokens(max_tokens),
        "temperature": temperature,
        "messages": chat,
    }
    if system_text:
        kwargs["system"] = system_text

    resp = await _anthropic_client().messages.create(**kwargs)
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()


async def _anthropic_stream_generate(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None,
    messages: Optional[List[Dict[str, str]]],
) -> AsyncGenerator[Dict[str, Any], None]:
    system_text, chat = _split_system_and_messages(prompt, messages)
    # See _anthropic_generate_once: omit `top_p` so we don't send it alongside
    # `temperature`, which Anthropic rejects for this model.
    kwargs: Dict[str, Any] = {
        "model": _anthropic_model(model),
        "max_tokens": _anthropic_max_tokens(max_tokens),
        "temperature": temperature,
        "messages": chat,
    }
    if system_text:
        kwargs["system"] = system_text

    async with _anthropic_client().messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            if text:
                yield {"response": text}

        final = await stream.get_final_message()
        usage = getattr(final, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None
        # Map to the same OpenAI-style usage keys the vLLM path emits, so
        # downstream consumers are provider-agnostic.
        yield {
            "done": True,
            "done_reason": "stop",
            "response": "",
            "usage": {
                "prompt_tokens": in_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": (in_tokens or 0) + (out_tokens or 0),
            },
        }


def _is_gpt5_style(model: str | None) -> bool:
    """
    OpenAI's GPT-5 family on /v1/chat/completions rejects `max_tokens`
    (requires `max_completion_tokens`) and only accepts the default
    `temperature`/`top_p` (custom values 400). Detect by model name, or force
    via OPENAI_GPT5_PARAM_STYLE for compatible deployments.
    """
    if getattr(S, "OPENAI_GPT5_PARAM_STYLE", False):
        return True
    name = (model or S.VLLM_MODEL or "").lower()
    return name.startswith("gpt-5")


def build_gen_payload(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build payload for non-streaming generation."""
    payload: Dict[str, Any] = {
        "model": model or S.VLLM_MODEL,
        "messages": messages if messages is not None else _wrap_prompt_as_messages(prompt),
    }

    if _is_gpt5_style(model):
        # GPT-5 family: no temperature/top_p, and the token cap is named differently.
        if max_tokens > 0:
            payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        top_p = _configured_top_p()
        if top_p is not None:
            payload["top_p"] = top_p
        # Only add max_tokens if it's positive (vLLM default is auto)
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

    return payload


def build_stream_payload(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build payload for streaming generation."""
    payload = build_gen_payload(prompt, temperature, max_tokens, model, messages=messages)
    payload["stream"] = True
    # Ask the OpenAI-compatible vLLM server to append a final chunk carrying real
    # token usage (prompt/completion/total). That chunk has an empty `choices`
    # list, so it does not affect the token streaming loop in `stream_generate`.
    payload["stream_options"] = {"include_usage": True}
    return payload


async def generate_once(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str | None = None,
    num_ctx: int | None = None,  # Compatibility with Ollama interface, not used in vLLM
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Non-streaming one-shot generation.
    If `messages` is provided, it is used as the OpenAI chat-completions array
    verbatim and `prompt` is ignored.
    """
    if _provider() == "anthropic":
        return await _anthropic_generate_once(prompt, temperature, max_tokens, model, messages)

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
            json=build_gen_payload(prompt, temperature, max_tokens, model, messages=messages)
        )
        r.raise_for_status()
        data = r.json()

        # Extract content from OpenAI response format
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            return message.get("content", "").strip()
        return ""


async def generate_vision_once(
    *,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    temperature: float = 0.1,
    max_tokens: int = 500,
    model: str | None = None,
) -> str:
    """
    One-shot vision generation against the configured vision-capable OpenAI-style
    chat endpoint.
    """
    base_url = (S.RUNPOD_VLLM_VISION_HOST or S.VLLM_URL).rstrip("/")
    resolved_model = model or S.VLLM_VISION_MODEL or S.VLLM_MODEL
    if not base_url or not resolved_model:
        raise RuntimeError("Vision model endpoint is not configured.")

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a careful agricultural vision assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=None)
    url = f"{base_url}/v1/chat/completions"

    async with httpx.AsyncClient(
        timeout=timeout,
        verify=S.VERIFY_SSL,
        transport=_transport,
        headers=_build_headers(S.VLLM_VISION_API_KEY),
        trust_env=False,
    ) as client:
        r = await client.post(
            url,
            json=build_gen_payload(
                prompt="",
                temperature=temperature,
                max_tokens=max_tokens,
                model=resolved_model,
                messages=messages,
            ),
        )
        r.raise_for_status()
        data = r.json()
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
    num_ctx: int | None = None,  # Compatibility with Ollama, not used in vLLM
    messages: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming generation using Server-Sent Events.
    If `messages` is provided, it is used as the OpenAI chat-completions array
    verbatim and `prompt` is ignored.
    Yields dicts with 'response' for tokens and 'done' when complete.
    """
    if _provider() == "anthropic":
        async for chunk in _anthropic_stream_generate(prompt, temperature, max_tokens, model, messages):
            yield chunk
        return

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
            payload = build_stream_payload(prompt, temperature, max_tokens, model, messages=messages)
            logger.debug(f"Request payload: {json.dumps(payload)[:500]}...")
            
            async with client.stream(
                "POST",
                url,
                json=payload,
            ) as r:
                logger.info(f"vLLM response status: {r.status_code}")
                if r.status_code >= 400:
                    # Read the error body while the stream is still open, then
                    # surface the real status/body. Accessing .text on an unread
                    # streaming response would raise httpx.ResponseNotRead and
                    # mask the actual upstream error (e.g. an OpenAI 400).
                    await r.aread()
                    detail = r.text
                    logger.error(f"vLLM HTTP error {r.status_code}: {detail[:500]}")
                    raise RuntimeError(
                        f"Upstream LLM returned HTTP {r.status_code}: {detail[:300]}"
                    )

                # vLLM/OpenAI SSE format: data: {...}
                token_count = 0
                usage = None
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
                    
                    # With stream_options.include_usage, vLLM appends a trailing
                    # chunk whose `choices` is empty and which carries real token
                    # usage. Keep the latest non-null usage we see.
                    if data.get("usage"):
                        usage = data["usage"]

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
                    "usage": usage,
                }
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to vLLM at {url}: {e}")
        raise
    except httpx.HTTPStatusError as e:
        # Defensive: the status check above already surfaces error bodies; this
        # only fires if a future code path reintroduces raise_for_status.
        logger.error(f"vLLM HTTP error {e.response.status_code}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in vLLM streaming: {e}")
        raise
