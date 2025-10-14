# app/routers/ask.py

import json
import logging
import re as _re

import httpx

from typing import Optional, List

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.clients.ollama_client import generate_once, stream_generate
from app.clients.opensearch_client import os_async_client, os_headers, os_auth
from app.config import get_settings
from app.services.search_service import build_search_payload, collect_os_items
from app.services.context_service import build_context_and_sources
from app.services.prompt_service import build_prompt, build_summary_prompt
from app.schemas import AskIn, AskOut, SourceItem, SummariseIn, SummariseOut

logger = logging.getLogger("farm-assistant.router")
S = get_settings()
router = APIRouter()

@router.post("/ask", response_model=AskOut)
async def ask(inp: AskIn) -> AskOut:
    """
    Non-streaming endpoint (kept for API parity). Returns the whole answer at once.
    """

    if not inp.question or not inp.question.strip():
        return AskOut(answer="Please provide a question.", used_context=[], meta={})

    headers = os_headers()
    auth = os_auth()
    search_payload = build_search_payload(inp)
    last_page = inp.page if (inp.page is not None and inp.page >= 1) else 1
    pages = list(range(1, last_page + 1))

    async with os_async_client(timeout=30.0) as client:
        try:
            items = await collect_os_items(client, search_payload, pages, headers, auth)
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:400]
            logger.error(f"OpenSearch error {e.response.status_code}: {body}")
            return AskOut(
                answer="Search upstream error.",
                used_context=[],
                meta={
                    "search": search_payload,
                    "upstream_status": e.response.status_code,
                    "upstream_body_snippet": body,
                },
            )

    top_k = inp.top_k if inp.top_k is not None else S.TOP_K
    contexts, sources = build_context_and_sources(
        items=items,
        question=inp.question,
        top_k=top_k,
        max_context_chars=S.MAX_CONTEXT_CHARS,
    )
    if not contexts:
        return AskOut(
            answer="I don't know based on the current context.",
            used_context=[],
            sources=sources,
            meta={"search": search_payload},
        )

    max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
    temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
    prompt = build_prompt(contexts, inp.question)
    answer = await generate_once(prompt, temperature, max_tokens)

    return AskOut(
        answer=answer, used_context=contexts, sources=sources, meta={"search": search_payload}
    )


@router.get("/ask/stream")
async def ask_stream(
    q: str,
    page: int = 1,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Emits: status, token, stats, (sources after finish), done, error.
    """

    async def emit(event: str, data):
        """Yield one SSE event dict with JSON-serialised data."""
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        yield {"event": event, "data": data}

    async def gen():
        user_q = (q or "").strip()
        if not user_q:
            async for x in emit("error", {"message": "Empty question"}):
                yield x
            return

        # --- Start
        async for x in emit("status", {"stage": "Start", "message": "Received query"}):
            yield x

        # --- Search
        inp = AskIn(
            question=user_q,
            page=page,
            k=k,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        search_payload = build_search_payload(inp)
        headers = os_headers()
        auth = os_auth()
        last_page = inp.page if (inp.page is not None and inp.page >= 1) else 1
        pages = list(range(1, last_page + 1))

        async for x in emit("status", {"stage": "Search", "message": "Searching sources..."}):
            yield x

        try:
            async with os_async_client(timeout=30.0) as client:
                items = await collect_os_items(client, search_payload, pages, headers, auth)
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:400]
            async for x in emit(
                "error",
                {"stage": "search", "status": e.response.status_code, "body": body},
            ):
                yield x
            return

        # --- Build contexts
        async for x in emit("status", {"stage": "Context", "message": "Preparing context..."}):
            yield x

        t_k = inp.top_k if inp.top_k is not None else S.TOP_K
        contexts, sources = build_context_and_sources(
            items=items, question=user_q, top_k=t_k, max_context_chars=S.MAX_CONTEXT_CHARS
        )

        # Keep the complete list for later filtering by citations
        all_sources = [
            {
                "n": i + 1,  # stable numeric index for [1], [2] ...
                "sid": getattr(s, "sid", None),
                "id": getattr(s, "id", None),
                "title": getattr(s, "title", None),
                "url": (getattr(s, "display_url", None) or getattr(s, "url", None)),
                "license": getattr(s, "license", None),
            }
            for i, s in enumerate(sources)
        ]

        if not contexts:
            async for x in emit("done", {"answer": "", "reason": "no_context"}):
                yield x
            return

        # --- LLM stream
        async for x in emit("status", {"stage": "LLM", "message": "Generating response..."}):
            yield x

        _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
        _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
        prompt = build_prompt(contexts, user_q)

        try:
            rounds = 0
            max_rounds = 3  # initial + 2 continuations
            ctx = None      # holds Ollama's context array from the last chunk
            answer_chunks: List[str] = []
            last_done_reason = None

            # Keep requesting more output until model decides to stop for a reason other than length.
            # Put a generous safety cap to prevent accidental infinite loops (e.g. 100 hops).
            hops = 0
            MAX_HOPS = 100

            while hops < MAX_HOPS:
                hops += 1

                # For first hop, send the full prompt; subsequent hops: continue from context with empty prompt
                _prompt = prompt if hops == 1 else ""

                try:
                    async for obj in stream_generate(
                            _prompt,
                            _temperature,
                            _max_tokens,
                            context=ctx,
                            model=model,
                            num_ctx=S.NUM_CTX,
                    ):
                        # stream tokens
                        if "response" in obj and obj["response"]:
                            chunk = obj["response"]
                            answer_chunks.append(chunk)
                            async for x in emit("token", chunk):
                                yield x

                        # capture kv-cache context
                        if "context" in obj and obj["context"]:
                            ctx = obj["context"]

                        if obj.get("done"):
                            # forward stats
                            stats = {k: v for k, v in obj.items() if k not in ("response", "prompt")}
                            async for x in emit("stats", stats):
                                yield x

                            last_done_reason = obj.get("done_reason")  # e.g. "length", "stop", "unload"
                            break
                    # if not length-limited, we're done
                    if last_done_reason != "length":
                        break
                except httpx.HTTPStatusError as e:
                    async for x in emit("error", {"stage": "LLM", "status": e.response.status_code}):
                        yield x
                    return

        except httpx.HTTPStatusError as e:
            async for x in emit("error", {"stage": "LLM", "status": e.response.status_code}):
                yield x
            return

        # --- After streaming: emit only the sources actually cited in the text
        full_text = "".join(answer_chunks)

        # Normalise assorted "source" forms to [num], e.g. "(source S2)", "source [3]", "source 4"
        norm_text = _re.sub(r"\(\s*source[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\s*\)", r"[\1]", full_text, flags=_re.IGNORECASE)
        norm_text = _re.sub(r"\bsource[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\b", r"[\1]", norm_text, flags=_re.IGNORECASE)

        # Collect citations from patterns like [1], [ 1 ], [1][3], [1, 2], [1,2,3]
        cited_nums = set()

        # 2.1 Simple/adjacent forms: [1] and also [ 1 ]
        for m in _re.finditer(r"\[\s*(\d+)\s*\]", norm_text):
            cited_nums.add(int(m.group(1)))

        # 2.2 Comma list inside one bracket pair: [1, 2, 5]
        for m in _re.finditer(r"\[\s*([\d\s,–-]+)\s*\]", norm_text):
            blob = m.group(1)
            for tok in _re.split(r"[,\s–-]+", blob):
                if tok.isdigit():
                    cited_nums.add(int(tok))

        if cited_nums:
            by_num = {s["n"]: s for s in all_sources}
            cited_sources = [by_num[n] for n in sorted(cited_nums) if n in by_num]
            if cited_sources:
                async for x in emit("sources", cited_sources):
                    yield x
        else:
            # Fallback: if the model didn't cite explicitly, send top few so UI isn’t empty
            fallback = all_sources[: min(len(all_sources), 5)]
            if fallback:
                async for x in emit("sources", fallback):
                    yield x

        # --- Done
        async for x in emit("done", {"message": "complete"}):
            yield x

    # Keep the SSE connection alive with periodic heartbeats
    return EventSourceResponse(gen(), ping=10)


@router.post("/summarise", response_model=SummariseOut)
async def summarise(inp: SummariseIn) -> SummariseOut:
    """
    Summarise a single text chunk according to a user-supplied prompt.
    No search/context retrieval; direct LLM call.
    """
    user_prompt = (inp.prompt or "").strip()
    text_chunk = (inp.text or "").strip()

    if not user_prompt:
        return SummariseOut(summary="", meta={"error": "Empty prompt"})
    if not text_chunk:
        return SummariseOut(summary="", meta={"error": "Empty text"})

    # Build a minimal, safe prompt around the supplied text
    prompt = build_summary_prompt(user_prompt, text_chunk)

    # Use request overrides if provided, otherwise fall back to settings
    max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
    temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE

    # Direct call to Ollama (non-streaming)
    try:
        summary = await generate_once(prompt, temperature, max_tokens)
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:400]
        logger.error(f"LLM error {e.response.status_code}: {body}")
        return SummariseOut(
            summary="",
            meta={
                "upstream_status": e.response.status_code,
                "upstream_body_snippet": body,
            },
        )

    return SummariseOut(
        summary=summary,
        meta={
            "model": S.LLM_MODEL,
            "num_ctx": S.NUM_CTX,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
