# app/routers/ask.py

import asyncio
import json
import logging
import re as _re
import time

import httpx

from typing import Optional, List

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

# from app.clients import hf_local_client
from app.clients.ollama_client import generate_once, stream_generate
from app.clients.opensearch_client import os_async_client, os_headers, os_auth
from app.config import get_settings
from app.services.chat_history import load_chat_state, format_history, save_chat_state, CHAT_BACKEND_URL
from app.services.context_service import build_context_and_sources
from app.services.prompt_service import build_prompt, build_summary_prompt, build_generic_prompt, build_title_prompt
from app.services.search_service import build_search_payload, collect_os_items
from app.schemas import AskIn, SummariseIn, SummariseOut
from app.utils.response_cache import make_key, get_cached, set_cached, hash_contexts

logger = logging.getLogger("farm-assistant.router")
S = get_settings()
router = APIRouter()


def _cache_params(_temp, _max, _model):
    return {
        "temperature": _temp,
        "max_tokens": _max,
        "model": (_model or S.LLM_MODEL),
        "num_ctx": S.NUM_CTX,
        "top_p": 0.9,
    }

SYSTEM_PROMPT_TAG_RAG = "rag-v1"
SYSTEM_PROMPT_TAG_GENERIC = "generic-v1"


async def _maybe_update_session_title(session_id: str, question: str, answer: str | None = None):
    if not session_id or not CHAT_BACKEND_URL:
        return

    title_prompt = build_title_prompt(question, answer)
    S = get_settings()
    try:
        # small, low-temperature call – we just want a short stable title
        raw_title = await generate_once(
            title_prompt,
            temperature=0.2,
            num_predict=32,
        )
    except httpx.HTTPError:
        return

    # post-process: single line, trim length
    title = (raw_title or "").strip()
    # drop newlines if model returns extra text
    title = title.splitlines()[0]
    title = title.strip(" \"'")[:120]

    if not title:
        return

    url = f"{CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    payload = {"title": title}

    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
        try:
            await client.patch(url, json=payload)
        except httpx.HTTPError:
            # best-effort only
            return

@router.get("/ask/stream")
async def ask_stream(
    q: str,
    page: int = 1,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    request: Request = None,
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

        t0 = time.perf_counter()

        # --- Load conversation state for this session (if provided) ---
        history_text: str = ""
        initial_llm_ctx: list[int] | None = None

        state = {"messages": [], "llm_context": None}
        if session_id:
            # shape: {"messages": [...], "llm_context": [...]}
            state = await load_chat_state(session_id)

        history_text = format_history(state.get("messages", []))
        initial_llm_ctx = state.get("llm_context")

        state = {"messages": [], "llm_context": None}
        if session_id:
            state = await load_chat_state(session_id)

        history_text = format_history(state.get("messages", []))
        initial_llm_ctx = state.get("llm_context")

        # is this the first turn of this chat?
        is_first_turn = not state.get("messages")

        # Concurrency gate helpers
        sem = getattr(request.app.state, "gen_semaphore", None)

        async def _acquire_or_queue():
            # If no semaphore configured, proceed
            if sem is None:
                return  # <-- return None (no value)
            # If no permits available, notify client once
            if getattr(sem, "_value", 1) == 0:  # _value is internal; good enough for a hint
                async for x in emit("status", {"stage": "Queue", "message": "Waiting for a free slot..."}):
                    yield x

            await sem.acquire()

        async def _release():
            try:
                if sem is not None:
                    sem.release()
            except Exception:
                pass

        # --- Intent routing ---
        async for x in emit("status", {"stage": "Intent", "message": "Routing query..."}):
            yield x

        intent = {"path": "RAG"}  # default fallback
        try:
            timeout = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
                ir = await client.post(
                    S.INTENT_ROUTER_URL,
                    json={"query": user_q, "force_prepare_if_needed": True},
                )
                print(ir)
                ir.raise_for_status()
                intent = ir.json() or intent
        except httpx.HTTPError as e:
            # Soft-fail to RAG if router is down
            logger.warning(f"Intent router unavailable, falling back to RAG: {e}")

        path = str(intent.get("path") or "RAG").upper()
        print("path: " , path)

        async for x in emit("meta", {"intent": intent}):
            yield x

        if path == "LLM_ONLY":
            # --- LLM-only short-circuit (no search/context) ---
            async for x in emit("status", {"stage": "LLM", "message": "Generating response..."}):
                yield x

            _max_tokens = max_tokens if max_tokens is not None else S.MAX_TOKENS
            _temperature = temperature if temperature is not None else S.TEMPERATURE
            # Include short history so the model can stay consistent
            prompt = build_generic_prompt(user_q, history_text)

            # Response cache lookup (no RAG context)
            params = _cache_params(_temperature, _max_tokens, model)
            ckey = make_key(
                user_q=user_q,
                system_tag=SYSTEM_PROMPT_TAG_GENERIC,
                model_id=params["model"],
                params=params,
                ctx_hash="",  # no context
            )
            cached = get_cached(ckey)
            if cached:
                # Stream cached text to keep UI behaviour
                for tok in cached["answer"].split():
                    async for x in emit("token", tok + " "):
                        yield x
                async for x in emit("sources", []):
                    yield x
                async for x in emit("timing", {"total_ms": int((time.perf_counter() - t0) * 1000),
                                               "search_ms": 0, "context_ms": 0, "llm_ms": 0}):
                    yield x
                async for x in emit("done", {"message": "complete", "cache": True}):
                    yield x
                return

            # Concurrency gate
            async for x in _acquire_or_queue():
                yield x

            t_llm_start = time.perf_counter()
            try:
                ctx = initial_llm_ctx
                answer_chunks: List[str] = []
                last_done_reason = None
                hops = 0
                MAX_HOPS = 100

                while hops < MAX_HOPS:
                    hops += 1
                    _prompt = prompt if hops == 1 else ""
                    async for obj in stream_generate(
                            _prompt, _temperature, _max_tokens,
                            context=ctx, model=model, num_ctx=S.NUM_CTX,
                    ):
                        if "response" in obj and obj["response"]:
                            chunk = obj["response"]
                            answer_chunks.append(chunk)
                            async for x in emit("token", chunk):
                                yield x

                        if "context" in obj and obj["context"]:
                            ctx = obj["context"]

                        if obj.get("done"):
                            stats = {k: v for k, v in obj.items() if k not in ("response", "prompt")}
                            async for x in emit("stats", stats):
                                yield x
                            last_done_reason = obj.get("done_reason")
                            break

                    if last_done_reason != "length":
                        break
            except httpx.HTTPStatusError as e:
                async for x in emit("error", {"stage": "LLM", "status": e.response.status_code}):
                    yield x
                return
            finally:
                await _release()

            t_llm_end = time.perf_counter()
            full_text = "".join(answer_chunks)

            # Persist the latest KV cache so next turn can reuse it
            if session_id:
                await save_chat_state(session_id, ctx)

                # If this is the first turn, ask the LLM to generate a nicer title
                if is_first_turn:
                    # run in the background so it doesn't block streaming completion
                    asyncio.create_task(
                        _maybe_update_session_title(session_id, user_q, full_text)
                    )

            # Save in cache
            set_cached(ckey, full_text, meta={
                "model": params["model"],
                "params": params,
                "system_tag": SYSTEM_PROMPT_TAG_GENERIC,
            })

            # No sources in LLM-only; still send an empty list so UI is deterministic
            async for x in emit("sources", []):
                yield x

            total_ms = int((time.perf_counter() - t0) * 1000)
            search_ms = 0
            context_ms = 0
            llm_ms = int((t_llm_end - t_llm_start) * 1000)
            async for x in emit("timing", {
                "total_ms": total_ms,
                "search_ms": search_ms,
                "context_ms": context_ms,
                "llm_ms": llm_ms
            }):
                yield x

            async for x in emit("done", {"message": "complete"}):
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

        t_search_start = time.perf_counter()

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

        t_search_end = time.perf_counter()

        # --- Build contexts
        async for x in emit("status", {"stage": "Context", "message": "Preparing context..."}):
            yield x

        t_ctx_start = time.perf_counter()
        t_k = inp.top_k if inp.top_k is not None else S.TOP_K
        contexts, sources = build_context_and_sources(
            items=items, question=user_q, top_k=t_k, max_context_chars=S.MAX_CONTEXT_CHARS
        )
        t_ctx_end = time.perf_counter()

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
            # Still emit timing so client can show a think time even when no context
            total_ms = int((time.perf_counter() - t0) * 1000)
            search_ms = int((t_search_end - t_search_start) * 1000)
            context_ms = int((t_ctx_end - t_ctx_start) * 1000)
            llm_ms = 0
            async for x in emit("timing", {
                "total_ms": total_ms,
                "search_ms": search_ms,
                "context_ms": context_ms,
                "llm_ms": llm_ms
            }):
                yield x

            async for x in emit("done", {"answer": "", "reason": "no_context"}):
                yield x
            return

        ctx_h = hash_contexts(contexts)
        _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
        _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
        params = _cache_params(_temperature, _max_tokens, model)

        ckey = make_key(
            user_q=user_q,
            system_tag=SYSTEM_PROMPT_TAG_RAG,
            model_id=params["model"],
            params=params,
            ctx_hash=ctx_h,
            user_scope="",  # fill with tenant/user id if needed later
        )
        cached = get_cached(ckey)
        if cached:
            for tok in cached["answer"].split():
                async for x in emit("token", tok + " "):
                    yield x

            fallback = all_sources[: min(len(all_sources), 5)]
            async for x in emit("sources", fallback):
                yield x

            total_ms = int((time.perf_counter() - t0) * 1000)
            search_ms = int((t_search_end - t_search_start) * 1000)
            context_ms = int((t_ctx_end - t_ctx_start) * 1000)
            async for x in emit("timing",
                                {"total_ms": total_ms, "search_ms": search_ms, "context_ms": context_ms, "llm_ms": 0}):
                yield x
            async for x in emit("done", {"message": "complete", "cache": True}):
                yield x
            return

        # Concurrency gate
        async for x in _acquire_or_queue():
            yield x

        # --- LLM stream
        async for x in emit("status", {"stage": "LLM", "message": "Generating response..."}):
            yield x

        _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
        _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
        prompt = build_prompt(contexts, user_q, history_text)

        t_llm_start = time.perf_counter()
        try:
            ctx = initial_llm_ctx
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
        finally:
            await _release()

        t_llm_end = time.perf_counter()

        # --- After streaming: emit only the sources actually cited in the text
        full_text = "".join(answer_chunks)

        # Persist the latest KV cache so next turn can reuse it
        if session_id:
            await save_chat_state(session_id, ctx)

            if is_first_turn:
                asyncio.create_task(
                    _maybe_update_session_title(session_id, user_q, full_text)
                )

        set_cached(ckey, full_text, meta={
            "model": params["model"],
            "params": params,
            "system_tag": SYSTEM_PROMPT_TAG_RAG,
            "ctx_len": len(contexts),
        })

        # Normalise assorted "source" forms to [num], e.g. "(source S2)", "source [3]", "source 4"
        norm_text = _re.sub(r"\(\s*source[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\s*\)", r"[\1]", full_text,
                            flags=_re.IGNORECASE)
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

        # --- Emit timing
        total_ms = int((time.perf_counter() - t0) * 1000)
        search_ms = int((t_search_end - t_search_start) * 1000)
        context_ms = int((t_ctx_end - t_ctx_start) * 1000)
        llm_ms = int((t_llm_end - t_llm_start) * 1000)
        async for x in emit("timing", {
            "total_ms": total_ms,
            "search_ms": search_ms,
            "context_ms": context_ms,
            "llm_ms": llm_ms
        }):
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
