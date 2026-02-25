# app/routers/ask.py

import asyncio
import base64
import hashlib
import json
import logging
import re as _re
import time

import httpx

from typing import Optional, List

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.clients.vllm_client import generate_once, stream_generate
from app.clients.opensearch_client import os_async_client, os_headers, os_auth
from app.config import get_settings
from app.services.chat_history import load_chat_state, format_history, save_chat_state, CHAT_BACKEND_URL
from app.services.context_service import build_context_and_sources
from app.services.pdf_service import (
    get_docs_for_user,
    build_pdf_contexts,
    ensure_pdf_processed,
    upsert_attachment_to_backend,
    fetch_session_attachments_from_backend,
    docs_from_attachment_records,
)
from app.services.prompt_service import build_prompt, build_summary_prompt, build_title_prompt
from app.services.search_service import build_search_payload, collect_os_items
from app.services.user_profile_service import UserProfileService
from app.schemas import AskIn, SummariseIn, SummariseOut
from app.utils.response_cache import make_key, get_cached, set_cached, hash_contexts

logger = logging.getLogger("farm-assistant.router")
S = get_settings()
router = APIRouter()


def _extract_user_uuid_from_token(auth_token: str) -> Optional[str]:
    """Extract user UUID from JWT token string."""
    if not auth_token.startswith("Bearer "):
        return None
    
    token = auth_token[7:]
    
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        
        decoded = base64.urlsafe_b64decode(payload)
        token_data = json.loads(decoded)
        
        user_id = token_data.get("uuid") or token_data.get("user_id") or token_data.get("sub")
        return str(user_id) if user_id else None
    except Exception:
        return None


def _normalize_model(model: Optional[str]) -> str:
    """Normalize model parameter to use the configured vLLM model."""
    if model and ":" in model:
        logger.warning(f"Received Ollama-style model name '{model}', using VLLM_MODEL '{S.VLLM_MODEL}' instead")
        return S.VLLM_MODEL
    return model or S.VLLM_MODEL


def _cache_params(_temp, _max, _model):
    return {
        "temperature": _temp,
        "max_tokens": _max,
        "model": _normalize_model(_model),
        "num_ctx": S.NUM_CTX,
        "top_p": 0.9,
    }


def _history_scope(history_text: str) -> str:
    """Turn history into a short, stable hash for cache keys."""
    if not history_text:
        return ""
    return hashlib.sha256(history_text.encode("utf-8")).hexdigest()[:16]


SYSTEM_PROMPT_TAG = "farm-assistant-v2"


def _estimate_tokens(text: str) -> int:
    # Fast heuristic: ~4 chars per token for mixed English-like text.
    return max(1, len(text or "") // 4)


def _short_title_2_3_words(raw: str) -> str:
    cleaned = (raw or "").strip().strip(" \"'")
    if not cleaned:
        return ""
    words = [w for w in cleaned.split() if w]
    if not words:
        return ""
    # Keep at most 3 words; if model returns 1 word, keep it as-is.
    return " ".join(words[:3])[:120]


def _is_short_affirmation(text: str) -> bool:
    t = (text or "").strip().lower()
    t = _re.sub(r"[^\w\s]", " ", t)
    t = _re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    tokens = [w for w in _re.split(r"\s+", t) if w]
    if len(tokens) > 4:
        return False
    canonical = {
        "yes", "yea", "yeah", "yep", "yup",
        "yes please", "please", "sure", "sure thing",
        "okay", "ok", "alright", "all right",
        "go ahead", "continue", "more", "tell me more", "sounds good",
    }
    return t in canonical


def _extract_last_assistant_question(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if (msg.get("role") or "").lower() != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if not content or "?" not in content:
            continue
        # take last sentence-like chunk ending with '?'
        parts = _re.split(r"(?<=[\?\!\.])\s+", content)
        for p in reversed(parts):
            p = p.strip()
            if p.endswith("?"):
                return p[:300]
        return content[:300]
    return ""


def _is_file_handoff_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    canonical = {
        "here you go",
        "see this file",
        "check this file",
        "this file",
        "look at this",
        "use this file",
        "read this",
    }
    return t in canonical


def _mentions_file_or_document(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keys = {"file", "document", "pdf", "attachment", "this doc", "this file"}
    return any(k in t for k in keys)


def _should_skip_query_normalization(text: str) -> bool:
    # Avoid accidental translation for non-ASCII queries (e.g., Greek).
    return any(ord(ch) > 127 for ch in (text or ""))


async def _normalize_query_for_retrieval(text: str) -> str:
    """
    Best-effort query cleanup for spelling/grammar to improve retrieval.
    Keeps meaning unchanged and falls back immediately on any error/timeout.
    """
    raw = (text or "").strip()
    if not raw:
        return raw

    prompt = (
        "Rewrite the user query with corrected spelling/grammar, preserving exact intent. "
        "Keep it concise, one line, no explanation.\n\n"
        f"Query: {raw}\n\n"
        "Rewritten query:"
    )
    try:
        rewritten = await asyncio.wait_for(
            generate_once(prompt, temperature=0.0, max_tokens=48),
            timeout=1.2,
        )
    except Exception:
        return raw

    line = (rewritten or "").strip().splitlines()
    if not line:
        return raw
    cleaned = line[0].strip(" \"'")
    return cleaned or raw


async def _maybe_update_session_title(
    session_id: str,
    question: str,
    answer: str | None = None,
    auth_token: str | None = None,
):
    if not session_id or not CHAT_BACKEND_URL:
        return

    title_prompt = build_title_prompt(question, answer)
    try:
        raw_title = await generate_once(
            title_prompt,
            temperature=0.2,
            max_tokens=8,
        )
    except httpx.HTTPError:
        return

    lines = (raw_title or "").strip().splitlines()
    if not lines:
        logger.warning(f"Title generation returned empty output for session {session_id[:8]}...")
        return

    title = _short_title_2_3_words(lines[0])
    if not title:
        return

    url = f"{CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    payload = {"title": title}

    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    headers = {}
    if auth_token:
        headers["Authorization"] = auth_token

    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
        try:
            r = await client.patch(url, json=payload, headers=headers)
            if r.is_error:
                body_snippet = (r.text or "")[:300]
                logger.warning(
                    "Failed to update session title: "
                    f"HTTP {r.status_code}, session={session_id[:8]}..., body={body_snippet}"
                )
                return
            logger.info(f"Updated session title: {r.status_code}")
        except httpx.HTTPError:
            pass


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
    followup_hint: Optional[str] = None,
    doc_ids: Optional[str] = None,
    request: Request = None,
):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Natural conversation flow - sends full history to LLM, lets it handle context.
    """

    async def emit(event: str, data):
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        yield {"event": event, "data": data}

    async def emit_app_error(payload: dict):
        # Use a custom event name to avoid collision with EventSource transport "error".
        async for x in emit("app_error", payload):
            yield x

    normalized_model = _normalize_model(model)

    async def gen():
        user_q = (q or "").strip()
        if not user_q:
            async for x in emit_app_error({"message": "Empty question"}):
                yield x
            return
        if _estimate_tokens(user_q) > S.MAX_USER_INPUT_TOKENS:
            async for x in emit_app_error({
                "message": (
                    f"Question is too long. Limit is ~{S.MAX_USER_INPUT_TOKENS} tokens per message."
                )
            }):
                yield x
            return
        requested_doc_ids = [d.strip() for d in (doc_ids or "").split(",") if d.strip()]

        t0 = time.perf_counter()
        
        # Extract auth info
        auth_token = ""
        if request:
            auth_token = request.headers.get("Authorization", "")
            if not auth_token:
                auth_token = request.query_params.get("auth_token", "")
        
        user_uuid = _extract_user_uuid_from_token(auth_token) if auth_token else None
        
        # Load conversation state
        history_text: str = ""
        initial_llm_ctx: list[int] | None = None

        state = {"messages": [], "llm_context": None}
        if session_id:
            state = await load_chat_state(session_id, auth_token)

        history_text = format_history(state.get("messages", []))
        initial_llm_ctx = state.get("llm_context")
        is_first_turn = not state.get("messages")
        effective_q = user_q
        prompt_q = user_q
        if _is_short_affirmation(user_q):
            prev_q = _extract_last_assistant_question(state.get("messages", []))
            if not prev_q and followup_hint:
                prev_q = (followup_hint or "").strip()[:300]
            if prev_q:
                effective_q = f"User confirmed previous assistant question. Continue from: {prev_q}"
                prompt_q = f"{user_q}\n\nContinue based on this previous assistant question: {prev_q}"

        retrieval_q = effective_q
        if (
            not retrieval_q.startswith("User confirmed previous assistant question. Continue from:")
            and not _should_skip_query_normalization(retrieval_q)
        ):
            retrieval_q = await _normalize_query_for_retrieval(retrieval_q)
        if requested_doc_ids and _is_file_handoff_query(user_q):
            retrieval_q = (
                "Summarize the uploaded PDF(s) with key points and practical takeaways, "
                "then ask one focused follow-up question."
            )

        # Load user profile
        profile_context = ""
        if user_uuid and auth_token:
            try:
                profile = await UserProfileService.get_or_create_profile(user_uuid, auth_token)
                facts = await UserProfileService.get_facts(user_uuid, auth_token, limit=5)
                profile_context = UserProfileService.build_profile_context(profile, facts)
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")

        # Concurrency gate
        sem = getattr(request.app.state, "gen_semaphore", None)

        async def _acquire_or_queue():
            if sem is None:
                return
            if getattr(sem, "_value", 1) == 0:
                async for x in emit("status", {"stage": "Queue", "message": "Waiting for a free slot..."}):
                    yield x
            await sem.acquire()

        async def _release():
            try:
                if sem is not None:
                    sem.release()
            except Exception:
                pass

        # --- Always search - let the LLM decide if results are useful ---
        async for x in emit("status", {"stage": "Search", "message": "Searching sources..."}):
            yield x

        inp = AskIn(
            question=retrieval_q,
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

        t_search_start = time.perf_counter()

        try:
            async with os_async_client(timeout=30.0) as client:
                items = await collect_os_items(client, search_payload, pages, headers, auth)
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:400]
            async for x in emit_app_error({"stage": "search", "status": e.response.status_code, "body": body}):
                yield x
            return

        t_search_end = time.perf_counter()

        # --- Build contexts ---
        async for x in emit("status", {"stage": "Context", "message": "Preparing context..."}):
            yield x

        t_ctx_start = time.perf_counter()
        t_k = inp.top_k if inp.top_k is not None else S.TOP_K
        contexts, sources = build_context_and_sources(
            items=items, question=retrieval_q, top_k=t_k, max_context_chars=S.MAX_CONTEXT_CHARS
        )

        # Merge uploaded PDF contexts (if provided)
        if requested_doc_ids:
            owner_scope = user_uuid or "anonymous"
            pdf_docs = get_docs_for_user(requested_doc_ids, owner_scope)
            if pdf_docs:
                async for x in emit("status", {"stage": "PDF", "message": "Extracting uploaded PDF(s)..."}):
                    yield x
                for d in pdf_docs:
                    await ensure_pdf_processed(d)
                    if session_id and auth_token:
                        asyncio.create_task(
                            upsert_attachment_to_backend(
                                chat_backend_url=CHAT_BACKEND_URL,
                                verify_ssl=S.VERIFY_SSL,
                                auth_token=auth_token,
                                session_uuid=session_id,
                                doc=d,
                            )
                        )
                remaining = max(2000, S.MAX_CONTEXT_CHARS - sum(len(c) for c in contexts))
                pdf_contexts, pdf_sources = build_pdf_contexts(
                    pdf_docs, question=retrieval_q, max_total_chars=remaining
                )
                contexts.extend(pdf_contexts)
                # Add as source-like entries to include in citation mapping
                for s in pdf_sources:
                    sources.append(type("PdfSrc", (), {
                        "sid": None,
                        "id": s.get("id"),
                        "title": s.get("title"),
                        "display_url": None,
                        "url": None,
                        "license": None,
                    })())
        elif session_id and auth_token and _mentions_file_or_document(user_q):
            # Fallback to persisted attachment summaries/text from Django for session continuity.
            records = await fetch_session_attachments_from_backend(
                chat_backend_url=CHAT_BACKEND_URL,
                verify_ssl=S.VERIFY_SSL,
                auth_token=auth_token,
                session_uuid=session_id,
            )
            if records:
                persisted_docs = docs_from_attachment_records(records, owner_id=(user_uuid or "persisted"))
                remaining = max(2000, S.MAX_CONTEXT_CHARS - sum(len(c) for c in contexts))
                pdf_contexts, pdf_sources = build_pdf_contexts(
                    persisted_docs, question=retrieval_q, max_total_chars=remaining
                )
                contexts.extend(pdf_contexts)
                for s in pdf_sources:
                    sources.append(type("PdfSrc", (), {
                        "sid": None,
                        "id": s.get("id"),
                        "title": s.get("title"),
                        "display_url": None,
                        "url": None,
                        "license": None,
                    })())
        t_ctx_end = time.perf_counter()

        all_sources = [
            {
                "n": i + 1,
                "sid": getattr(s, "sid", None),
                "id": getattr(s, "id", None),
                "title": getattr(s, "title", None),
                "url": (getattr(s, "display_url", None) or getattr(s, "url", None)),
                "license": getattr(s, "license", None),
            }
            for i, s in enumerate(sources)
        ]

        # --- Build prompt with conversation history ---
        _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_OUTPUT_TOKENS
        if _max_tokens <= 0 or _max_tokens > S.MAX_OUTPUT_TOKENS:
            _max_tokens = S.MAX_OUTPUT_TOKENS
        _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
        
        prompt = build_prompt(
            contexts=contexts if contexts else [],
            question=prompt_q,
            history=history_text,
            user_profile_context=profile_context
        )
        prompt_tokens = _estimate_tokens(prompt)
        prompt_cap = min(
            S.MAX_INPUT_TOKENS,
            max(256, int(S.NUM_CTX) - int(_max_tokens) - 256),
        )
        if prompt_tokens > prompt_cap:
            async for x in emit_app_error({
                "message": (
                    f"Input context is too large (~{prompt_tokens} tokens). "
                    f"Please shorten your question or reduce attached content."
                )
            }):
                yield x
            return

        # Cache key
        ctx_h = hash_contexts(contexts) if contexts else ""
        params = _cache_params(_temperature, _max_tokens, model)
        history_scope = _history_scope(history_text)

        ckey = make_key(
            user_q=prompt_q,
            system_tag=SYSTEM_PROMPT_TAG,
            model_id=params["model"],
            params=params,
            ctx_hash=ctx_h,
            user_scope=history_scope,
        )
        
        cached = get_cached(ckey)
        if cached:
            for tok in cached["answer"].split():
                async for x in emit("token", tok + " "):
                    yield x
            async for x in emit("sources", []):
                yield x
            total_ms = int((time.perf_counter() - t0) * 1000)
            async for x in emit("timing", {"total_ms": total_ms, "search_ms": 0, "context_ms": 0, "llm_ms": 0}):
                yield x
            async for x in emit("done", {"message": "complete", "cache": True}):
                yield x
            return

        # Concurrency gate
        async for x in _acquire_or_queue():
            yield x

        # --- LLM stream ---
        async for x in emit("status", {"stage": "LLM", "message": "Generating response..."}):
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
                    context=ctx, model=normalized_model, num_ctx=S.NUM_CTX,
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
                    
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to LLM backend: {e}")
            async for x in emit_app_error({"stage": "LLM", "message": f"Cannot connect to LLM: {e}"}):
                yield x
            return
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error {e.response.status_code}: {e.response.text[:500]}")
            async for x in emit_app_error({"stage": "LLM", "status": e.response.status_code}):
                yield x
            return
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            async for x in emit_app_error({"stage": "LLM", "message": f"Error: {str(e)}"}):
                yield x
            return
        finally:
            await _release()

        t_llm_end = time.perf_counter()
        full_text = "".join(answer_chunks)

        # Persist state
        if session_id:
            await save_chat_state(session_id, ctx)
            if is_first_turn:
                asyncio.create_task(
                    _maybe_update_session_title(session_id, user_q, full_text, auth_token)
                )

        # Update profile (fire-and-forget)
        if user_uuid and session_id and auth_token:
            asyncio.create_task(
                UserProfileService.process_conversation_turn(
                    user_uuid, session_id, user_q, full_text, auth_token
                )
            )

        # Cache
        set_cached(ckey, full_text, meta={
            "model": params["model"],
            "params": params,
            "system_tag": SYSTEM_PROMPT_TAG,
            "ctx_len": len(contexts),
        })

        # Extract citations
        norm_text = _re.sub(r"\(\s*source[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\s*\)", r"[\1]", full_text, flags=_re.IGNORECASE)
        norm_text = _re.sub(r"\bsource[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\b", r"[\1]", norm_text, flags=_re.IGNORECASE)
        norm_text = _re.sub(r"\[\s*[sS](\d+)\s*\]", r"[\1]", norm_text)
        norm_text = _re.sub(r"\(\s*[sS](\d+)\s*\)", r"[\1]", norm_text)

        cited_nums = set()
        for m in _re.finditer(r"\[\s*(\d+)\s*\]", norm_text):
            cited_nums.add(int(m.group(1)))
        for m in _re.finditer(r"\[\s*([\d\s,–-]+)\s*\]", norm_text):
            for tok in _re.split(r"[,\s–-]+", m.group(1)):
                if tok.isdigit():
                    cited_nums.add(int(tok))

        if cited_nums:
            by_num = {s["n"]: s for s in all_sources}
            cited_sources = [by_num[n] for n in sorted(cited_nums) if n in by_num]
            if cited_sources:
                async for x in emit("sources", cited_sources):
                    yield x
        else:
            async for x in emit("sources", []):
                yield x

        # Timing
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

        async for x in emit("done", {"message": "complete"}):
            yield x

    return EventSourceResponse(gen(), ping=10)


@router.post("/summarise", response_model=SummariseOut)
async def summarise(inp: SummariseIn) -> SummariseOut:
    """Summarise a single text chunk according to a user-supplied prompt."""
    user_prompt = (inp.prompt or "").strip()
    text_chunk = (inp.text or "").strip()

    if not user_prompt:
        return SummariseOut(summary="", meta={"error": "Empty prompt"})
    if not text_chunk:
        return SummariseOut(summary="", meta={"error": "Empty text"})

    prompt = build_summary_prompt(user_prompt, text_chunk)

    max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
    temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE

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
