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

# from app.clients import hf_local_client
from app.clients.vllm_client import generate_once, stream_generate
from app.clients.opensearch_client import os_async_client, os_headers, os_auth
from app.config import get_settings
from app.services.chat_history import load_chat_state, format_history, save_chat_state, CHAT_BACKEND_URL
from app.services.context_service import build_context_and_sources
from app.services.prompt_service import (
    build_prompt, build_summary_prompt, build_generic_prompt, build_title_prompt,
    build_intent_classification_prompt, build_context_aware_prompt, is_agriculture_related
)
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
    
    token = auth_token[7:]  # Remove "Bearer " prefix
    
    try:
        # JWT tokens are base64 encoded in 3 parts: header.payload.signature
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning(f"Invalid JWT format: {len(parts)} parts instead of 3")
            return None
        
        # Decode payload (middle part)
        import base64
        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        
        decoded = base64.urlsafe_b64decode(payload)
        token_data = json.loads(decoded)
        
        # Try common fields for user ID
        user_id = token_data.get("uuid") or token_data.get("user_id") or token_data.get("sub")
        if user_id:
            logger.debug(f"Extracted user_id from JWT: {user_id}")
        return str(user_id) if user_id else None
    except Exception as e:
        logger.warning(f"Failed to decode JWT: {e}")
        return None


def _normalize_model(model: Optional[str]) -> str:
    """
    Normalize model parameter to use the configured vLLM model.
    If the model looks like an Ollama name (contains ':'), ignore it.
    """
    if model and ":" in model:
        # This looks like an Ollama model name (e.g., "deepseek-llm:7b-chat-q5_K_M")
        # Return the configured vLLM model instead
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
    """
    Turn the current text history into a short, stable hash.

    This ensures the cache key changes when the conversation history changes,
    so responses reflect the actual context of this thread.
    """
    if not history_text:
        return ""
    return hashlib.sha256(history_text.encode("utf-8")).hexdigest()[:16]


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
            max_tokens=32,
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
            r = await client.patch(url, json=payload)
            logger.info(f"Updated session title: {r.status_code}")
        except httpx.HTTPStatusError as e:
            # Log specific error but don't fail
            logger.warning(f"Failed to update session title: HTTP {e.response.status_code} - {e.response.text[:200]}")
        except httpx.HTTPError as e:
            # best-effort only
            logger.warning(f"Failed to update session title: {e}")

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

    # Normalize model parameter upfront
    normalized_model = _normalize_model(model)
    if model != normalized_model:
        logger.info(f"Model normalized from '{model}' to '{normalized_model}'")

    async def gen():
        user_q = (q or "").strip()
        if not user_q:
            async for x in emit("error", {"message": "Empty question"}):
                yield x
            return

        # --- Check domain relevance ---
        is_agri, reason = is_agriculture_related(user_q)
        logger.info(f"Domain check: is_agriculture={is_agri}, reason={reason}, question='{user_q[:50]}...'")
        
        # Note: We don't reject here - we let the LLM handle it with the domain restriction prompt
        # This gives us logging for analytics while allowing the LLM to politely decline

        t0 = time.perf_counter()
        
        # --- Extract auth info early ---
        # Try header first, then query param (for SSE which can't send headers)
        auth_token = ""
        if request:
            auth_token = request.headers.get("Authorization", "")
            # Fallback to query param for SSE
            if not auth_token:
                auth_token = request.query_params.get("auth_token", "")
        
        user_uuid = _extract_user_uuid_from_token(auth_token) if auth_token else None
        
        logger.info(f"Auth extracted: user_uuid={user_uuid}, has_auth={bool(auth_token)}")

        # --- Load conversation state for this session (if provided) ---
        history_text: str = ""
        initial_llm_ctx: list[int] | None = None

        state = {"messages": [], "llm_context": None}
        if session_id:
            # shape: {"messages": [...], "llm_context": [...]}
            state = await load_chat_state(session_id, auth_token)

        history_text = format_history(state.get("messages", []))
        initial_llm_ctx = state.get("llm_context")

        # is this the first turn of this chat?
        is_first_turn = not state.get("messages")

        # --- Load user profile for personalization ---
        profile_context = ""
        
        if user_uuid and auth_token:
            try:
                profile = await UserProfileService.get_or_create_profile(user_uuid, auth_token)
                facts = await UserProfileService.get_facts(user_uuid, auth_token, limit=5)
                profile_context = UserProfileService.build_profile_context(profile, facts)
                logger.info(f"Loaded profile for user {user_uuid}")
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")

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

        # --- Intent routing using LLM ---
        async for x in emit("status", {"stage": "Intent", "message": "Analyzing query..."}):
            yield x

        # Determine if this is a short/ambiguous response that needs context
        last_assistant_msg = ""
        if state.get("messages"):
            # Find the last assistant message
            for msg in reversed(state["messages"]):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break
        
        # Check for ambiguous short responses
        is_ambiguous_short = (
            len(user_q.split()) <= 3 and 
            last_assistant_msg and
            any(word in user_q.lower() for word in ["yes", "no", "please", "more", "go on", "tell me", "why", "how"])
        )
        
        # Build intent classification prompt
        intent_prompt = build_intent_classification_prompt(user_q, history_text if history_text else None)
        
        intent_response = ""
        use_rag = True  # default to RAG for safety
        direct_answer = None
        
        try:
            # Use low temperature for deterministic classification
            intent_response = await generate_once(
                intent_prompt,
                temperature=0.1,
                max_tokens=100,  # Limit to get concise response
                model=normalized_model,
            )
            intent_response = intent_response.strip()
            logger.info(f"Intent classification response: {intent_response[:100]}...")
            
            # Check if LLM wants RAG or provided direct answer
            if intent_response.upper() == "RAG_NEEDED" or "RAG_NEEDED" in intent_response.upper():
                use_rag = True
                logger.info("Intent: RAG_NEEDED")
            else:
                # LLM provided a direct answer
                use_rag = False
                direct_answer = intent_response
                logger.info("Intent: LLM_ONLY (direct answer provided)")
                
        except Exception as e:
            logger.warning(f"Intent classification failed, defaulting to RAG: {e}")
            use_rag = True

        async for x in emit("meta", {"intent": "RAG" if use_rag else "LLM_ONLY", "ambiguous_short": is_ambiguous_short}):
            yield x

        if not use_rag:
            # --- LLM-only short-circuit (no search/context) ---
            async for x in emit("status", {"stage": "LLM", "message": "Generating response..."}):
                yield x

            _max_tokens = max_tokens if max_tokens is not None else S.MAX_TOKENS
            _temperature = temperature if temperature is not None else S.TEMPERATURE
            
            # For ambiguous short responses, use context-aware prompt
            if is_ambiguous_short and last_assistant_msg:
                prompt = build_context_aware_prompt(user_q, last_assistant_msg)
                logger.info(f"Using context-aware prompt for ambiguous response: '{user_q}'")
            else:
                # Include short history and profile so the model can stay consistent
                prompt = build_generic_prompt(user_q, history_text, profile_context)

            # Response cache lookup (no RAG context)
            params = _cache_params(_temperature, _max_tokens, model)
            history_scope = _history_scope(history_text)
            ckey = make_key(
                user_q=user_q,
                system_tag=SYSTEM_PROMPT_TAG_GENERIC,
                model_id=params["model"],
                params=params,
                ctx_hash="",
                user_scope=history_scope,
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
                async for x in emit("error", {"stage": "LLM", "message": f"Cannot connect to LLM: {e}"}):
                    yield x
                return
            except httpx.HTTPStatusError as e:
                logger.error(f"LLM HTTP error {e.response.status_code}: {e.response.text[:500]}")
                async for x in emit("error", {"stage": "LLM", "status": e.response.status_code}):
                    yield x
                return
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}")
                async for x in emit("error", {"stage": "LLM", "message": f"Error: {str(e)}"}):
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

            # No sources in LLM-only; emit null to explicitly clear any previous sources
            async for x in emit("sources", None):
                yield x

            # --- Update user profile with this conversation turn (fire-and-forget) ---
            if user_uuid and session_id and auth_token:
                logger.info(f"Triggering profile update for user {user_uuid} (LLM-only)")
                asyncio.create_task(
                    UserProfileService.process_conversation_turn(
                        user_uuid, session_id, user_q, full_text, auth_token
                    )
                )

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
        history_scope = _history_scope(history_text)

        ckey = make_key(
            user_q=user_q,
            system_tag=SYSTEM_PROMPT_TAG_RAG,
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

            # For cached responses, emit empty sources (original citations not stored)
            async for x in emit("sources", []):
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
        prompt = build_prompt(contexts, user_q, history_text, profile_context)

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
                            model=normalized_model,
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
                except httpx.ConnectError as e:
                    logger.error(f"Cannot connect to LLM backend: {e}")
                    async for x in emit("error", {"stage": "LLM", "message": f"Cannot connect to LLM: {e}"}):
                        yield x
                    return
                except httpx.HTTPStatusError as e:
                    logger.error(f"LLM HTTP error {e.response.status_code}: {e.response.text[:500]}")
                    async for x in emit("error", {"stage": "LLM", "status": e.response.status_code}):
                        yield x
                    return
                except Exception as e:
                    logger.error(f"Unexpected LLM error: {e}")
                    async for x in emit("error", {"stage": "LLM", "message": f"Error: {str(e)}"}):
                        yield x
                    return

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to LLM backend: {e}")
            async for x in emit("error", {"stage": "LLM", "message": f"Cannot connect to LLM: {e}"}):
                yield x
            return
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error {e.response.status_code}: {e.response.text[:500]}")
            async for x in emit("error", {"stage": "LLM", "status": e.response.status_code}):
                yield x
            return
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            async for x in emit("error", {"stage": "LLM", "message": f"Error: {str(e)}"}):
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
        
        # --- Update user profile with this conversation turn (fire-and-forget) ---
        if user_uuid and session_id and auth_token:
            logger.info(f"Triggering profile update for user {user_uuid}")
            asyncio.create_task(
                UserProfileService.process_conversation_turn(
                    user_uuid, session_id, user_q, full_text, auth_token
                )
            )
        else:
            logger.debug(f"Skipping profile update: user_uuid={user_uuid}, session_id={session_id}, has_auth={bool(auth_token)}")

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
            # No citations in the response - don't show any sources
            async for x in emit("sources", []):
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
