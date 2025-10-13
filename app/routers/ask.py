# app/routers/ask.py

import json, logging
from typing import Optional, List
import httpx
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.schemas import AskIn, AskOut, SourceItem
from app.config import get_settings
from app.clients.opensearch_client import os_async_client, os_headers, os_auth
from app.services.search_service import build_search_payload, collect_os_items
from app.services.context_service import build_context_and_sources
from app.services.prompt_service import build_prompt
from app.clients.ollama_client import generate_once, stream_generate

logger = logging.getLogger("farm-assistant.router")
S = get_settings()
router = APIRouter()

@router.post("/ask", response_model=AskOut)
async def ask(inp: AskIn) -> AskOut:
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
                meta={"search": search_payload, "upstream_status": e.response.status_code,
                      "upstream_body_snippet": body}
            )

    top_k = inp.top_k if inp.top_k is not None else S.TOP_K
    contexts, sources = build_context_and_sources(
        items=items,
        question=inp.question,
        top_k=top_k,
        max_context_chars=S.MAX_CONTEXT_CHARS
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

    return AskOut(answer=answer, used_context=contexts, sources=sources, meta={"search": search_payload})


@router.get("/ask/stream")
async def ask_stream(
    q: str,
    page: int = 1,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    async def emit(event: str, data):
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        yield {"event": event, "data": data}

    async def gen():
        user_q = (q or "").strip()
        if not user_q:
            async for x in emit("error", {"message": "Empty question"}):
                yield x
            return

        async for x in emit("status", {"stage": "start", "message": "Received query"}):
            yield x

        inp = AskIn(question=user_q, page=page, k=k, top_k=top_k,
                    max_tokens=max_tokens, temperature=temperature)
        search_payload = build_search_payload(inp)
        headers = os_headers()
        auth = os_auth()
        last_page = inp.page if (inp.page is not None and inp.page >= 1) else 1
        pages = list(range(1, last_page + 1))

        async for x in emit("status", {"stage": "search", "message": "Searching sources…"}):
            yield x

        try:
            async with os_async_client(timeout=30.0) as client:
                items = await collect_os_items(client, search_payload, pages, headers, auth)
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:400]
            async for x in emit("error", {"stage": "search", "status": e.response.status_code, "body": body}):
                yield x
            return

        async for x in emit("status", {"stage": "context", "message": "Preparing context…"}):
            yield x

        t_k = inp.top_k if inp.top_k is not None else S.TOP_K
        contexts, sources = build_context_and_sources(
            items=items, question=user_q, top_k=t_k, max_context_chars=S.MAX_CONTEXT_CHARS
        )
        brief_sources = [
            {
                "sid": s.sid,
                "id": s.id,
                "title": s.title,
                "url": (s.display_url or s.url),
                "score": s.score,
                "license": s.license,
            }
            for s in sources
        ]

        async for x in emit("sources", brief_sources):
            yield x

        if not contexts:
            async for x in emit("done", {"answer": "", "reason": "no_context"}):
                yield x
            return

        async for x in emit("status", {"stage": "llm", "message": "Generating response..."}):
            yield x

        _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
        _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
        prompt = build_prompt(contexts, user_q)

        try:
            _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
            _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE

            prompt = build_prompt(contexts, user_q)

            rounds = 0
            max_rounds = 3  # initial + 2 continuations
            ctx = None  # will hold Ollama's context array from the last chunk

            while rounds < max_rounds:
                rounds += 1
                try:
                    async for obj in stream_generate(
                            prompt if rounds == 1 else "Continue.",
                            _temperature,
                            _max_tokens,
                            context=ctx
                    ):
                        # forward tokens
                        if "response" in obj and obj["response"]:
                            async for x in emit("token", obj["response"]):
                                yield x

                        # remember context if provided
                        if "context" in obj and obj["context"]:
                            ctx = obj["context"]

                        if obj.get("done"):
                            # emit stats
                            stats = {k: v for k, v in obj.items() if k not in ("response", "prompt")}
                            async for x in emit("stats", stats):
                                yield x

                            # continue only if it stopped due to token limit
                            if obj.get("done_reason") == "length" and rounds < max_rounds:
                                # loop once more with "Continue."
                                break
                            else:
                                # finished naturally
                                rounds = max_rounds
                                break
                except httpx.HTTPStatusError as e:
                    async for x in emit("error", {"stage": "llm", "status": e.response.status_code}):
                        yield x
                    return
        except httpx.HTTPStatusError as e:
            async for x in emit("error", {"stage": "llm", "status": e.response.status_code}):
                yield x
            return

        async for x in emit("done", {"message": "complete"}):
            yield x

    return EventSourceResponse(gen())
