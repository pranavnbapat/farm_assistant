# app/services/chat_history.py

import httpx
import logging

from typing import Optional

from app.config import get_settings


S = get_settings()
logger = logging.getLogger("farm-assistant.chat_history")

CHAT_BACKEND_URL = (S.CHAT_BACKEND_URL or "").rstrip("/")

async def load_chat_state(session_id: Optional[str], auth_token: Optional[str] = None) -> dict:
    """Load chat history from Django backend."""
    if not session_id or not CHAT_BACKEND_URL:
        return {"messages": [], "llm_context": None}

    url = f"{CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    
    headers = {}
    if auth_token:
        headers["Authorization"] = auth_token

    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
        try:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            
            data = r.json() or {}
            msgs = data.get("messages", [])
            logger.info(f"Loaded {len(msgs)} messages from session {session_id[:8]}...")
            return {"messages": msgs, "llm_context": None}
            
        except httpx.HTTPStatusError as e:
            # 401/403 means auth issue - log but don't crash
            if e.response.status_code in (401, 403):
                logger.warning(f"Auth failed when loading chat history (HTTP {e.response.status_code}). Continuing without history.")
            else:
                logger.warning(f"Failed to load chat state: HTTP {e.response.status_code}")
            return {"messages": [], "llm_context": None}
            
        except httpx.HTTPError as e:
            logger.warning(f"Failed to load chat state: {e}. Continuing without history.")
            return {"messages": [], "llm_context": None}

async def save_chat_state(session_id: Optional[str], llm_context: Optional[list[int]]) -> None:
    """
    Persist the latest LLM context back to Django for this session.
    Note: vLLM is stateless and doesn't support KV cache passing.
    This is kept for backward compatibility with Ollama.
    """
    if not session_id or not CHAT_BACKEND_URL:
        return

    # vLLM doesn't support context passing, so we skip saving
    if llm_context is None:
        return

    url = f"{CHAT_BACKEND_URL}/api/chat/{session_id}/state/"
    payload = {"llm_context": llm_context}
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)

    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
        try:
            await client.patch(url, json=payload)
        except httpx.HTTPError:
            # Don't break the main flow if state save fails
            return


def format_history(messages: list[dict], max_chars: int = 4000) -> str:
    """
    Turn [{role, content}, ...] into a compact turn-by-turn snippet.

    We walk from newest to oldest until we hit `max_chars`,
    then reverse back to chronological order.
    """
    buf: list[str] = []
    total = 0

    for msg in reversed(messages):
        role = msg.get("role") or "user"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        line = f"{role.capitalize()}: {content}"
        if total + len(line) > max_chars:
            break
        buf.append(line)
        total += len(line)

    buf.reverse()
    return "\n".join(buf)
