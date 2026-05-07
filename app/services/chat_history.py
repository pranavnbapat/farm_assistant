# app/services/chat_history.py

import logging
import httpx

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


def _estimate_tokens(text: str) -> int:
    # Same heuristic as ask.py: ~4 chars per token for mixed text.
    return max(1, len(text or "") // 4)


def format_history(messages: list[dict], max_tokens: int = 1000) -> str:
    """
    Turn [{role, content}, ...] into a compact turn-by-turn snippet, budgeted
    by *estimated tokens* rather than raw chars. Char-budgets understate cost
    on multi-byte content (CJK, accented Latin) and overstate it on dense ASCII;
    the rough 4-chars/token rule is closer for both.

    We walk newest -> oldest until we hit `max_tokens`, then reverse back to
    chronological order. This output is consumed only by small helper LLM
    calls (turn-context resolve, turn-strategy router); the main streaming
    prompt now uses structured chat messages directly.
    """
    buf: list[str] = []
    total = 0

    for msg in reversed(messages):
        role = msg.get("role") or "user"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        line = f"{role.capitalize()}: {content}"
        line_tokens = _estimate_tokens(line)
        if total + line_tokens > max_tokens:
            break
        buf.append(line)
        total += line_tokens

    buf.reverse()
    return "\n".join(buf)


def merge_messages(
    backend_messages: list[dict] | None,
    client_messages: list[dict] | None,
) -> list[dict]:
    """
    Merge persisted session messages with recent client-side messages.
    This avoids losing the latest turns when backend persistence is slightly behind.

    Strategy:
    - Keep backend history as the stable base.
    - Append only client messages not already present in-order near the tail.
    - Compare on `(role, content)` because that is what prompt/history construction uses.
    """
    merged = [m for m in (backend_messages or []) if isinstance(m, dict)]
    recent = [m for m in (client_messages or []) if isinstance(m, dict)]
    if not recent:
        return merged

    existing_pairs = [
        (
            (m.get("role") or "user").strip().lower(),
            (m.get("content") or "").strip(),
        )
        for m in merged
        if (m.get("content") or "").strip()
    ]

    scan_start = 0
    for msg in recent:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        pair = (role, content)
        found_at = -1
        for idx in range(scan_start, len(existing_pairs)):
            if existing_pairs[idx] == pair:
                found_at = idx
                break
        if found_at >= 0:
            scan_start = found_at + 1
            continue

        merged.append({"role": role, "content": content})
        existing_pairs.append(pair)
        scan_start = len(existing_pairs)

    return merged
