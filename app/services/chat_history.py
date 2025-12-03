# app/services/chat_history.py

import httpx

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from app.config import get_settings

S = get_settings()

CHAT_BACKEND_URL = (S.CHAT_BACKEND_URL or "").rstrip("/")

# One user–assistant exchange
@dataclass
class ChatTurn:
    user: str
    assistant: str

@dataclass
class ChatState:
    turns: List[ChatTurn] = field(default_factory=list)
    llm_context: Optional[list[int]] = None  # Ollama KV cache from last reply

# Very simple in-memory store – fine for now.
# For prod you can replace this with Redis or your Django DB.
_STORE: Dict[str, ChatState] = {}

# Keep only the last N turns to avoid gigantic prompts
_MAX_TURNS = 10


async def load_chat_state(session_id: Optional[str]) -> dict:
    if not session_id or not CHAT_BACKEND_URL:
        return {"messages": [], "llm_context": None}

    url = f"{CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)

    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
        try:
            r = await client.get(url)
            r.raise_for_status()
        except httpx.HTTPError:
            return {"messages": [], "llm_context": None}

        data = r.json() or {}

        msgs = data.get("messages", [])

        return {
            "messages": msgs,
            "llm_context": None,
        }

async def save_chat_state(session_id: Optional[str], llm_context: Optional[list[int]]) -> None:
    """
    Persist the latest Ollama context back to Django for this session.
    Adjust URL/shape to your API.
    """
    if not session_id or not CHAT_BACKEND_URL:
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
