# main.py

import logging
import os
import re

import httpx

from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
)
logger = logging.getLogger("farm-assistant")

app = FastAPI(title="Farm Assistant RAG", version="0.1.0")

# ---- OpenSearch API config (proxy endpoint with BASIC AUTH) ----
OS_API_URL = os.getenv("OPENSEARCH_API_URL", "").rstrip("/")
OS_API_USR = os.getenv("OPENSEARCH_API_USR", "").strip() or None
OS_API_PWD = os.getenv("OPENSEARCH_API_PWD", "").strip() or None
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
OS_API_PATH = "/neural_search_relevant"

# ---- Search defaults (overridable per request) ----
SEARCH_MODEL = os.getenv("SEARCH_MODEL", "msmarco")
SEARCH_INCLUDE_FULLTEXT = os.getenv("SEARCH_INCLUDE_FULLTEXT", "true").lower() == "true"
SEARCH_SORT_BY = os.getenv("SEARCH_SORT_BY", "score_desc")
SEARCH_PAGE = int(os.getenv("SEARCH_PAGE", "1"))
SEARCH_K = int(os.getenv("SEARCH_K", "0"))
SEARCH_DEV = os.getenv("SEARCH_DEV", "false").lower() == "true"

# ---- LLM (Ollama / DeepSeek) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:7b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
TOP_K = int(os.getenv("TOP_K", "4"))  # how many contexts we pass to LLM
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))

class AskIn(BaseModel):
    question: str
    # Optional overrides per request (all are optional)
    page: Optional[int] = None
    k: Optional[int] = None
    model: Optional[str] = None
    include_fulltext: Optional[bool] = None
    sort_by: Optional[str] = None
    dev: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None


class SourceItem(BaseModel):
    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None

class AskOut(BaseModel):
    answer: str
    used_context: List[str]
    sources: List[SourceItem] = []
    meta: Dict[str, Any] = {}

# class AskOut(BaseModel):
#     answer: str
#     used_context: List[str]
#     meta: Dict[str, Any] = {}

def _os_api_endpoint() -> str:
    # join safely without double slashes
    return f"{OS_API_URL}{OS_API_PATH}"

logger.info(f"OS_API_URL={OS_API_URL} path={OS_API_PATH} user_set={bool(OS_API_USR)} ssl_verify={VERIFY_SSL}")

def _build_search_payload(inp: AskIn) -> Dict[str, Any]:
    page = 1 if (inp.page is None or inp.page < 1) else inp.page
    default_k = 5

    payload: Dict[str, Any] = {
        "search_term": inp.question,
        "page": page,
        "dev": False,
        "model": "msmarco",
        "include_fulltext": True,  # <— changed
        "sort_by": "score_desc",
    }

    k_val = inp.k if inp.k is not None else default_k
    if isinstance(k_val, int) and k_val > 0:
        payload["k"] = k_val

    return payload

def _split_paragraphs(text: str) -> list[str]:
    # split on blank lines or sentence-ish boundaries; keep it simple and fast
    parts = re.split(r'\n{2,}|(?<=[\.\?\!])\s+\n?', text)
    # normalise whitespace and drop tiny fragments
    clean = [re.sub(r'\s+', ' ', p).strip() for p in parts]
    return [p for p in clean if len(p) > 40]

def _rank_paragraphs(paragraphs: list[str], question: str) -> list[tuple[int, str]]:
    """
    Extremely light extractive scoring:
      - token overlap with question keywords
      - slight position boost for earlier paragraphs
    Returns list of (score, paragraph) sorted desc.
    """
    q_tokens = {t for t in re.findall(r"[a-zA-Z]+", question.lower()) if len(t) > 2}
    ranked: list[tuple[int, str]] = []
    for idx, p in enumerate(paragraphs):
        p_tokens = {t for t in re.findall(r"[a-zA-Z]+", p.lower()) if len(t) > 2}
        overlap = len(q_tokens & p_tokens)
        score = overlap * 10 + max(0, 5 - idx)  # front-loading slight bias
        ranked.append((score, p))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked

async def _fetch_os_page(client: httpx.AsyncClient, base_payload: Dict[str, Any], page_no: int,
                         headers: Dict[str, str], auth: Optional[httpx.Auth]) -> Dict[str, Any]:
    """Fetch a single page from the OpenSearch proxy."""
    payload = dict(base_payload)
    payload["page"] = page_no  # force requested page
    res = await client.post(_os_api_endpoint(), json=payload, headers=headers, auth=auth)
    res.raise_for_status()
    return res.json()

async def _collect_os_items(client: httpx.AsyncClient, base_payload: Dict[str, Any], pages: List[int],
                            headers: Dict[str, str], auth: Optional[httpx.Auth]) -> List[Dict[str, Any]]:
    """
    Fetch multiple pages (1..N), merge items from `data` (or common fallbacks), and de-duplicate by `_id`.
    """
    all_items: List[Dict[str, Any]] = []
    seen_ids = set()

    for p in pages:
        os_json = await _fetch_os_page(client, base_payload, p, headers, auth)
        items = (
            os_json.get("data")
            or os_json.get("results")
            or os_json.get("hits")
            or os_json.get("items")
            or []
        )
        logger.debug(f"Fetched page={p}, items={len(items)}")
        for it in items:
            _id = (it.get("_id") if isinstance(it, dict) else None) or id(it)
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            all_items.append(it)

    logger.info(f"Collected total_items={len(all_items)} from pages={pages}")
    return all_items


def _build_context_and_sources(items: List[Dict[str, Any]],
                               question: str,
                               top_k: int,
                               max_context_chars: int) -> tuple[list[str], list[SourceItem]]:
    """
    From raw items, create:
      - contexts: short ranked snippets to feed to the LLM (titles + 2–3 best paragraphs)
      - sources : clean metadata for the API response (title, url, _id, _score)
    """
    contexts: List[str] = []
    sources: List[SourceItem] = []
    total_chars = 0

    def norm(v):
        if isinstance(v, list): return " ".join(map(str, v))
        return "" if v is None else str(v)

    for i, it in enumerate(items):
        if top_k > 0 and len(contexts) >= top_k:
            break

        src = it.get("_source", {}) if isinstance(it, dict) and "_source" in it else it

        _id   = it.get("_id") if isinstance(it, dict) else None
        _score = it.get("_score") if isinstance(it, dict) else None
        title = (src.get("title") or "").strip()
        # desc  = norm(src.get("description")).strip()
        url   = src.get("@id")

        # -- collect a clean source record for UI
        sources.append(SourceItem(id=_id, url=url, title=title or None, score=_score))

        # -- build the LLM snippet (title + short desc + selected paragraphs)
        header_parts = []
        if title: header_parts.append(f"Title: {title}")
        # if desc:  header_parts.append(f"Description: {desc[:800]}")
        header = "\n".join(header_parts).strip()

        # gather ko_content_flat
        flat_list = src.get("ko_content_flat")
        flat_text = ""
        if isinstance(flat_list, list): flat_text = " ".join(map(str, flat_list))
        elif isinstance(flat_list, str): flat_text = flat_list

        chosen_paras: list[str] = []
        if flat_text:
            paras = _split_paragraphs(flat_text)
            ranked = _rank_paragraphs(paras, question=question or title)
            for _, p in ranked[:3]:
                if len(p) < 120:
                    continue
                chosen_paras.append(p[:800])
                if sum(len(x) for x in chosen_paras) > 1200:
                    break

        parts: list[str] = []
        if header: parts.append(header)
        if chosen_paras:
            parts.append("Content:\n- " + "\n- ".join(chosen_paras))
        chunk = "\n".join(parts).strip() or (f"Title: {title}" if title else "")

        if chunk:
            chunk = chunk[:2000]
            if total_chars + len(chunk) > max_context_chars:
                break
            contexts.append(chunk)
            total_chars += len(chunk)

    logger.info(f"Extracted {len(contexts)} context chunk(s); total_chars={total_chars}")
    return contexts, sources


def _extract_contexts(os_json: Dict[str, Any], top_k: int) -> List[str]:
    contexts: List[str] = []
    total_chars = 0
    items = (
        os_json.get("data")
        or os_json.get("results")
        or os_json.get("hits")
        or os_json.get("items")
        or []
    )

    logger.debug(f"items_count={len(items)}; top_k={top_k}; max_context_chars={MAX_CONTEXT_CHARS}")

    def norm(v):
        if isinstance(v, list): return " ".join(map(str, v))
        return "" if v is None else str(v)

    for i, it in enumerate(items):
        if top_k > 0 and len(contexts) >= top_k:
            break

        src = it.get("_source", {}) if isinstance(it, dict) and "_source" in it else it

        title = (src.get("title") or "").strip()
        desc  = norm(src.get("description")).strip()

        # Build a header for this item
        header_parts = []
        if title: header_parts.append(f"Title: {title}")
        if desc:  header_parts.append(f"Description: {desc[:800]}")  # brief abstract
        header = "\n".join(header_parts).strip()

        # Collect fulltext paragraphs from ko_content_flat (list[str]) if present
        flat_list = src.get("ko_content_flat")
        flat_text = ""
        if isinstance(flat_list, list):
            flat_text = " ".join(map(str, flat_list))
        elif isinstance(flat_list, str):
            flat_text = flat_list

        # Rank paragraphs against the user's question to pick the best few
        chosen_paras: list[str] = []
        if flat_text:
            paras = _split_paragraphs(flat_text)
            ranked = _rank_paragraphs(paras, question=title or desc or "")
            # Take top 2–3 concise paragraphs (~500–700 chars combined)
            for _, p in ranked[:3]:
                if len(p) < 120:  # discard super short lines
                    continue
                chosen_paras.append(p[:800])
                if sum(len(x) for x in chosen_paras) > 1200:
                    break

        # Assemble chunk
        parts: list[str] = []
        if header: parts.append(header)
        if chosen_paras:
            parts.append("Content: " + "\n".join(chosen_paras))
        chunk = "\n".join(parts).strip()

        if not chunk:
            # As a last resort, if we have only a title, keep it—better than nothing
            if title:
                chunk = f"Title: {title}"

        if chunk:
            # Respect per-item and global budgets
            chunk = chunk[:2000]
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                logger.debug(f"Context budget reached at item {i}")
                break
            contexts.append(chunk)
            total_chars += len(chunk)
            logger.debug(f"Context[{i}] added; len={len(chunk)}; total_chars={total_chars}")

    logger.info(f"Extracted {len(contexts)} context chunk(s); total_chars={total_chars}")
    return contexts



def _build_prompt(contexts: List[str], question: str) -> str:
    """Prompt tuned for natural, ChatGPT-style language."""
    joined_contexts = "\n\n".join(contexts)
    return (
        "System: You are a knowledgeable agricultural assistant. "
        "Read the provided context carefully and craft a clear, natural-sounding answer. "
        "Use complete sentences and provide brief reasoning if helpful. "
        "Do NOT invent facts beyond the context. "
        "If the context lacks enough information, politely say you don't know.\n\n"
        f"Context:\n{joined_contexts}\n\n"
        f"User question: {question}\n\n"
        "Assistant:"
    )


@app.post("/ask", response_model=AskOut)
async def ask(inp: AskIn) -> AskOut:
    if not inp.question or not inp.question.strip():
        return AskOut(answer="Please provide a question.", used_context=[], meta={})

    headers = {"accept": "application/json", "Content-Type": "application/json"}
    auth = httpx.BasicAuth(OS_API_USR, OS_API_PWD) if (OS_API_USR and OS_API_PWD) else None

    # 1) Query OpenSearch API with BASIC AUTH (if provided) and SSL verify control
    search_payload = _build_search_payload(inp)

    # Build the list of pages to fetch
    last_page = inp.page if (inp.page is not None and inp.page >= 1) else 1
    pages = list(range(1, last_page + 1))  # e.g., page=3 -> [1,2,3]; default -> [1]

    async with httpx.AsyncClient(timeout=30.0, verify=VERIFY_SSL) as client:
        try:
            items = await _collect_os_items(client, search_payload, pages, headers, auth)
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:400]
            logger.error(f"OpenSearch error {e.response.status_code}: {body}")
            return AskOut(
                answer="Search upstream error.",
                used_context=[],
                meta={"search": search_payload, "upstream_status": e.response.status_code,
                      "upstream_body_snippet": body}
            )

    # Wrap merged items in the shape _extract_contexts expects
    os_json = {"data": items}

    # 2) Extract contexts (top_k controls how many chunks we stuff)
    top_k = inp.top_k if inp.top_k is not None else TOP_K
    contexts, sources = _build_context_and_sources(
        items=items,
        question=inp.question,
        top_k=top_k,
        max_context_chars=MAX_CONTEXT_CHARS
    )
    if not contexts:
        return AskOut(
            answer="I don't know based on the current context.",
            used_context=[],
            sources=sources,
            meta={"search": search_payload},
        )

    # 3) Ask the LLM (Ollama → DeepSeek). IMPORTANT: stream=false for r.json()
    max_tokens = inp.max_tokens if inp.max_tokens is not None else MAX_TOKENS
    temperature = inp.temperature if inp.temperature is not None else TEMPERATURE
    prompt = _build_prompt(contexts, inp.question)

    gen_payload = {
      "model": LLM_MODEL,
      "prompt": prompt,
      "options": {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": NUM_CTX
      },
      "stream": False  # ensure non-stream response for r.json()
    }

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=gen_payload)
        r.raise_for_status()
        data = r.json()

    answer = (data.get("response") or "").strip()

    return AskOut(answer=answer, used_context=contexts, sources=sources, meta={"search": search_payload})

