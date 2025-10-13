# app/services/context_service.py

import logging, re

from typing import Dict, Any, List

from app.schemas import SourceItem
from app.config import get_settings

logger = logging.getLogger("farm-assistant.context")
S = get_settings()

def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r'\n{2,}|(?<=[\.\?\!])\s+\n?', text)
    clean = [re.sub(r'\s+', ' ', p).strip() for p in parts]
    return [p for p in clean if len(p) > 40]

def rank_paragraphs(
    paragraphs: list[str],
    question: str,
    boost_terms: set[str] | None = None
) -> list[tuple[int, str]]:
    q_tokens = {t for t in re.findall(r"[a-zA-Z]+", question.lower()) if len(t) > 2}
    bt = boost_terms or set()  # keep constant per call

    ranked: list[tuple[int, str]] = []
    for idx, p in enumerate(paragraphs):
        p_tokens = {t for t in re.findall(r"[a-zA-Z]+", p.lower()) if len(t) > 2}
        overlap = len(q_tokens & p_tokens)
        boost_overlap = len(p_tokens & bt)
        score = overlap * 10 + boost_overlap * 4 + max(0, 5 - idx)  # slight front-load
        ranked.append((score, p))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked

def build_context_and_sources(
    items: List[Dict[str, Any]],
    question: str,
    top_k: int,
    max_context_chars: int
) -> tuple[list[str], list[SourceItem]]:
    contexts: List[str] = []
    sources: List[SourceItem] = []
    total_chars = 0

    def norm(v):
        if isinstance(v, list): return " ".join(map(str, v))
        return "" if v is None else str(v)

    for i, it in enumerate(items):
        sid = f"S{i + 1}"
        if top_k > 0 and len(contexts) >= top_k:
            break

        src = it.get("_source", {}) if isinstance(it, dict) and "_source" in it else it
        _id    = it.get("_id") if isinstance(it, dict) else None
        _score = it.get("_score") if isinstance(it, dict) else None
        title  = (src.get("title") or "").strip()
        url    = src.get("@id")
        subtitle = (src.get("subtitle") or "").strip()
        desc = (src.get("description") or "").strip()
        proj = (src.get("project_display_name") or "")
        acronym = (src.get("project_acronym") or "")
        ptype = (src.get("project_type") or "")
        license_ = (src.get("license") or "")
        keywords = src.get("keywords") or []
        topics = src.get("topics") or []
        themes = src.get("themes") or []
        langs = src.get("languages") or []
        creators = src.get("creators") or []
        datec = (src.get("date_of_completion") or "")
        nice_url = (src.get("project_url") or url or None)

        # Build a compact provenance string, e.g. "NETPOULSAFE (Horizon 2020)"
        proj_str = ""
        if proj or acronym:
            cap = f"{proj}".strip() or f"{acronym}".strip()
            if ptype:
                proj_str = f"{cap} ({ptype})"
            else:
                proj_str = cap

        sources.append(SourceItem(
            id=_id, url=url, display_url=nice_url,
            title=title or None, score=_score,
            subtitle=subtitle or None,
            description=(desc[:300] if desc else None),
            project=(proj_str or None),
            license=(license_ or None),
            keywords=(keywords or None) if keywords else None,
            topics=(topics or None) if topics else None,
            themes=(themes or None) if themes else None,
            languages=(langs or None) if langs else None,
            creators=(creators or None) if creators else None,
            date_of_completion=(datec or None),
            sid=sid,
        ))

        header_parts = [f"[{sid}]"]
        if title: header_parts.append(f"Title: {title}")
        if subtitle:
            header_parts.append(f"Subtitle: {subtitle}")
        if desc:
            header_parts.append(f"Description: {desc[:800]}")
        if proj_str or license_ or datec:
            meta_bits = []
            if proj_str: meta_bits.append(proj_str)
            if datec:    meta_bits.append(f"Completed: {datec}")
            if license_: meta_bits.append(f"License: {license_}")
            header_parts.append(" · ".join(meta_bits))
        if keywords:
            header_parts.append("Keywords: " + ", ".join(keywords[:8]))
        if topics:
            header_parts.append("Topics: " + ", ".join(topics[:6]))
        header = "\n".join(header_parts).strip()

        flat_list = src.get("ko_content_flat")
        flat_text = ""
        if isinstance(flat_list, list):
            flat_text = " ".join(map(str, flat_list))
        elif isinstance(flat_list, str):
            flat_text = flat_list

        chosen_paras: list[str] = []
        if flat_text:
            paras = split_paragraphs(flat_text)
            boost_terms = set(t.lower() for t in (keywords or [])) | set(t.lower() for t in (topics or []))
            ranked = rank_paragraphs(paras, question=question or title, boost_terms=boost_terms)
            for _, p in ranked[:3]:
                if len(p) < 120: continue
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
