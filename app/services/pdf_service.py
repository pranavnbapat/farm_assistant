import asyncio
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import httpx

from app.clients.vllm_client import generate_once

logger = logging.getLogger("farm-assistant.pdf")

UPLOAD_DIR = Path("/tmp/farm_assistant_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PdfDocument:
    doc_id: str
    owner_id: str
    filename: str
    path: Path | None
    created_at: str
    extracted_text: str = ""
    summary: str = ""
    processed: bool = False
    processing_error: str = ""


_DOC_STORE: Dict[str, PdfDocument] = {}
_DOC_LOCKS: Dict[str, asyncio.Lock] = {}


def _tokenise(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z]{3,}", text or "")}


def _overlap_score(query: str, text: str) -> int:
    q = _tokenise(query)
    if not q:
        return 0
    t = _tokenise(text)
    return len(q & t)


def _split_chunks(text: str, chunk_size: int = 1600) -> List[str]:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return []
    out: List[str] = []
    i = 0
    while i < len(t):
        out.append(t[i:i + chunk_size])
        i += chunk_size
    return out


def _extract_text_with_pypdf(path: Path, max_pages: int = 80, max_chars: int = 240000) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError("pypdf is required for PDF extraction. Install dependency first.") from e

    reader = PdfReader(str(path))
    pages = reader.pages[:max_pages]
    parts: List[str] = []
    total = 0
    for p in pages:
        try:
            txt = (p.extract_text() or "").strip()
        except Exception:
            txt = ""
        if not txt:
            continue
        remain = max_chars - total
        if remain <= 0:
            break
        txt = txt[:remain]
        parts.append(txt)
        total += len(txt)
    return "\n\n".join(parts).strip()


async def _summarise_text(text: str, filename: str) -> str:
    excerpt = (text or "")[:24000]
    if not excerpt:
        return "No extractable text was found in this PDF."

    prompt = (
        "Summarize this PDF in 6-10 concise bullet points for an agricultural assistant. "
        "Focus on key facts, practical takeaways, and any location/crop/livestock specifics. "
        "If the content is not agriculture-related, still summarize faithfully.\n\n"
        f"Filename: {filename}\n\n"
        f"PDF content excerpt:\n{excerpt}\n\n"
        "Summary:"
    )
    try:
        res = await generate_once(prompt, temperature=0.2, max_tokens=320)
        cleaned = (res or "").strip()
        if cleaned:
            return cleaned
    except Exception as e:
        logger.warning(f"PDF summary generation failed for {filename}: {e}")
    return excerpt[:1200]


def create_pdf_stub(owner_id: str, filename: str, payload: bytes) -> PdfDocument:
    digest = hashlib.sha256(payload).hexdigest()[:16]
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{digest}_{Path(filename).name}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(payload)

    doc = PdfDocument(
        doc_id=doc_id,
        owner_id=owner_id,
        filename=filename,
        path=out_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _DOC_STORE[doc_id] = doc
    _DOC_LOCKS[doc_id] = asyncio.Lock()
    return doc


async def ensure_pdf_processed(doc: PdfDocument) -> PdfDocument:
    if doc.processed:
        return doc

    lock = _DOC_LOCKS.setdefault(doc.doc_id, asyncio.Lock())
    async with lock:
        if doc.processed:
            return doc
        if doc.processing_error:
            return doc
        try:
            if not doc.path:
                doc.processing_error = "PDF binary is no longer available."
                return doc
            doc.extracted_text = _extract_text_with_pypdf(doc.path)
            doc.summary = await _summarise_text(doc.extracted_text, doc.filename)
            doc.processed = True
            try:
                if doc.path.exists():
                    doc.path.unlink()
            except Exception:
                pass
            doc.path = None
        except Exception as e:
            doc.processing_error = str(e)
            logger.warning(f"Failed to process PDF {doc.doc_id}: {e}")
            try:
                if doc.path and doc.path.exists():
                    doc.path.unlink()
            except Exception:
                pass
            doc.path = None
    return doc


def get_docs_for_user(doc_ids: List[str], owner_id: str) -> List[PdfDocument]:
    docs: List[PdfDocument] = []
    for did in doc_ids:
        d = _DOC_STORE.get(did)
        if not d:
            continue
        if d.owner_id != owner_id:
            continue
        docs.append(d)
    return docs


def delete_doc_for_user(doc_id: str, owner_id: str) -> bool:
    d = _DOC_STORE.get(doc_id)
    if not d or d.owner_id != owner_id:
        return False
    try:
        if d.path and d.path.exists():
            d.path.unlink()
    except Exception:
        pass
    _DOC_STORE.pop(doc_id, None)
    _DOC_LOCKS.pop(doc_id, None)
    return True


def build_pdf_contexts(
    docs: List[PdfDocument],
    question: str,
    max_total_chars: int = 10000,
    max_chunks_per_doc: int = 3,
) -> tuple[List[str], List[dict]]:
    contexts: List[str] = []
    sources: List[dict] = []
    used = 0

    for doc in docs:
        sources.append({
            "id": doc.doc_id,
            "title": doc.filename,
            "url": None,
            "license": None,
        })

        text_for_chunks = doc.extracted_text or ""
        chunks = _split_chunks(text_for_chunks)
        if not chunks:
            chunk_texts = [doc.summary or "No extractable text was found in this PDF."]
        else:
            ranked = sorted(chunks, key=lambda c: _overlap_score(question, c), reverse=True)
            chunk_texts = ranked[:max_chunks_per_doc]

        body = "\n\n".join([c[:1200] for c in chunk_texts if c.strip()])
        header = f"PDF File: {doc.filename}\nSummary:\n{(doc.summary or '')[:1200]}"
        block = f"{header}\n\nRelevant Extract:\n{body}".strip()
        if not block:
            continue

        if used + len(block) > max_total_chars:
            remain = max_total_chars - used
            if remain <= 0:
                break
            block = block[:remain]
        contexts.append(block)
        used += len(block)

        if used >= max_total_chars:
            break

    return contexts, sources


def docs_from_attachment_records(records: List[dict], owner_id: str = "persisted") -> List[PdfDocument]:
    out: List[PdfDocument] = []
    for r in records or []:
        doc_id = str(r.get("attachment_uuid") or r.get("id") or uuid.uuid4())
        out.append(PdfDocument(
            doc_id=doc_id,
            owner_id=owner_id,
            filename=(r.get("filename") or "document.pdf"),
            path=None,
            created_at=str(r.get("created_at") or datetime.now(timezone.utc).isoformat()),
            extracted_text=(r.get("extracted_text") or ""),
            summary=(r.get("summary") or ""),
            processed=True,
        ))
    return out


async def upsert_attachment_to_backend(
    *,
    chat_backend_url: str,
    verify_ssl: bool,
    auth_token: str,
    session_uuid: str,
    doc: PdfDocument,
    message_id: int | None = None,
) -> None:
    if not chat_backend_url or not auth_token or not session_uuid or not doc:
        return

    url = f"{chat_backend_url.rstrip('/')}/chat/attachments/upsert/"
    payload = {
        "attachment_uuid": doc.doc_id,
        "session_uuid": session_uuid,
        "message_id": message_id,
        "attachment_type": "pdf",
        "filename": doc.filename,
        "mime_type": "application/pdf",
        "process_status": "processed" if doc.processed and not doc.processing_error else "failed",
        "process_error": doc.processing_error or "",
        "summary": doc.summary or "",
        "extracted_text": (doc.extracted_text or "")[:20000],
        "extracted_meta": {
            "extracted_chars": len(doc.extracted_text or ""),
            "summary_chars": len(doc.summary or ""),
        },
        "extra": {"source": "fastapi_pdf_service"},
    }
    headers = {"Authorization": auth_token}
    timeout = httpx.Timeout(connect=5.0, read=8.0, write=8.0, pool=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.is_error:
                logger.warning(f"Attachment upsert failed HTTP {r.status_code}: {(r.text or '')[:200]}")
    except Exception as e:
        logger.warning(f"Attachment upsert request failed: {e}")


async def fetch_session_attachments_from_backend(
    *,
    chat_backend_url: str,
    verify_ssl: bool,
    auth_token: str,
    session_uuid: str,
) -> List[dict]:
    if not chat_backend_url or not auth_token or not session_uuid:
        return []

    url = f"{chat_backend_url.rstrip('/')}/chat/attachments/"
    headers = {"Authorization": auth_token}
    timeout = httpx.Timeout(connect=5.0, read=8.0, write=8.0, pool=5.0)
    params = {"session_uuid": session_uuid}

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            r = await client.get(url, params=params, headers=headers)
            if r.is_error:
                return []
            data = r.json() or {}
            results = data.get("results") or []
            if isinstance(results, list):
                return results
    except Exception:
        return []
    return []
