import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx

from app.clients.vllm_client import generate_vision_once

logger = logging.getLogger("farm-assistant.image")

UPLOAD_DIR = Path("/tmp/farm_assistant_image_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageDocument:
    doc_id: str
    owner_id: str
    filename: str
    mime_type: str
    path: Path | None
    created_at: str
    summary: str = ""
    processed: bool = False
    processing_error: str = ""
    agriculture_related: bool = False
    observed_subjects: List[str] = field(default_factory=list)
    potential_agri_topics: List[str] = field(default_factory=list)


_IMAGE_STORE: Dict[str, ImageDocument] = {}
_IMAGE_LOCKS: Dict[str, asyncio.Lock] = {}


def create_image_stub(owner_id: str, filename: str, payload: bytes, mime_type: str) -> ImageDocument:
    digest = hashlib.sha256(payload).hexdigest()[:16]
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{digest}_{Path(filename).name}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(payload)

    doc = ImageDocument(
        doc_id=doc_id,
        owner_id=owner_id,
        filename=filename,
        mime_type=mime_type,
        path=out_path,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _IMAGE_STORE[doc_id] = doc
    _IMAGE_LOCKS[doc_id] = asyncio.Lock()
    return doc


def get_images_for_user(doc_ids: List[str], owner_id: str) -> List[ImageDocument]:
    docs: List[ImageDocument] = []
    for did in doc_ids:
        d = _IMAGE_STORE.get(did)
        if not d:
            continue
        if d.owner_id != owner_id:
            continue
        docs.append(d)
    return docs


def delete_image_for_user(doc_id: str, owner_id: str) -> bool:
    d = _IMAGE_STORE.get(doc_id)
    if not d or d.owner_id != owner_id:
        return False
    try:
        if d.path and d.path.exists():
            d.path.unlink()
    except Exception:
        pass
    _IMAGE_STORE.pop(doc_id, None)
    _IMAGE_LOCKS.pop(doc_id, None)
    return True


def _vision_prompt(filename: str) -> str:
    return (
        "Analyze this uploaded image for EU-FarmBook Farm Assistant.\n"
        "Return valid JSON only with this exact shape:\n"
        "{\n"
        '  "agriculture_related": true,\n'
        '  "summary": "2-4 sentence factual summary of what is visibly present",\n'
        '  "observed_subjects": ["crop leaves", "tractor"],\n'
        '  "potential_agri_topics": ["plant health", "machinery"]\n'
        "}\n\n"
        "Rules:\n"
        "- Set agriculture_related to true only if the image is clearly about agriculture, farming, crops, livestock, soil, agricultural machinery, farm infrastructure, or food production.\n"
        "- If the image is not agriculture-related, set agriculture_related to false and say that plainly in the summary.\n"
        "- Be factual. Describe visible symptoms, crops, animals, machinery, fields, greenhouses, soil, or packaging only if they are actually visible.\n"
        "- Do not guess a disease or diagnosis unless it is visually obvious. Prefer cautious wording such as 'visible leaf damage' or 'possible discoloration'.\n"
        "- Keep observed_subjects and potential_agri_topics short and concrete.\n\n"
        f"Filename: {filename}"
    )


def _coerce_analysis(raw: str) -> dict[str, Any]:
    try:
        match_start = raw.find("{")
        match_end = raw.rfind("}")
        if match_start == -1 or match_end == -1 or match_end <= match_start:
            raise ValueError("No JSON object found")
        data = json.loads(raw[match_start:match_end + 1])
        if not isinstance(data, dict):
            raise ValueError("Vision response is not an object")
        return data
    except Exception as exc:
        raise RuntimeError(f"Unable to parse vision output: {exc}") from exc


async def ensure_image_processed(doc: ImageDocument) -> ImageDocument:
    if doc.processed:
        return doc

    lock = _IMAGE_LOCKS.setdefault(doc.doc_id, asyncio.Lock())
    async with lock:
        if doc.processed:
            return doc
        if doc.processing_error:
            return doc
        try:
            if not doc.path:
                doc.processing_error = "Image binary is no longer available."
                return doc
            image_bytes = doc.path.read_bytes()
            raw = await generate_vision_once(
                prompt=_vision_prompt(doc.filename),
                image_bytes=image_bytes,
                mime_type=doc.mime_type,
                temperature=0.1,
                max_tokens=420,
            )
            analysis = _coerce_analysis(raw)

            doc.agriculture_related = bool(analysis.get("agriculture_related"))
            doc.summary = str(analysis.get("summary") or "").strip()
            doc.observed_subjects = [
                str(x).strip()
                for x in (analysis.get("observed_subjects") or [])
                if str(x).strip()
            ][:8]
            doc.potential_agri_topics = [
                str(x).strip()
                for x in (analysis.get("potential_agri_topics") or [])
                if str(x).strip()
            ][:8]
            if not doc.summary:
                doc.summary = "No reliable visual summary could be produced for this image."
            doc.processed = True
            try:
                if doc.path.exists():
                    doc.path.unlink()
            except Exception:
                pass
            doc.path = None
        except Exception as e:
            doc.processing_error = str(e)
            logger.warning(f"Failed to process image {doc.doc_id}: {e}")
            try:
                if doc.path and doc.path.exists():
                    doc.path.unlink()
            except Exception:
                pass
            doc.path = None
    return doc


def build_image_contexts(
    docs: List[ImageDocument],
    max_total_chars: int = 8000,
) -> tuple[List[str], List[dict], Dict[str, int]]:
    contexts: List[str] = []
    sources: List[dict] = []
    used = 0
    stats = {"total": len(docs), "agri_related": 0, "non_agri": 0}

    for doc in docs:
        if doc.agriculture_related:
            stats["agri_related"] += 1
        else:
            stats["non_agri"] += 1

        sources.append({
            "id": doc.doc_id,
            "title": doc.filename,
            "url": None,
            "license": None,
        })

        parts = [f"Uploaded image: {doc.filename}"]
        if doc.processing_error:
            # Surface the processing failure so the LLM doesn't have to guess
            # why it has no real summary to work with.
            parts.append(f"Visual analysis failed: {doc.processing_error[:300]}")
        else:
            parts.append(
                "Agriculture-related: "
                f"{'yes' if doc.agriculture_related else 'no'}"
            )
            if doc.summary:
                parts.append(f"Visual summary: {doc.summary[:1200]}")
            if doc.observed_subjects:
                parts.append("Observed subjects: " + ", ".join(doc.observed_subjects[:8]))
            if doc.agriculture_related and doc.potential_agri_topics:
                parts.append("Potential agricultural topics: " + ", ".join(doc.potential_agri_topics[:8]))

        block = "\n".join(parts).strip()
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

    return contexts, sources, stats


def images_from_attachment_records(records: List[dict], owner_id: str = "persisted") -> List[ImageDocument]:
    out: List[ImageDocument] = []
    for r in records or []:
        attachment_type = str(r.get("attachment_type") or "").lower()
        if attachment_type not in {"image", "image_upload"}:
            continue
        extracted_meta = r.get("extracted_meta") or {}
        out.append(ImageDocument(
            doc_id=str(r.get("attachment_uuid") or r.get("id") or uuid.uuid4()),
            owner_id=owner_id,
            filename=(r.get("filename") or "image"),
            mime_type=(r.get("mime_type") or "image/jpeg"),
            path=None,
            created_at=str(r.get("created_at") or datetime.now(timezone.utc).isoformat()),
            summary=(r.get("summary") or ""),
            processed=True,
            processing_error=(r.get("process_error") or ""),
            agriculture_related=bool(extracted_meta.get("agriculture_related")),
            observed_subjects=[
                str(x).strip()
                for x in (extracted_meta.get("observed_subjects") or [])
                if str(x).strip()
            ][:8],
            potential_agri_topics=[
                str(x).strip()
                for x in (extracted_meta.get("potential_agri_topics") or [])
                if str(x).strip()
            ][:8],
        ))
    return out


async def upsert_image_attachment_to_backend(
    *,
    chat_backend_url: str,
    verify_ssl: bool,
    auth_token: str,
    session_uuid: str,
    doc: ImageDocument,
    message_id: int | None = None,
) -> None:
    if not chat_backend_url or not auth_token or not session_uuid or not doc:
        return

    url = f"{chat_backend_url.rstrip('/')}/chat/attachments/upsert/"
    payload = {
        "attachment_uuid": doc.doc_id,
        "session_uuid": session_uuid,
        "message_id": message_id,
        "attachment_type": "image",
        "filename": doc.filename,
        "mime_type": doc.mime_type,
        "process_status": "processed" if doc.processed and not doc.processing_error else "failed",
        "process_error": doc.processing_error or "",
        "summary": doc.summary or "",
        "extracted_text": doc.summary or "",
        "extracted_meta": {
            "agriculture_related": doc.agriculture_related,
            "observed_subjects": doc.observed_subjects,
            "potential_agri_topics": doc.potential_agri_topics,
        },
        "extra": {"source": "fastapi_image_service"},
    }
    headers = {"Authorization": auth_token}
    timeout = httpx.Timeout(connect=5.0, read=8.0, write=8.0, pool=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.is_error:
                logger.warning(f"Image attachment upsert failed HTTP {r.status_code}: {(r.text or '')[:200]}")
    except Exception as e:
        logger.warning(f"Image attachment upsert request failed: {e}")
