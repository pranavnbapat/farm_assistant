import base64
import json
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.pdf_service import create_pdf_stub, delete_doc_for_user

router = APIRouter()


def _extract_user_uuid_from_token(auth_token: str) -> Optional[str]:
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


@router.post("/files/pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    auth_token = request.headers.get("Authorization", "")
    owner_id = _extract_user_uuid_from_token(auth_token) if auth_token else None
    owner_id = owner_id or "anonymous"

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large (max 25MB).")

    doc = create_pdf_stub(owner_id=owner_id, filename=file.filename, payload=payload)
    return {
        "status": "success",
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "summary": "",
        "processing": "deferred",
        "created_at": doc.created_at,
    }


@router.delete("/files/pdf/{doc_id}")
async def delete_pdf(request: Request, doc_id: str):
    auth_token = request.headers.get("Authorization", "")
    owner_id = _extract_user_uuid_from_token(auth_token) if auth_token else None
    owner_id = owner_id or "anonymous"

    ok = delete_doc_for_user(doc_id=doc_id, owner_id=owner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="PDF not found.")
    return {"status": "success", "doc_id": doc_id}
