import base64
import json
from typing import Literal, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.document_export_service import generate_document
from app.services.image_service import create_image_stub, delete_image_for_user
from app.services.pdf_service import create_pdf_stub, delete_doc_for_user

router = APIRouter()


class DocumentExportIn(BaseModel):
    title: str = Field(default="Farm Assistant response", max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    format: Literal["pdf", "docx", "csv", "xlsx", "pptx"]


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


@router.post("/chatbot/api/files/export", tags=["Files"], summary="Export an assistant response")
@router.post("/files/export", include_in_schema=False)
async def export_document(request: Request, body: DocumentExportIn):
    owner_id = _extract_user_uuid_from_token(request.headers.get("Authorization", ""))
    if not owner_id:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        document = generate_document(
            title=body.title.strip() or "Farm Assistant response",
            content=body.content,
            export_format=body.format,
        )
    except ImportError as error:
        raise HTTPException(status_code=503, detail=f"{body.format.upper()} export is not installed.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to generate {body.format.upper()} document.") from error

    return Response(
        content=document.payload,
        media_type=document.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{document.filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

@router.post("/chatbot/api/files/pdf", tags=["Files"], summary="Upload a PDF for chat context")
@router.post("/files/pdf", include_in_schema=False)
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


@router.delete("/chatbot/api/files/pdf/{doc_id}", tags=["Files"], summary="Delete an uploaded PDF")
@router.delete("/files/pdf/{doc_id}", include_in_schema=False)
async def delete_pdf(request: Request, doc_id: str):
    auth_token = request.headers.get("Authorization", "")
    owner_id = _extract_user_uuid_from_token(auth_token) if auth_token else None
    owner_id = owner_id or "anonymous"

    ok = delete_doc_for_user(doc_id=doc_id, owner_id=owner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="PDF not found.")
    return {"status": "success", "doc_id": doc_id}


@router.post("/chatbot/api/files/image", tags=["Files"], summary="Upload an image for chat context")
@router.post("/files/image", include_in_schema=False)
async def upload_image(request: Request, file: UploadFile = File(...)):
    auth_token = request.headers.get("Authorization", "")
    owner_id = _extract_user_uuid_from_token(auth_token) if auth_token else None
    owner_id = owner_id or "anonymous"

    filename = file.filename or ""
    lowered = filename.lower()
    allowed_exts = (".jpg", ".jpeg", ".png")
    allowed_mime = {"image/jpeg", "image/png"}

    if not filename or not lowered.endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are supported.")
    if file.content_type not in allowed_mime:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(payload) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB).")

    doc = create_image_stub(owner_id=owner_id, filename=filename, payload=payload, mime_type=file.content_type)
    return {
        "status": "success",
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "mime_type": doc.mime_type,
        "summary": "",
        "processing": "deferred",
        "created_at": doc.created_at,
    }


@router.delete("/chatbot/api/files/image/{doc_id}", tags=["Files"], summary="Delete an uploaded image")
@router.delete("/files/image/{doc_id}", include_in_schema=False)
async def delete_image(request: Request, doc_id: str):
    auth_token = request.headers.get("Authorization", "")
    owner_id = _extract_user_uuid_from_token(auth_token) if auth_token else None
    owner_id = owner_id or "anonymous"

    ok = delete_image_for_user(doc_id=doc_id, owner_id=owner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"status": "success", "doc_id": doc_id}
