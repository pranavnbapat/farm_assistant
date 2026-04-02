# app/main.py

import asyncio
import base64
import json
import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast, Any, Optional

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import get_settings
from app.routers.ask import router as ask_router
from app.routers.files import router as files_router
from app.schemas import (
    ChatSessionCreateIn,
    ChatSessionPatchIn,
    ChatTurnLogIn,
    LogoutIn,
    UserFactCreateIn,
    UserProfileBuildIn,
    UserProfilePatchIn,
)
from app.services.user_profile_service import UserProfileService

S = get_settings()

logging.basicConfig(
    level=S.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
)
logger = logging.getLogger("farm-assistant")


# LOGIN_ENDPOINTS for different environments
LOGIN_ENDPOINTS = {
    "local": "http://127.0.0.1:8000/fastapi/login/",
    "dev":   "https://backend-admin.dev.farmbook.ugent.be/fastapi/login/",
    "prd":   "https://backend-admin.prd.farmbook.ugent.be/fastapi/login/",
}

# Get auth settings from environment (these are NOT in pydantic settings)
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "")
AUTH_BACKEND_URL = os.getenv("AUTH_BACKEND_URL", "")

# Determine LOGIN_URL based on settings
if AUTH_BACKEND_URL:
    # Explicit override takes precedence
    LOGIN_URL = AUTH_BACKEND_URL.rstrip("/") + "/fastapi/login/"
else:
    # Use FA_ENV from pydantic settings (reads from .env properly)
    LOGIN_URL = LOGIN_ENDPOINTS.get(S.FA_ENV, LOGIN_ENDPOINTS["local"])

logger.info(f"FA_ENV={S.FA_ENV}, Using auth backend: {LOGIN_URL}")

ENABLE_DOCS: bool = True


def _relay_upstream_response(upstream: httpx.Response) -> JSONResponse:
    """
    Return upstream response with original status code and best-effort JSON body.
    Avoid masking upstream 4xx/5xx as generic 502 so frontend can show real errors.
    """
    try:
        body = upstream.json()
    except ValueError:
        body = {
            "status": "error",
            "message": "Upstream returned non-JSON response",
            "upstream_status": upstream.status_code,
            "upstream_body": (upstream.text or "")[:1000],
        }

    if upstream.is_error:
        logger.warning(f"Upstream proxy error: HTTP {upstream.status_code}, body={str(body)[:300]}")

    return JSONResponse(content=body, status_code=upstream.status_code)


def _chat_backend_headers(request: Request) -> dict[str, str]:
    return {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }


async def _proxy_json_request(
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    json_body: Any = None,
    params: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                params=params,
            )
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy {method} {url}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


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
        token_data = json.loads(base64.urlsafe_b64decode(payload))
        user_id = token_data.get("uuid") or token_data.get("user_id") or token_data.get("sub")
        return str(user_id) if user_id else None
    except Exception:
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    limit = int(os.getenv("MAX_ACTIVE_GENERATIONS", "3"))

    state = cast(Any, app).state
    state.gen_semaphore = asyncio.Semaphore(limit)

    yield

app = FastAPI(
    title=S.APP_TITLE,
    version=S.APP_VERSION,
    docs_url="/docs" if S.ENABLE_DOCS else None,
    redoc_url="/redoc" if getattr(S, "ENABLE_REDOC", False) else None,
    openapi_url="/openapi.json" if S.ENABLE_DOCS else None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

LOGIN_HTML = TEMPLATES_DIR / "login.html"
CHAT_HTML = TEMPLATES_DIR / "ask_stream.html"

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Health-check (optional)
@app.get("/health")
async def health():
    return {"status": "ok"}

# Mount feature routers
app.include_router(ask_router)
app.include_router(files_router)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def login_page():
    """Public login page."""
    return LOGIN_HTML.read_text(encoding="utf-8")


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_page(request: Request):
    """Chat UI – frontend will redirect to / if not logged in."""
    return templates.TemplateResponse(
        request=request,
        name="ask_stream.html",
        context={
            "FA_ENV": S.FA_ENV,
        },
    )
    # return CHAT_HTML.read_text(encoding="utf-8")

class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/chatbot/api/auth/login", tags=["Authentication"], summary="Authorize a user")
@app.post("/api/login", include_in_schema=False)
async def api_login(body: LoginBody):
    payload = {
        "email": body.email,
        "password": body.password,
    }

    headers = {"Content-Type": "application/json"}
    if ADMIN_API_TOKEN:
        headers["Authorization"] = f"Bearer {ADMIN_API_TOKEN}"

    logger.info(f"Attempting login for user: {body.email} to {LOGIN_URL}")

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=True) as client:
            upstream = await client.post(LOGIN_URL, json=payload, headers=headers)
    except httpx.ConnectError as exc:
        logger.error(f"Cannot connect to auth backend at {LOGIN_URL}: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Auth backend is unavailable. Please check that the backend server is running at {LOGIN_URL}"
        )
    except httpx.RequestError as exc:
        logger.error(f"Request error to auth backend: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Unable to reach auth backend: {exc}"
        )

    if upstream.status_code != 200:
        # Login failed on the auth backend
        try:
            error_data = upstream.json()
            error_text = error_data.get("message", error_data.get("detail", upstream.text))
        except ValueError:
            error_text = upstream.text
        
        logger.warning(f"Login failed for {body.email}: HTTP {upstream.status_code} - {error_text}")
        raise HTTPException(
            status_code=upstream.status_code,
            detail=error_text or "Authentication failed"
        )

    # Return exactly what the auth backend sent (token etc.)
    try:
        data = upstream.json()
        logger.info(f"Login successful for user: {body.email}")
    except ValueError:
        logger.error("Auth backend returned non-JSON response")
        raise HTTPException(
            status_code=502,
            detail="Auth backend returned non-JSON response"
        )

    return JSONResponse(content=data, status_code=200)


# =========================
# Proxy endpoints for chat functionality (avoids CORS issues)
# =========================

@app.get("/chatbot/api/chats", tags=["Chats"], summary="Read Chats")
@app.get("/proxy/chat/sessions/", include_in_schema=False)
async def proxy_get_sessions(request: Request):
    """Proxy GET sessions request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/"
    return await _proxy_json_request("GET", url, headers=_chat_backend_headers(request))


@app.post("/chatbot/api/chats", tags=["Chats"], summary="Create Chat")
async def api_create_session(body: ChatSessionCreateIn, request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/"
    return await _proxy_json_request("POST", url, headers=_chat_backend_headers(request), json_body=body.model_dump())


@app.post("/proxy/chat/sessions/", include_in_schema=False)
async def proxy_create_session(request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/"
    return await _proxy_json_request("POST", url, headers=_chat_backend_headers(request), json_body=await request.json())


@app.get("/chatbot/api/chats/{session_id}", tags=["Chats"], summary="Get a specific chat")
@app.get("/proxy/chat/sessions/{session_id}/", include_in_schema=False)
async def proxy_get_session(session_id: str, request: Request):
    """Proxy GET single session request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    return await _proxy_json_request("GET", url, headers=_chat_backend_headers(request))


@app.patch("/chatbot/api/chats/{session_id}", tags=["Chats"], summary="Update chat metadata")
async def api_patch_session(session_id: str, body: ChatSessionPatchIn, request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    payload = body.model_dump(exclude_none=True)
    return await _proxy_json_request("PATCH", url, headers=_chat_backend_headers(request), json_body=payload)


@app.patch("/proxy/chat/sessions/{session_id}/", include_in_schema=False)
async def proxy_patch_session(session_id: str, request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    return await _proxy_json_request("PATCH", url, headers=_chat_backend_headers(request), json_body=await request.json())


@app.delete("/chatbot/api/chats/{session_id}", tags=["Chats"], summary="Delete a specific chat")
@app.delete("/proxy/chat/sessions/{session_id}/", include_in_schema=False)
async def proxy_delete_session(session_id: str, request: Request):
    """Proxy DELETE session request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    return await _proxy_json_request("DELETE", url, headers=_chat_backend_headers(request))


@app.post("/chatbot/api/chats/log-turn", tags=["Chats"], summary="Store a completed chat turn")
async def api_log_turn(body: ChatTurnLogIn, request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/log-turn/"
    logger.info(f"Proxying chat log to {url}")
    return await _proxy_json_request("POST", url, headers=_chat_backend_headers(request), json_body=body.model_dump())


@app.post("/chatbot/api/chats/{session_id}/log-turn", tags=["Chats"], summary="Store a completed chat turn for an existing chat")
async def api_log_turn_for_session(session_id: str, body: ChatTurnLogIn, request: Request):
    payload = body.model_dump()
    payload["session_uuid"] = session_id
    return await api_log_turn(ChatTurnLogIn(**payload), request)


@app.post("/proxy/chat/log-turn/", include_in_schema=False)
async def proxy_log_turn(request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/log-turn/"
    logger.info(f"Proxying chat log to {url}")
    return await _proxy_json_request("POST", url, headers=_chat_backend_headers(request), json_body=await request.json())


@app.get("/chatbot/api/users/me/profile", tags=["User Profile"], summary="Read the current user's profile")
async def api_get_user_profile(request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/user/profile/"
    return await _proxy_json_request("GET", url, headers={"Authorization": request.headers.get("Authorization", "")})


@app.patch("/chatbot/api/users/me/profile", tags=["User Profile"], summary="Update the current user's profile")
async def api_patch_user_profile(body: UserProfilePatchIn, request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/user/profile/"
    payload = body.model_dump(exclude_none=True)
    return await _proxy_json_request(
        "PATCH",
        url,
        headers={"Authorization": request.headers.get("Authorization", "")},
        json_body=payload,
    )


@app.get("/chatbot/api/users/me/facts", tags=["User Profile"], summary="Read the current user's stored facts")
async def api_get_user_facts(request: Request, category: Optional[str] = None, limit: int = 10):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/user/facts/"
    params: dict[str, Any] = {"limit": limit}
    if category:
        params["category"] = category
    return await _proxy_json_request(
        "GET",
        url,
        headers={"Authorization": request.headers.get("Authorization", "")},
        params=params,
    )


@app.post("/chatbot/api/users/me/facts", tags=["User Profile"], summary="Add a fact for the current user")
async def api_add_user_fact(body: UserFactCreateIn, request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/chat/user/facts/"
    return await _proxy_json_request(
        "POST",
        url,
        headers={"Authorization": request.headers.get("Authorization", "")},
        json_body=body.model_dump(),
    )


@app.post("/chatbot/api/users/me/profile/build", tags=["User Profile"], summary="Build or update the current user's profile from a conversation turn")
async def api_build_user_profile(body: UserProfileBuildIn, request: Request):
    auth_token = request.headers.get("Authorization", "")
    user_uuid = _extract_user_uuid_from_token(auth_token) if auth_token else None
    if not user_uuid:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    await UserProfileService.process_conversation_turn(
        user_uuid=user_uuid,
        session_uuid=body.session_uuid,
        user_message=body.user_message,
        assistant_message=body.assistant_message,
        auth_token=auth_token,
    )
    return {
        "status": "accepted",
        "message": "Profile build/update request completed",
        "user_uuid": user_uuid,
        "session_uuid": body.session_uuid,
    }


@app.post("/chatbot/api/auth/logout", tags=["Authentication"], summary="Logout the current user")
async def api_logout(body: LogoutIn):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/fastapi/logout/"
    return await _proxy_json_request("POST", url, json_body=body.model_dump())


@app.post("/proxy/logout/", include_in_schema=False)
async def proxy_logout(request: Request):
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    url = f"{S.CHAT_BACKEND_URL}/fastapi/logout/"
    return await _proxy_json_request("POST", url, json_body=await request.json())
