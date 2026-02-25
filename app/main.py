# app/main.py

import asyncio
import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast, Any

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
from app.routers.tts import router as tts_router

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
app.include_router(tts_router)

@app.get("/", response_class=HTMLResponse)
def login_page():
    """Public login page."""
    return LOGIN_HTML.read_text(encoding="utf-8")


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    """Chat UI – frontend will redirect to / if not logged in."""
    return templates.TemplateResponse(
        "ask_stream.html",
        {
            "request": request,
            "FA_ENV": S.FA_ENV,
        },
    )
    # return CHAT_HTML.read_text(encoding="utf-8")

class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/api/login")
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

@app.get("/proxy/chat/sessions/")
async def proxy_get_sessions(request: Request):
    """Proxy GET sessions request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/"
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.get(url, headers=headers)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy sessions: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/proxy/chat/sessions/")
async def proxy_create_session(request: Request):
    """Proxy POST session request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/"
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }
    body = await request.json()
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.post(url, json=body, headers=headers)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy create session: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/proxy/chat/sessions/{session_id}/")
async def proxy_get_session(session_id: str, request: Request):
    """Proxy GET single session request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.get(url, headers=headers)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy get session: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.patch("/proxy/chat/sessions/{session_id}/")
async def proxy_patch_session(session_id: str, request: Request):
    """Proxy PATCH session request to Django backend (e.g., rename title)."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")

    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }
    body = await request.json()

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.patch(url, json=body, headers=headers)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy patch session: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/proxy/chat/sessions/{session_id}/")
async def proxy_delete_session(session_id: str, request: Request):
    """Proxy DELETE session request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.delete(url, headers=headers)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy delete session: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/proxy/chat/log-turn/")
async def proxy_log_turn(request: Request):
    """Proxy chat turn logging to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/chat/log-turn/"
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Refresh-Token": request.headers.get("X-Refresh-Token", ""),
    }
    body = await request.json()
    
    logger.info(f"Proxying chat log to {url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.post(url, json=body, headers=headers)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy log turn: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/proxy/logout/")
async def proxy_logout(request: Request):
    """Proxy logout request to Django backend."""
    if not S.CHAT_BACKEND_URL:
        raise HTTPException(status_code=503, detail="Chat backend not configured")
    
    url = f"{S.CHAT_BACKEND_URL}/fastapi/logout/"
    body = await request.json()
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=S.VERIFY_SSL) as client:
            upstream = await client.post(url, json=body)
            return _relay_upstream_response(upstream)
    except httpx.HTTPError as e:
        logger.error(f"Failed to proxy logout: {e}")
        raise HTTPException(status_code=502, detail=str(e))
