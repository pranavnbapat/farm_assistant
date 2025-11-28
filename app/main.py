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
from fastapi.staticfiles import StaticFiles
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


FA_ENV = os.getenv("FA_ENV", "local")

LOGIN_ENDPOINTS = {
    "local": "http://127.0.0.1:8000/fastapi/login/",
    "dev":   "https://backend-admin.dev.farmbook.ugent.be/fastapi/login/",
    "prd":   "https://backend-admin.prd.farmbook.ugent.be/fastapi/login/",
}

LOGIN_URL = LOGIN_ENDPOINTS.get(FA_ENV, LOGIN_ENDPOINTS["local"])
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "")

ENABLE_DOCS: bool = True

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
def chat_page():
    """Chat UI – frontend will redirect to / if not logged in."""
    return CHAT_HTML.read_text(encoding="utf-8")

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

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=True) as client:
            upstream = await client.post(LOGIN_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to reach auth backend: {exc}"
        )

    if upstream.status_code != 200:
        # Login failed on the auth backend
        try:
            error_text = upstream.text
        except Exception:
            error_text = ""
        raise HTTPException(
            status_code=upstream.status_code,
            detail=f"Auth backend error: {error_text}"
        )

    # Return exactly what the auth backend sent (token etc.)
    try:
        data = upstream.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Auth backend returned non-JSON response"
        )

    return JSONResponse(content=data, status_code=200)
