# app/main.py

import logging

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.routers.ask import router as ask_router

S = get_settings()

logging.basicConfig(
    level=S.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
)
logger = logging.getLogger("farm-assistant")

app = FastAPI(title=S.APP_TITLE, version=S.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
INDEX_HTML = TEMPLATES_DIR / "ask_stream.html"

# Health-check (optional)
@app.get("/health")
async def health():
    return {"status": "ok"}

# Mount feature routers
app.include_router(ask_router)

@app.get("/", response_class=HTMLResponse)
def ui():
    # Read the static HTML (no templating needed)
    return INDEX_HTML.read_text(encoding="utf-8")
