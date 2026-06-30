from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException

from app.config import get_settings
from app.services.auto_eval.models import AutoEvalRunIn
from app.services.auto_eval.runner import run_automation_cycle

router = APIRouter()
S = get_settings()
logger = logging.getLogger(__name__)

# Keep strong references to fire-and-forget background cycles so they are not
# garbage-collected mid-run; entries remove themselves on completion.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _check_automation_token(token: str | None) -> None:
    expected = getattr(S, "AUTOMATION_TOKEN", None)
    if not expected:
        raise HTTPException(status_code=503, detail="AUTOMATION_TOKEN is not configured")
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid automation token")


async def _run_background(body: AutoEvalRunIn, batch_id: str) -> None:
    try:
        summary = await run_automation_cycle(body, batch_id=batch_id)
        logger.info("Automation cycle %s finished: %s", batch_id, {
            "persisted": summary.get("persisted_run_count"),
            "skipped": summary.get("skipped_question_count"),
            "model_outages": summary.get("model_outages"),
        })
    except Exception:
        logger.exception("Automation cycle %s failed", batch_id)


@router.post("/chatbot/api/experiments/automation/runs", tags=["Automation"], summary="Run automated Arena evaluation")
async def run_auto_eval(body: AutoEvalRunIn, x_automation_token: str | None = Header(default=None)):
    _check_automation_token(x_automation_token)

    if body.background:
        # A full cycle (24+ runs, each 3 generations + 2 judges) far outlives a normal
        # request; return immediately and let it run on the event loop.
        batch_id = f"auto_{uuid4().hex[:16]}"
        task = asyncio.create_task(_run_background(body, batch_id))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
        return {"status": "started", "batch_id": batch_id, "background": True}

    try:
        return await run_automation_cycle(body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
