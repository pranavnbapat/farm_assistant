from __future__ import annotations

import asyncio
from uuid import uuid4

from app.config import get_settings

from .fanout import answer_question, assign_labels
from .judges import evaluate_all
from .models import AutoEvalRunIn, LocalizedQuestion
from .persistence import persist_comparison_run, persist_judge_result, service_login
from .planning import build_base_question_plan, resolve_languages
from .question_gen import generate_localized_questions

S = get_settings()


async def run_automation_cycle(inp: AutoEvalRunIn, batch_id: str | None = None) -> dict:
    if not getattr(S, "AUTOMATION_ENABLED", False):
        raise RuntimeError("Automation is disabled. Set AUTOMATION_ENABLED=true to run it.")

    batch_id = batch_id or f"auto_{uuid4().hex[:16]}"
    base_count = inp.base_question_count or getattr(S, "AUTOMATION_BASE_QUESTION_COUNT", 1)
    topic_ratio = inp.topic_ratio or getattr(S, "AUTOMATION_TOPIC_RATIO", "3:1")
    languages = resolve_languages(inp.languages)
    plan = build_base_question_plan(base_count, topic_ratio, seed=inp.seed)
    localized_questions = await generate_localized_questions(plan, languages)
    # Always log in: a dry run must still authenticate so fan-out + judging are exercised
    # against the real backends; only the final persistence writes are skipped.
    tokens = await service_login()
    semaphore = asyncio.Semaphore(inp.max_concurrency or getattr(S, "AUTOMATION_MAX_CONCURRENCY", 3))
    min_answers = getattr(S, "AUTOMATION_MIN_ANSWERS", 2)
    summary = {
        "batch_id": batch_id,
        "dry_run": inp.dry_run,
        "base_question_count": base_count,
        "localized_question_count": len(localized_questions),
        "persisted_runs": [],
        "skipped_questions": [],
        "model_failures": {},
        "judge_errors": {},
    }

    async def process_question(index: int, question: LocalizedQuestion):
        async with semaphore:
            raw_answers = await answer_question(question, tokens.access_token, tokens.refresh_token, batch_id)
            usable_answers = [answer for answer in raw_answers if answer.usable]
            for answer in raw_answers:
                if answer.error:
                    summary["model_failures"].setdefault(answer.backend, 0)
                    summary["model_failures"][answer.backend] += 1
            if len(usable_answers) < min_answers:
                summary["skipped_questions"].append({
                    "base_question_id": question.base_question_id,
                    "language": question.language,
                    "reason": "fewer_than_min_answers",
                    "usable_answers": len(usable_answers),
                })
                return
            labelled_answers = assign_labels(usable_answers, seed=(inp.seed or 0) + index if inp.seed is not None else None)
            judge_results = await evaluate_all(question, labelled_answers)
            for result in judge_results:
                if result.error_text:
                    summary["judge_errors"].setdefault(result.evaluator_provider, 0)
                    summary["judge_errors"][result.evaluator_provider] += 1
            if inp.dry_run:
                summary["persisted_runs"].append({
                    "comparison_run_id": None,
                    "base_question_id": question.base_question_id,
                    "language": question.language,
                    "answer_count": len(labelled_answers),
                    "judge_count": len(judge_results),
                })
                return
            comparison_run_id = await persist_comparison_run(tokens, batch_id, question, labelled_answers)
            for result in judge_results:
                await persist_judge_result(tokens, comparison_run_id, batch_id, result)
            summary["persisted_runs"].append({
                "comparison_run_id": comparison_run_id,
                "base_question_id": question.base_question_id,
                "language": question.language,
                "answer_count": len(labelled_answers),
                "judge_count": len(judge_results),
            })

    await asyncio.gather(*(process_question(index, question) for index, question in enumerate(localized_questions)))
    summary["persisted_run_count"] = len(summary["persisted_runs"])
    summary["skipped_question_count"] = len(summary["skipped_questions"])
    # A backend that failed on *every* localized question is a full outage, not a one-off
    # glitch — surface it so a silently-down chatbot (e.g. Mistral) is visible even though
    # cycles still "succeed" on the remaining 2/3 answers.
    attempted = len(localized_questions)
    summary["model_outages"] = [
        backend for backend, failures in summary["model_failures"].items()
        if attempted and failures >= attempted
    ]
    return summary
