from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from app.config import get_settings

from .controlled import answer_question_controlled
from .fanout import answer_question, assign_labels
from .judges import evaluate_all
from .models import AutoEvalRunIn, LocalizedQuestion, VariantAnswer
from .persistence import fetch_runs, persist_comparison_run, persist_judge_result, service_login
from .planning import build_base_question_plan, resolve_languages
from .question_gen import generate_localized_questions

S = get_settings()
logger = logging.getLogger(__name__)

ALL_JUDGE_PROVIDERS = {"openai", "anthropic"}


def _question_from_persisted(run: dict) -> LocalizedQuestion:
    """Rebuild a LocalizedQuestion from a persisted comparison run (judge phase)."""
    qm = run.get("question_metadata") or {}
    domain = qm.get("domain")
    if domain not in {"agriculture", "non_agriculture"}:
        domain = "agriculture" if qm.get("is_agriculture") else "non_agriculture"
    question_text = run.get("question") or ""
    return LocalizedQuestion(
        base_question_id=str(qm.get("base_question_id") or run.get("comparison_run_id") or ""),
        topic_category=domain,
        source_language=str(qm.get("source_language") or "en"),
        source_question=str(qm.get("source_question") or question_text),
        language=str(qm.get("language") or "en"),
        language_name=str(qm.get("language_name") or qm.get("language") or "en"),
        question=question_text,
    )


def _answers_from_persisted(run: dict) -> list[VariantAnswer]:
    """Rebuild labelled VariantAnswers from a persisted run's stored answers."""
    answers: list[VariantAnswer] = []
    for entry in run.get("answers") or []:
        message = str(entry.get("assistant_message") or "")
        answers.append(VariantAnswer(
            variant_id=str(entry.get("variant_id") or entry.get("backend") or ""),
            backend=str(entry.get("backend") or ""),
            label=str(entry.get("label") or ""),
            assistant_message=message,
            error=entry.get("error"),
        ))
    return answers


async def _run_generate(inp: AutoEvalRunIn, tokens, batch_id: str, summary: dict, *, do_judge: bool) -> None:
    base_count = inp.base_question_count or getattr(S, "AUTOMATION_BASE_QUESTION_COUNT", 1)
    topic_ratio = inp.topic_ratio or getattr(S, "AUTOMATION_TOPIC_RATIO", "3:1")
    languages = resolve_languages(inp.languages)
    controlled = inp.controlled if inp.controlled is not None else getattr(S, "AUTOMATION_CONTROLLED_ENABLED", False)
    plan = build_base_question_plan(base_count, topic_ratio, seed=inp.seed)
    logger.info("PHASE generate | batch=%s controlled=%s base=%d langs=%d (target %d runs)",
                batch_id, controlled, base_count, len(languages), base_count * len(languages))
    t_gen = time.perf_counter()
    localized_questions = await generate_localized_questions(plan, languages)
    logger.info("PHASE generate | %d questions ready in %.1fs; now answering...",
                len(localized_questions), time.perf_counter() - t_gen)
    semaphore = asyncio.Semaphore(inp.max_concurrency or getattr(S, "AUTOMATION_MAX_CONCURRENCY", 3))
    min_answers = getattr(S, "AUTOMATION_MIN_ANSWERS", 2)
    total_q = len(localized_questions)
    summary["base_question_count"] = base_count
    summary["localized_question_count"] = total_q
    summary["grounding_mismatch_count"] = 0
    summary["retrieval_mode"] = "controlled_shared_context" if controlled else "independent_full_system"

    async def process_question(index: int, question: LocalizedQuestion):
        async with semaphore:
            tag = f"[ans {index + 1}/{total_q}] {question.base_question_id}/{question.language}"
            t_ans = time.perf_counter()
            logger.info("%s answering (%s)...", tag, "controlled" if controlled else "independent")
            if controlled:
                raw_answers = await answer_question_controlled(question, tokens.access_token, tokens.refresh_token, batch_id)
            else:
                raw_answers = await answer_question(question, tokens.access_token, tokens.refresh_token, batch_id)
            usable_answers = [answer for answer in raw_answers if answer.usable]
            logger.info("%s %d/%d usable answers in %.1fs (%s)", tag, len(usable_answers), len(raw_answers),
                        time.perf_counter() - t_ans,
                        ", ".join(f"{a.backend}:{'ok' if a.usable else 'fail'}" for a in raw_answers))
            for answer in raw_answers:
                if answer.error:
                    summary["model_failures"].setdefault(answer.backend, 0)
                    summary["model_failures"][answer.backend] += 1
            if len(usable_answers) < min_answers:
                logger.warning("%s SKIPPED: only %d/%d usable (< min_answers=%d)", tag, len(usable_answers), len(raw_answers), min_answers)
                summary["skipped_questions"].append({
                    "base_question_id": question.base_question_id,
                    "language": question.language,
                    "reason": "fewer_than_min_answers",
                    "usable_answers": len(usable_answers),
                })
                return
            labelled_answers = assign_labels(usable_answers, seed=(inp.seed or 0) + index if inp.seed is not None else None)
            token_rows = [
                {
                    "backend": a.backend,
                    "label": a.label,
                    **(a.runtime_metadata.get("token_usage") or {}),
                }
                for a in labelled_answers
            ]
            grounding_rows = [
                {
                    "backend": a.backend,
                    "label": a.label,
                    "grounding_mode": a.grounding_mode,
                    "sources": len(a.sources),
                }
                for a in labelled_answers
            ]
            # Flag runs where the internal RAG instances disagree on grounding (e.g. one
            # answered `euf_supported` with sources, another `general_knowledge` with none).
            # That signals a retrieval/config confound rather than a real model difference.
            # External backends (Mistral/TNO) report no grounding_mode, so they're excluded.
            internal_modes = {a.grounding_mode for a in labelled_answers if a.grounding_mode}
            grounding_mismatch = len(internal_modes) > 1
            if grounding_mismatch:
                summary["grounding_mismatch_count"] += 1
            judge_results = []
            if do_judge:
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
                    "tokens": token_rows,
                    "grounding": grounding_rows,
                    "grounding_mismatch": grounding_mismatch,
                })
                return
            t_persist = time.perf_counter()
            comparison_run_id = await persist_comparison_run(tokens, batch_id, question, labelled_answers)
            for result in judge_results:
                await persist_judge_result(tokens, comparison_run_id, batch_id, result)
            logger.info("%s persisted run %s (%d answers, %d judges) in %.1fs%s", tag, comparison_run_id,
                        len(labelled_answers), len(judge_results), time.perf_counter() - t_persist,
                        " [grounding mismatch]" if grounding_mismatch else "")
            summary["persisted_runs"].append({
                "comparison_run_id": comparison_run_id,
                "base_question_id": question.base_question_id,
                "language": question.language,
                "answer_count": len(labelled_answers),
                "judge_count": len(judge_results),
                "tokens": token_rows,
                "grounding": grounding_rows,
                "grounding_mismatch": grounding_mismatch,
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


async def _run_judge_pending(inp: AutoEvalRunIn, tokens, summary: dict) -> None:
    experiment_id = getattr(S, "AUTOMATION_EXPERIMENT_ID", "automated_eval")
    runs = await fetch_runs(tokens, experiment_id, limit=500)
    semaphore = asyncio.Semaphore(inp.max_concurrency or getattr(S, "AUTOMATION_MAX_CONCURRENCY", 3))
    min_answers = getattr(S, "AUTOMATION_MIN_ANSWERS", 2)
    summary["candidate_run_count"] = len(runs)
    summary["judged_runs"] = []
    summary["already_judged"] = 0
    summary["skipped_runs"] = []

    async def judge_run(run: dict):
        async with semaphore:
            missing = ALL_JUDGE_PROVIDERS - set(run.get("llm_evaluation_providers") or [])
            if not missing:
                summary["already_judged"] += 1
                return
            usable = [answer for answer in _answers_from_persisted(run) if answer.usable]
            run_id = run.get("comparison_run_id")
            if len(usable) < min_answers or not run_id:
                summary["skipped_runs"].append({"comparison_run_id": run_id, "usable_answers": len(usable)})
                return
            question = _question_from_persisted(run)
            judge_results = await evaluate_all(question, usable, providers=missing)
            for result in judge_results:
                if result.error_text:
                    summary["judge_errors"].setdefault(result.evaluator_provider, 0)
                    summary["judge_errors"][result.evaluator_provider] += 1
            if inp.dry_run:
                summary["judged_runs"].append({"comparison_run_id": run_id, "providers": sorted(missing), "persisted": False})
                return
            batch_id = (run.get("question_metadata") or {}).get("batch_id") or "judge_backfill"
            for result in judge_results:
                await persist_judge_result(tokens, run_id, batch_id, result)
            summary["judged_runs"].append({"comparison_run_id": run_id, "providers": sorted(missing), "persisted": True})

    await asyncio.gather(*(judge_run(run) for run in runs))
    summary["judged_run_count"] = len(summary["judged_runs"])


async def run_automation_cycle(inp: AutoEvalRunIn, batch_id: str | None = None) -> dict:
    if not getattr(S, "AUTOMATION_ENABLED", False):
        raise RuntimeError("Automation is disabled. Set AUTOMATION_ENABLED=true to run it.")

    phase = (inp.phase or "both").lower()
    batch_id = batch_id or f"auto_{uuid4().hex[:16]}"
    t_cycle = time.perf_counter()
    logger.info("=== auto-eval cycle start | batch=%s phase=%s dry_run=%s ===", batch_id, phase, inp.dry_run)
    # Always log in: even a dry run authenticates so fan-out / judging hit the real backends;
    # only the final persistence writes are skipped.
    tokens = await service_login()
    logger.info("service login OK")
    summary = {
        "phase": phase,
        "batch_id": batch_id,
        "dry_run": inp.dry_run,
        "persisted_runs": [],
        "skipped_questions": [],
        "model_failures": {},
        "judge_errors": {},
    }

    if phase in {"generate", "both"}:
        await _run_generate(inp, tokens, batch_id, summary, do_judge=(phase == "both"))
    elif phase == "judge":
        await _run_judge_pending(inp, tokens, summary)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    logger.info("=== auto-eval cycle done | batch=%s | persisted=%s skipped=%s in %.1fs ===",
                batch_id, len(summary.get("persisted_runs", [])), len(summary.get("skipped_questions", [])),
                time.perf_counter() - t_cycle)
    return summary
