#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.auto_eval.models import AutoEvalRunIn
from app.services.auto_eval.runner import run_automation_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated Farm Assistant Arena evaluation")
    parser.add_argument("--languages", default="", help="Comma-separated EU language codes; default is all 24")
    parser.add_argument("--topic-ratio", default="3:1")
    parser.add_argument("--base-question-count", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument(
        "--phase", choices=["generate", "judge", "both"], default="both",
        help="generate = questions+answers only; judge = score all pending persisted runs; both = full cycle",
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING (default INFO = step-by-step progress)")
    parser.add_argument("--dry-run", action="store_true")
    controlled = parser.add_mutually_exclusive_group()
    controlled.add_argument("--controlled", dest="controlled", action="store_true",
                            help="Design 2: Qwen+EuroLLM generate from one shared context (model-only comparison)")
    controlled.add_argument("--independent", dest="controlled", action="store_false",
                            help="Design 1: each model runs its own full RAG pipeline")
    parser.set_defaults(controlled=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    body = AutoEvalRunIn(
        languages=[code.strip() for code in args.languages.split(",") if code.strip()] or None,
        topic_ratio=args.topic_ratio,
        base_question_count=args.base_question_count,
        max_concurrency=args.max_concurrency,
        phase=args.phase,
        controlled=args.controlled,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    result = await run_automation_cycle(body)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
