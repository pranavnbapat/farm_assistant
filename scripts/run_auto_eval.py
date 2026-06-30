#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    body = AutoEvalRunIn(
        languages=[code.strip() for code in args.languages.split(",") if code.strip()] or None,
        topic_ratio=args.topic_ratio,
        base_question_count=args.base_question_count,
        max_concurrency=args.max_concurrency,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    result = await run_automation_cycle(body)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
