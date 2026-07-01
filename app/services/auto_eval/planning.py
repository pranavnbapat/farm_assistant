from __future__ import annotations

import random
from uuid import uuid4

from .constants import (
    AGRICULTURE_SUBTOPICS,
    EU_LANGUAGE_BY_CODE,
    EU_LANGUAGES,
    NON_AGRICULTURE_TOPICS,
)
from .models import PlannedBaseQuestion


def parse_topic_ratio(raw: str) -> tuple[int, int]:
    parts = (raw or "3:1").split(":", 1)
    if len(parts) != 2:
        raise ValueError("topic_ratio must be formatted as agriculture:non_agriculture, e.g. 3:1")
    agri = int(parts[0])
    non_agri = int(parts[1])
    if agri < 1 or non_agri < 1:
        raise ValueError("topic_ratio values must be positive")
    return agri, non_agri


def resolve_languages(codes: list[str] | None) -> list[dict[str, str]]:
    if not codes:
        return EU_LANGUAGES
    result = []
    for code in codes:
        normalized = code.strip().lower()
        if normalized not in EU_LANGUAGE_BY_CODE:
            raise ValueError(f"Unsupported EU language code: {code}")
        result.append(EU_LANGUAGE_BY_CODE[normalized])
    return result


def build_base_question_plan(count: int, topic_ratio: str, seed: int | None = None) -> list[PlannedBaseQuestion]:
    agri, non_agri = parse_topic_ratio(topic_ratio)
    pattern = ["agriculture"] * agri + ["non_agriculture"] * non_agri
    rng = random.Random(seed)
    categories = [pattern[index % len(pattern)] for index in range(count)]
    rng.shuffle(categories)

    # Shuffle each sub-topic pool and walk through it so consecutive questions get DISTINCT
    # sub-topics (cycling only once the pool is exhausted). This is what makes questions unique
    # instead of the model defaulting to the same go-to question.
    ag_pool = AGRICULTURE_SUBTOPICS[:]
    nonag_pool = NON_AGRICULTURE_TOPICS[:]
    rng.shuffle(ag_pool)
    rng.shuffle(nonag_pool)
    ag_i = nonag_i = 0

    plan: list[PlannedBaseQuestion] = []
    for category in categories:
        if category == "agriculture":
            hint = ag_pool[ag_i % len(ag_pool)]
            ag_i += 1
        else:
            hint = nonag_pool[nonag_i % len(nonag_pool)]
            nonag_i += 1
        plan.append(PlannedBaseQuestion(
            base_question_id=f"bq_{uuid4().hex[:12]}",
            topic_category=category,
            topic_hint=hint,
        ))
    return plan
