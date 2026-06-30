from __future__ import annotations

import random
from uuid import uuid4

from .constants import EU_LANGUAGE_BY_CODE, EU_LANGUAGES
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
    return [
        PlannedBaseQuestion(base_question_id=f"bq_{uuid4().hex[:12]}", topic_category=category)
        for category in categories
    ]
