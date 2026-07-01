from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AutoEvalRunIn(BaseModel):
    languages: list[str] | None = None
    # None -> fall back to AUTOMATION_TOPIC_RATIO; an explicit "3:1" still overrides.
    topic_ratio: str | None = None
    base_question_count: int | None = Field(default=None, ge=1, le=100)
    max_concurrency: int | None = Field(default=None, ge=1, le=20)
    # "generate" = make questions + fan out + persist runs (no judging).
    # "judge"    = pull pending persisted runs and run the LLM judges only.
    # "both"     = the full inline cycle (default).
    phase: Literal["generate", "judge", "both"] = "both"
    # Design 2: shared-context comparison (Qwen+EuroLLM generate from one identical prompt;
    # Mistral stays full-system). None -> AUTOMATION_CONTROLLED_ENABLED.
    controlled: bool | None = None
    dry_run: bool = False
    # Run the cycle as a fire-and-forget background task; the endpoint returns a
    # batch_id immediately instead of blocking until every run is persisted.
    background: bool = False
    seed: int | None = None


class PlannedBaseQuestion(BaseModel):
    base_question_id: str
    topic_category: Literal["agriculture", "non_agriculture"]
    # A specific sub-topic to steer the question (e.g. "soil erosion") so questions are unique.
    topic_hint: str = ""


class LocalizedQuestion(BaseModel):
    base_question_id: str
    topic_category: Literal["agriculture", "non_agriculture"]
    source_language: str = "en"
    source_question: str
    language: str
    language_name: str
    question: str


class VariantAnswer(BaseModel):
    variant_id: str
    backend: str
    label: str = ""
    assistant_message: str
    latency_ms: int | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    grounding_mode: str | None = None
    error: str | None = None
    variant_metadata: dict[str, Any] = Field(default_factory=dict)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return bool(self.assistant_message.strip()) and not self.error


class CriterionJudgment(BaseModel):
    winner_label: str
    rationale: str = ""
    scores: dict[str, int] = Field(default_factory=dict)

    @field_validator("winner_label")
    @classmethod
    def validate_winner(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if value not in {"A", "B", "C", "N/A"}:
            raise ValueError("winner_label must be A, B, C, or N/A")
        return value

    @field_validator("scores")
    @classmethod
    def clamp_scores(cls, value: dict[str, int]) -> dict[str, int]:
        clean: dict[str, int] = {}
        for label, score in (value or {}).items():
            key = str(label).strip().upper()
            if key not in {"A", "B", "C"}:
                continue
            try:
                numeric = int(score)
            except (TypeError, ValueError):
                continue
            clean[key] = min(5, max(1, numeric))
        return clean


class JudgeResult(BaseModel):
    evaluator_provider: Literal["openai", "anthropic"]
    evaluator_model: str
    relevant: CriterionJudgment
    most_trustworthy: CriterionJudgment
    clearest: CriterionJudgment
    most_useful: CriterionJudgment
    handled_uncertainty_best: CriterionJudgment
    best_overall: CriterionJudgment
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    error_text: str = ""

    def to_persistence_payload(self, comparison_run_id: str, batch_id: str) -> dict[str, Any]:
        criteria = {
            "relevant": self.relevant,
            "most_trustworthy": self.most_trustworthy,
            "clearest": self.clearest,
            "most_useful": self.most_useful,
            "handled_uncertainty_best": self.handled_uncertainty_best,
            "best_overall": self.best_overall,
        }
        return {
            "comparison_run_id": comparison_run_id,
            "evaluator_provider": self.evaluator_provider,
            "evaluator_model": self.evaluator_model,
            "relevant_label": self.relevant.winner_label,
            "most_trustworthy_label": self.most_trustworthy.winner_label,
            "clearest_label": self.clearest.winner_label,
            "most_useful_label": self.most_useful.winner_label,
            "handled_uncertainty_best_label": self.handled_uncertainty_best.winner_label,
            "best_overall_label": self.best_overall.winner_label,
            "scores": {key: value.scores for key, value in criteria.items()},
            "rationales": {key: value.rationale for key, value in criteria.items()},
            "raw_response": self.raw_response,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "error_text": self.error_text,
            "batch_id": batch_id,
            "metadata": {"source": "automated_llm_judge"},
        }
