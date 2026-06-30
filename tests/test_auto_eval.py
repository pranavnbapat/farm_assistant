from app.services.auto_eval.fanout import assign_labels
from app.services.auto_eval.models import CriterionJudgment, VariantAnswer
from app.services.auto_eval.planning import build_base_question_plan, parse_topic_ratio, resolve_languages


def test_parse_topic_ratio():
    assert parse_topic_ratio("3:1") == (3, 1)


def test_build_base_question_plan_honors_ratio_count():
    plan = build_base_question_plan(8, "3:1", seed=42)
    categories = [item.topic_category for item in plan]
    assert categories.count("agriculture") == 6
    assert categories.count("non_agriculture") == 2


def test_resolve_languages_defaults_to_all_24():
    assert len(resolve_languages(None)) == 24
    assert [item["code"] for item in resolve_languages(["en", "fr"])] == ["en", "fr"]


def test_assign_labels_is_stable_with_seed():
    answers = [
        VariantAnswer(variant_id="qwen3", backend="um_qwen3", assistant_message="a"),
        VariantAnswer(variant_id="mistral", backend="euf_chatbot_tnods", assistant_message="b"),
        VariantAnswer(variant_id="eurollm", backend="eurollm", assistant_message="c"),
    ]
    labelled = assign_labels(answers, seed=1)
    assert [answer.label for answer in labelled] == ["A", "B", "C"]
    assert sorted(answer.variant_id for answer in labelled) == ["eurollm", "mistral", "qwen3"]


def test_criterion_judgment_clamps_scores():
    judgment = CriterionJudgment(winner_label="a", scores={"A": 7, "B": 0, "Z": 5})
    assert judgment.winner_label == "A"
    assert judgment.scores == {"A": 5, "B": 1}
