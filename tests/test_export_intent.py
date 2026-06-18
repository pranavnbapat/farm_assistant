import asyncio

from app.routers import ask
from app.schemas import ExportIntentIn


def test_parse_export_previous_intent():
    result = ask._parse_export_intent(
        '{"intent":"export_previous","format":"pdf","confidence":0.97}'
    )

    assert result.intent == "export_previous"
    assert result.format == "pdf"
    assert result.confidence == 0.97


def test_parse_low_confidence_falls_back_to_normal_chat():
    result = ask._parse_export_intent(
        '{"intent":"generate_export","format":"xlsx","confidence":0.4}'
    )

    assert result.intent == "normal_chat"
    assert result.format is None
    assert result.meta["reason"] == "low_confidence"


def test_parse_invalid_classifier_output_falls_back_to_normal_chat():
    result = ask._parse_export_intent("not json")

    assert result.intent == "normal_chat"
    assert result.format is None
    assert result.meta["reason"] == "invalid_json"


def test_classifier_accepts_multilingual_export_request(monkeypatch):
    captured = {}

    async def fake_generate_once(prompt, temperature, max_tokens, messages=None):
        captured["messages"] = messages
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return '{"intent":"export_previous","format":"pdf","confidence":0.99}'

    monkeypatch.setattr(ask, "generate_once", fake_generate_once)

    result = asyncio.run(
        ask.classify_export_intent(
            ExportIntentIn(
                query="この回答をPDFにしてください",
                previous_assistant_message="Previous agricultural answer",
            )
        )
    )

    assert result.intent == "export_previous"
    assert result.format == "pdf"
    assert result.confidence == 0.99
    assert "この回答をPDFにしてください" in captured["messages"][1]["content"]
    assert captured["messages"][0]["role"] == "system"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 96



def test_parse_export_intent_includes_scope():
    result = ask._parse_export_intent(
        "{\"intent\":\"export_previous\",\"format\":\"pdf\",\"scope\":\"conversation\",\"confidence\":0.98}"
    )

    assert result.intent == "export_previous"
    assert result.format == "pdf"
    assert result.scope == "conversation"


def test_deterministic_conversation_export_scope():
    result = ask._deterministic_export_intent(
        "Please put all of this in a PDF and give me.",
        has_conversation=True,
    )

    assert result is not None
    assert result.intent == "export_previous"
    assert result.format == "pdf"
    assert result.scope == "conversation"
    assert result.confidence == 1.0


def test_deterministic_format_followup_reuses_previous_export_scope():
    result = ask._deterministic_export_intent(
        "What about an excel?",
        previous_export_scope="conversation",
        has_conversation=True,
    )

    assert result is not None
    assert result.format == "xlsx"
    assert result.scope == "conversation"


def test_classifier_returns_deterministic_conversation_scope_without_llm(monkeypatch):
    async def fail_generate_once(*args, **kwargs):
        raise AssertionError("classifier should not call LLM for deterministic export scope")

    monkeypatch.setattr(ask, "generate_once", fail_generate_once)

    result = asyncio.run(
        ask.classify_export_intent(
            ExportIntentIn(
                query="Please put all of this in a PDF and give me.",
                previous_assistant_message="Previous answer",
                conversation_messages=[
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            )
        )
    )

    assert result.intent == "export_previous"
    assert result.format == "pdf"
    assert result.scope == "conversation"
    assert result.meta["model"] == "deterministic"
