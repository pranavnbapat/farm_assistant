from app.services import search_service
from app.services.context_service import estimate_semantic_quality
from app.schemas import AskIn


# --- estimate_semantic_quality (context_service) ---------------------------

def test_semantic_quality_is_best_of_top_n():
    items = [
        {"semantic_score": 0.6},
        {"semantic_score": 0.8},   # best within top_n -> this is returned
        {"semantic_score": 0.4},
        {"semantic_score": 0.95},  # outside top_n=3, ignored
    ]
    assert estimate_semantic_quality(items, top_n=3) == 0.8


def test_semantic_quality_none_when_field_absent():
    # No semantic_score on any item -> None so the caller falls back to word-overlap.
    assert estimate_semantic_quality([{"title": "x"}, {"title": "y"}]) is None


def test_semantic_quality_uses_only_scored_items():
    items = [{"title": "x"}, {"semantic_score": 0.5}]
    assert estimate_semantic_quality(items, top_n=3) == 0.5


# --- build_search_payload opt-in flag (search_service) ----------------------

def test_payload_requests_semantic_score_in_semantic_mode(monkeypatch):
    monkeypatch.setattr(search_service.S, "RELEVANCE_MODE", "semantic")
    payload = search_service.build_search_payload(AskIn(question="soil health"))
    assert payload.get("include_semantic_score") is True


def test_payload_omits_flag_in_overlap_mode(monkeypatch):
    monkeypatch.setattr(search_service.S, "RELEVANCE_MODE", "overlap")
    payload = search_service.build_search_payload(AskIn(question="soil health"))
    assert "include_semantic_score" not in payload
