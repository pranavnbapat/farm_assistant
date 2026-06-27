from app.routers.ask import _hard_route_turn_mode


def test_math_and_academic_questions_hard_route_offtopic():
    assert _hard_route_turn_mode("What is the square root of 2342423423?") == "off_topic"
    assert _hard_route_turn_mode("What is vedic maths?") == "off_topic"
    assert _hard_route_turn_mode("Explain the quadratic formula") == "off_topic"
    assert _hard_route_turn_mode("Teach me trigonometry") == "off_topic"


def test_agriculture_questions_are_not_hard_routed_offtopic():
    # Substantive on-topic questions return None so the normal router/retrieval path runs.
    assert _hard_route_turn_mode("What are examples of cover crops?") is None
    assert _hard_route_turn_mode("How do I manage potato late blight?") is None
