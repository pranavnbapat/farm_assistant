from app.routers.ask import _renumber_citations


def test_renumber_simple_singletons():
    # Gapped citations -> clean sequential.
    assert _renumber_citations("A [5] and B [9].", {5: 4, 9: 8}) == "A [4] and B [8]."


def test_renumber_grouped_citation():
    assert _renumber_citations("see [2, 5, 9]", {2: 2, 5: 4, 9: 8}) == "see [2, 4, 8]"


def test_renumber_drops_unmapped_tokens():
    # A token with no mapping is dropped from the group.
    assert _renumber_citations("[5, 7]", {5: 4}) == "[4]"


def test_renumber_empties_fully_unmapped_bracket():
    assert _renumber_citations("x [7] y", {}) == "x  y"


def test_renumber_leaves_plain_text_untouched():
    assert _renumber_citations("plain text, no cites here", {1: 1}) == "plain text, no cites here"
