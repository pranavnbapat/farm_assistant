# nlp/normalise.py

from __future__ import annotations

import regex as re

from typing import Tuple, List

from symspellpy import SymSpell, Verbosity
from wordfreq import zipf_frequency
from wordsegment import load as ws_load, segment as ws_segment

# ------------- initialise global resources once -------------
ws_load()  # loads word frequency list for segmentation

_sym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
for w in ["use", "of", "drones", "in", "norway", "good", "morning", "hello", "hi", "how", "are", "you"]:
    _sym.create_dictionary_entry(w, 1)

WORD = re.compile(r"[A-Za-z]+", re.U)

def _segment_if_unspaced(text: str) -> str:
    s = text.strip()
    if " " not in s and len(s) >= 8 and s.isalpha():
        # wordsegment returns lowercase tokens
        return " ".join(ws_segment(s))
    return s

def _symspell_correct_token(tok: str) -> str:
    # Skip non-alpha and very short tokens — prevents "asd" → "use"
    if not tok.isalpha() or len(tok) <= 3:
        return tok

    if zipf_frequency(tok.lower(), "en") >= 3.5:
        return tok

    sug = _sym.lookup(tok.lower(), Verbosity.TOP, max_edit_distance=1)
    return sug[0].term if sug else tok


def _spell_correct_line(text: str) -> str:
    toks = re.findall(r"[A-Za-z]+|[^A-Za-z\s]+|\s+", text)

    # Track corrections to avoid over-correcting noisy strings
    alpha_tokens = [t for t in toks if WORD.fullmatch(t)]
    before = alpha_tokens[:]

    out: List[str] = []
    for t in toks:
        if WORD.fullmatch(t):
            out.append(_symspell_correct_token(t))
        else:
            out.append(t)
    corrected = "".join(out)

    # Compute correction rate
    after_tokens = re.findall(r"[A-Za-z]+", corrected)
    changed = sum(1 for a, b in zip(before, after_tokens[:len(before)]) if a != b)
    len_diff = abs(len(after_tokens) - len(before))
    if before:
        corr_rate = (changed + len_diff) / len(before)
        if corr_rate > 0.30:
            return text
    return corrected

def _garbage_score(text: str) -> float:
    """
    Rough 0..1 score. >0.7 => likely garbage.
    Components:
      - proportion of tokens with very low Zipf frequency
      - symbol noise ratio (non-letters)
      - lack of vowels (e.g., 'fjghsd')
    """
    s = text.strip()
    if not s:
        return 1.0

    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    others = len(s) - (letters + digits)  # symbols only
    symbol_ratio = others / max(1, len(s))

    tokens = [t for t in re.findall(r"[A-Za-z]+", s)]
    if not tokens:
        return min(1.0, 0.5 + symbol_ratio)

    low_freq = sum(1 for t in tokens if zipf_frequency(t.lower(), "en") < 2.0)
    vowelless = sum(1 for t in tokens if not re.search(r"[aeiou]", t.lower()))

    rare_ratio = low_freq / len(tokens)
    vowelless_ratio = vowelless / len(tokens)

    # Slightly reduce the symbol weight; digits are common
    score = 0.5 * rare_ratio + 0.30 * symbol_ratio + 0.20 * vowelless_ratio
    return max(0.0, min(1.0, score))

def normalise_and_score(text: str) -> Tuple[str, float]:
    """
    Returns (normalised_text, garbage_score).
    Pipeline: trim → lower for segmentation → segment_if_unspaced → spell-correct → tidy spaces
    """
    if not text or not text.strip():
        return "", 1.0

    s = text.strip()
    # Try segmentation only if looks like a glued word string
    segged = _segment_if_unspaced(s.lower())
    # Keep original casing for multi-word proper nouns? For this task, lower is fine.
    corrected = _spell_correct_line(segged)
    corrected = re.sub(r"\s+", " ", corrected).strip()
    g = _garbage_score(corrected)
    return corrected, g

def question_likeness(text: str) -> float:
    """
    0..1 score (no word lists) favouring natural-language queries:
      + sensible length
      + 2+ tokens (3+ better)
      + mostly letters
      + average Zipf frequency (real words)
      + vowel presence in tokens (gibberish often lacks vowels)
      + small bonus for '?'
    """
    import regex as re
    from wordfreq import zipf_frequency

    s = (text or "").strip()
    if not s:
        return 0.0

    n = len(s)
    # length: 10..180 ideal; 6..10 or 180..260 acceptable
    chars_ok = 1.0 if 10 <= n <= 180 else 0.5 if 6 <= n < 10 or 180 < n <= 260 else 0.0

    letters = sum(c.isalpha() for c in s)
    alpha_ratio = letters / max(1, n)

    toks = re.findall(r"[A-Za-z]+", s)
    tok_count = len(toks)
    tok_ok = 1.0 if tok_count >= 3 else 0.7 if tok_count == 2 else 0.3 if tok_count == 1 else 0.0

    if toks:
        zipfs = [zipf_frequency(t.lower(), "en") for t in toks]
        avg_zipf = sum(zipfs) / len(zipfs)
        min_zipf = min(zipfs)
        vowels_ok_ratio = sum(1 for t in toks if re.search(r"[aeiou]", t.lower())) / len(toks)
    else:
        avg_zipf = 0.0
        min_zipf = 0.0
        vowels_ok_ratio = 0.0

    # Map to [0..1]
    zipf_ok   = 1.0 if avg_zipf >= 3.1 else 0.7 if avg_zipf >= 2.7 else 0.2
    min_ok    = 1.0 if min_zipf >= 2.3 else 0.5 if min_zipf >= 1.8 else 0.0
    vowels_ok = min(1.0, max(0.0, (vowels_ok_ratio - 0.4) / 0.6))  # 0 at 0.4, 1 at 1.0

    qmark_bonus = 0.1 if "?" in s else 0.0

    # Heavier weight on "real-word-ness"
    score = (
        0.30 * zipf_ok +
        0.20 * min_ok +
        0.20 * min(1.0, alpha_ratio) +
        0.15 * tok_ok +
        0.10 * chars_ok +
        0.05 * vowels_ok +
        qmark_bonus
    )
    return max(0.0, min(1.0, score))


