# nlp/intent_embeddings.py

import numpy as np

from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

from .normalise import normalise_and_score, question_likeness

GARBAGE_THRESHOLD = 0.80

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # fast on CPU

# Keep 5–10 prototypes per class;
PROTOTYPES = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good evening",
        "how are you", "how's it going"
    ],
    "chit-chat": [
        "what's up", "how's your day", "nice weather", "lol", "haha"
    ],
    "information question": [
        "what is the capital of", "who invented", "explain how", "define"
    ],
    "task request": [
        "build me", "help me implement", "how do I deploy", "fix this bug",
        "write a script to", "create an API that"
    ]
}

# Precompute prototype embeddings once (e.g., at app startup)
proto_texts = [p for plist in PROTOTYPES.values() for p in plist]
proto_labels = [lbl for lbl, plist in PROTOTYPES.items() for _ in plist]
proto_embs = model.encode(proto_texts, normalize_embeddings=True)

@lru_cache(maxsize=512)
def decide_intent_cached(text: str):
    return decide_intent(text)

def classify(text: str) -> dict:
    q = model.encode([text], normalize_embeddings=True)
    sims = util.cos_sim(q, proto_embs).cpu().numpy()[0]   # cosine similarity
    best_idx = int(np.argmax(sims))
    return {"label": proto_labels[best_idx], "score": float(sims[best_idx])}

def decide_intent(text: str) -> dict:
    norm, g = normalise_and_score(text)

    # Hard garbage: very noisy + short
    if g >= GARBAGE_THRESHOLD and len(norm) < 20:
        return {"original": text, "normalised": norm, "garbage_score": g, "intent": "garbage", "confidence": g}

    qlikeness = question_likeness(norm)

    # Require both "looks like NL" AND enough content *or* a question mark
    import regex as re
    toks = re.findall(r"[A-Za-z]+", norm)
    has_qmark = "?" in norm
    enough_content = (len(toks) >= 3)

    if qlikeness >= 0.60 and g < 0.70 and (enough_content or has_qmark):
        return {"original": text, "normalised": norm, "garbage_score": g, "intent": "information question",
                "confidence": qlikeness}

    # If it doesn't meet those structural conditions, don't let embeddings up-class it
    pred = classify(norm)
    if qlikeness < 0.45 and len(toks) <= 2 and not has_qmark:
        # short & not question-like → treat as "other"
        return {"original": text, "normalised": norm, "garbage_score": g, "intent": "other",
                "confidence": 1.0 - qlikeness}

    # Else keep the embedding label, but remember it’s advisory
    return {"original": text, "normalised": norm, "garbage_score": g, "intent": pred["label"],
            "confidence": pred["score"]}

