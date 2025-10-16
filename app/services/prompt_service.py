# app/services/prompt_service.py

def build_prompt(contexts: list[str], question: str) -> str:
    # Label each context block so the model knows what [1], [2], ... refer to
    labelled = [f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)]
    joined = "\n\n".join(labelled)

    return (
        "System: You are a knowledgeable agricultural assistant who writes in a natural, precise, and concise style—"
        "similar to ChatGPT-5.\n\n"
        "Instructions:\n"
        "- Read ONLY the information provided in the sources below.\n"
        "- Respond naturally in clear, fluent English — informative but not verbose.\n"
        "- Keep your answer short and well-structured (usually 2–4 short paragraphs, or a few bullet points if useful).\n"
        "- Avoid artificial numbering or section labels like '1)', '2)', etc.\n"
        "- Be concise and precise: avoid repetition, filler, or generalities.\n"
        "- Cite sources inline using numeric brackets like [1], [2] immediately after the relevant sentence.\n"
        "- Use only numbers that correspond to the provided sources. Never invent citations.\n"
        "- If the sources are insufficient, briefly say so.\n\n"
        "Formatting style:\n"
        "- Start with a brief overview paragraph introducing the topic.\n"
        "- Present key facts naturally in flowing text or short bullet points.\n"
        "- Conclude with one brief summarising sentence (no new claims).\n\n"
        "Citation example:\n"
        "Precision feeding and robotic milking have improved efficiency in Dutch dairy farms. [2][3]\n\n"
        f"Sources:\n{joined}\n\n"
        f"User question: {question}\n\n"
        "Assistant:"
    )


def build_summary_prompt(user_prompt: str, text: str) -> str:
    """
    Keep user's custom prompt authoritative.
    """
    # Small guard: trim extremely long inputs to fit context
    from app.config import get_settings
    S = get_settings()

    # Leave some headroom for the user prompt and model system tokens
    max_text_chars = max(1000, S.MAX_CONTEXT_CHARS - len(user_prompt) - 1000)
    safe_text = text if len(text) <= max_text_chars else text[:max_text_chars]

    # Present the text under a delimiter the model can recognise easily
    return (
        f"{user_prompt.strip()}\n\n"
        f"--- BEGIN TEXT ---\n"
        f"{safe_text}\n"
        f"--- END TEXT ---\n"
    )

def build_fallback_prompt(intent_info: dict, user_q: str) -> str:
    """
    Small, no-context prompt for non-informational queries.
    - British English, concise, friendly.
    - No citations, no external facts.
    - Nudge the user towards an actionable question.
    """
    intent = (intent_info.get("intent") or "other").lower()
    norm_q = intent_info.get("normalised") or user_q

    if intent == "greeting":
        style = (
            "You are friendly and brief. Greet the user naturally, then ask how you can help with an "
            "information request or task. Keep it to one or two short sentences."
        )
    elif intent == "chit-chat":
        style = (
            "You are light and concise. Give a brief, friendly one-liner, then invite the user to ask a "
            "specific question you can help with. One or two short sentences max."
        )
    elif intent == "garbage":
        style = (
            "The input looks noisy or unclear. Politely ask for clarification and offer an example of a "
            "clear, specific question. Keep it to one sentence."
        )
    else:  # "other" or low-confidence intent
        style = (
            "The input is unclear. Ask one clarifying question to narrow the user’s goal. One sentence."
        )

    return (
        "System: You are a precise, concise assistant (British English). "
        "Do not invent facts. Do not use citations. Keep answers short.\n\n"
        f"Instruction: {style}\n\n"
        f"User input: {norm_q}\n\n"
        "Assistant:"
    )
