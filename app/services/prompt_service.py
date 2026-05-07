# app/services/prompt_service.py

from typing import Optional


_IDENTITY = "You are an agricultural assistant for EU-FarmBook."

_LANGUAGE_RULE = (
    "Respond in the same language as the user's latest message, "
    "unless the user explicitly asks for another language."
)

_FOLLOWUP_RULE = (
    "If a follow-up question would help the user, end with one. "
    "Skip the follow-up for greetings, thanks, confirmations, or closings."
)

_HISTORY_USE_RULE = (
    "Use the prior conversation for continuity when the user refers to earlier turns."
)


def _normalize_history_messages(history_messages: Optional[list[dict]]) -> list[dict]:
    """
    Map raw stored messages ({role, content, ...}) into OpenAI-style
    {role: 'user' | 'assistant', content: str}, dropping empties and unknown roles.
    """
    if not history_messages:
        return []
    out: list[dict] = []
    for m in history_messages:
        if not isinstance(m, dict):
            continue
        role_in = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role_in in ("user", "you", "human"):
            role = "user"
        elif role_in in ("assistant", "model", "ai", "bot"):
            role = "assistant"
        else:
            continue
        out.append({"role": role, "content": content})
    return out


def _assemble_messages(
    system_text: str,
    history_messages: Optional[list[dict]],
    user_content: str,
    user_profile_context: Optional[str] = None,
) -> list[dict]:
    if user_profile_context:
        # Frame the profile as latent background — not as facts to drop into every
        # reply. Without this guard, instruction-tuned models grab the region/farm
        # type and shoehorn it into greetings, acknowledgements, and unrelated
        # chit-chat ("...your farming activities in Norway").
        system_text = (
            f"{system_text}\n\n"
            "## Background you have learned about this user (from prior conversations)\n"
            f"{user_profile_context}\n\n"
            "Use this background **only** when it is directly relevant to the user's "
            "current question. Do not bring up the user's region, farm type, crops, "
            "or other profile details in greetings, acknowledgements, thanks, "
            "small talk, or otherwise unrelated turns. Treat it as something you "
            "happen to know, not as a topic to introduce."
        )
    messages: list[dict] = [{"role": "system", "content": system_text}]
    messages.extend(_normalize_history_messages(history_messages))
    messages.append({"role": "user", "content": user_content})
    return messages


def _attach_sources(question: str, contexts: list[str]) -> str:
    """
    For retrieval turns, prepend a numbered sources block to the user's question.
    Citations stay anchored to the [N] used in this block.
    """
    if not contexts:
        return question
    labelled = [f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)]
    sources_block = "\n\n".join(labelled)
    return (
        "Relevant EU-FarmBook sources for this question:\n\n"
        f"{sources_block}\n\n"
        f"Question: {question}"
    )


def build_messages(
    contexts: list[str],
    question: str,
    history_messages: Optional[list[dict]] = None,
    user_profile_context: Optional[str] = None,
    has_relevant_sources: bool = True,
) -> list[dict]:
    """Standard retrieval-grounded turn."""
    directives = [
        _IDENTITY,
        (
            "Only answer questions related to agriculture, farming, agri-tech, food systems, "
            "or EU-FarmBook project topics. For anything outside that scope, politely refuse "
            "in 1-2 sentences and ask the user to ask an agriculture-related question instead."
        ),
        _LANGUAGE_RULE,
        f"Answer the user's actual question directly before adding extra detail. {_HISTORY_USE_RULE}",
        (
            "Cite sources inline as [1], [2] etc., matching the numbers in the sources block on the user's turn. "
            "Cite only numbers that exist in that block. Do not invent citation numbers."
        ),
        (
            "If a claim is not supported by the available sources, say that plainly. "
            "If the request is ambiguous or depends on missing facts, ask one specific clarifying question."
        ),
    ]
    if has_relevant_sources:
        directives.append(
            "When sources are present in the user's turn, treat them as the primary grounding material for your answer."
        )
    else:
        directives.append(
            "No EU-FarmBook source material was found for this turn. Say so in one short sentence, "
            "then give a cautious best-effort answer from general agricultural knowledge. "
            "Do not add citations and do not imply the sources support the fallback answer."
        )
    directives.append(
        "If uploaded PDF content appears in the sources block, treat it as available context "
        "and answer from it; do not say you cannot access files."
    )
    directives.append(_FOLLOWUP_RULE)

    system_text = "\n\n".join(directives)
    user_content = _attach_sources(question, contexts)
    return _assemble_messages(system_text, history_messages, user_content, user_profile_context)


def build_history_only_messages(
    question: str,
    history_messages: Optional[list[dict]] = None,
    user_profile_context: Optional[str] = None,
) -> list[dict]:
    """The user is asking about the conversation itself. Stay strictly within history."""
    directives = [
        _IDENTITY,
        (
            "The user is asking about the conversation itself. "
            "Answer strictly from the prior conversation in this thread. "
            "Do not use outside knowledge or retrieved sources to fill gaps. "
            "If the conversation does not contain the requested information, say that plainly."
        ),
        _LANGUAGE_RULE,
        (
            "Be concrete, brief, and faithful to what was actually said. "
            "Skip a follow-up question when a direct recap is enough."
        ),
    ]
    system_text = "\n\n".join(directives)
    user_content = question
    msgs = _assemble_messages(system_text, history_messages, user_content, user_profile_context)
    if not _normalize_history_messages(history_messages):
        # Inject a small note so the model doesn't fabricate a recap.
        msgs[0]["content"] += (
            "\n\nNote: there is no earlier conversation in the current session. "
            "Acknowledge this honestly if the user asks for a recap."
        )
    return msgs


def build_conversation_only_messages(
    question: str,
    history_messages: Optional[list[dict]] = None,
    user_profile_context: Optional[str] = None,
) -> list[dict]:
    """
    Lightweight conversational turn (greeting, thanks, confirmation, brief clarification).
    No retrieval-style grounding required. Profile background is intentionally
    *not* injected on this turn — it tempts the model into shoehorning farming
    or location details into casual replies.
    """
    directives = [
        _IDENTITY,
        (
            "This is a conversational turn — a greeting, thanks, confirmation, "
            "small talk, or a casual statement, not a research question. "
            "Reply briefly and naturally, matching the user's register: warm and "
            "playful when the message is casual, concise and direct when it is. "
            "Do not preach about your purpose. Do not bring up farming, the user's "
            "region, the user's crops, or EU-FarmBook unless the user actually "
            "raises one of those topics."
        ),
        (
            "If the user is asking about a next step but the subject is unclear, "
            "ask one short clarifying question instead of guessing."
        ),
        _LANGUAGE_RULE,
        _FOLLOWUP_RULE,
    ]
    system_text = "\n\n".join(directives)
    # Deliberately ignore user_profile_context on conversational turns.
    return _assemble_messages(system_text, history_messages, question, user_profile_context=None)


def build_capabilities_messages(
    question: str,
    history_messages: Optional[list[dict]] = None,
    user_profile_context: Optional[str] = None,
) -> list[dict]:
    """The user is asking what the assistant can do. Answer from product behavior, not retrieval."""
    directives = [
        _IDENTITY,
        (
            "The user is asking about your capabilities. Answer from your intended product behavior, "
            "not from retrieved sources, and do not present capability claims as sourced facts."
        ),
        (
            "You can help with agriculture and EU-FarmBook-related questions, explain concepts, "
            "summarize or discuss uploaded PDF content when available, recap the conversation, "
            "compare options, and guide the user step by step on a farming or project-related topic. "
            "Do not claim external actions you cannot perform; if a capability depends on the user "
            "providing more information or a document, say that clearly."
        ),
        _LANGUAGE_RULE,
        (
            "Keep the answer concise, practical, and organized. "
            "If helpful, end with a short question offering 2-4 concrete next-step options."
        ),
    ]
    system_text = "\n\n".join(directives)
    # Capability answers describe the product, not the user — drop the profile.
    return _assemble_messages(system_text, history_messages, question, user_profile_context=None)


def build_summary_prompt(user_prompt: str, text: str) -> str:
    """One-shot summarization. Keep user's custom prompt authoritative."""
    from app.config import get_settings
    S = get_settings()

    max_text_chars = max(1000, S.MAX_CONTEXT_CHARS - len(user_prompt) - 1000)
    safe_text = text if len(text) <= max_text_chars else text[:max_text_chars]

    return (
        f"{user_prompt.strip()}\n\n"
        f"--- BEGIN TEXT ---\n"
        f"{safe_text}\n"
        f"--- END TEXT ---\n"
    )


def build_title_prompt(question: str, answer: str | None = None) -> str:
    """Small one-shot prompt for chat title generation."""
    base = (
        "Generate a short, specific chat title using 2-3 words only. "
        "No punctuation, no quotes, no emojis, no trailing period. Output ONLY the title text.\n\n"
        f"User's question: {question.strip()}\n"
    )
    if answer:
        base += f"Assistant's response: {answer.strip()[:200]}...\n"
    base += "\nTitle:"
    return base
