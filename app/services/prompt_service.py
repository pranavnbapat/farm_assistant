# app/services/prompt_service.py

from typing import Optional


def build_prompt(
    contexts: list[str],
    question: str,
    history: Optional[str] = None,
    user_profile_context: Optional[str] = None
) -> str:
    """
    Build a natural conversation prompt.
    The LLM receives full context and handles the conversation naturally.
    """
    # Label each context block
    labelled = [f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)] if contexts else []
    joined = "\n\n".join(labelled)

    # Build the conversation
    parts = []
    
    # System instruction
    parts.append(
        "You are an agricultural assistant for EU-FarmBook. "
        "Only answer questions related to agriculture, farming, agri-tech, food systems, "
        "or EU-FarmBook project topics. "
        "If the question is outside this scope, do NOT provide the factual answer; "
        "instead politely refuse in 1-2 short sentences and ask the user to ask an agriculture-related question."
    )

    parts.append(
        "Language rule: respond in the same language as the user's latest message, "
        "unless the user explicitly asks for a different language."
    )

    parts.append(
        "Answer the user's actual question directly before adding extra detail. "
        "Use the Previous Conversation section for continuity when the user refers to earlier turns."
    )

    parts.append(
        "When relevant sources are provided, ground your answer in them and add inline citations "
        "using numeric brackets like [1], [2] tied to the source list. "
        "Include citations for factual claims that rely on those sources."
    )

    parts.append(
        "If uploaded PDF context is present in Relevant Sources, do not say you cannot access files. "
        "Treat that PDF content as available context and answer from it."
    )

    parts.append(
        "If the request is ambiguous, under-specified, or depends on missing facts, "
        "ask one specific clarifying question instead of making up assumptions. "
        "If the available sources do not support a claim, say that plainly."
    )

    parts.append(
        "If helpful, end with one short follow-up question. "
        "Do not force a follow-up question for simple greetings, thanks, confirmations, or closings."
    )
    
    # Available information
    if user_profile_context:
        parts.append(f"\nUser Profile:\n{user_profile_context}")
    
    if joined:
        parts.append(f"\nRelevant Sources:\n{joined}")
    
    # Conversation history
    if history:
        parts.append(f"\nPrevious Conversation:\n{history}")
    
    # Current question
    parts.append(f"\nUser: {question}")
    parts.append("\nAssistant:")
    
    return "\n".join(parts)


def build_history_only_prompt(
    question: str,
    history: Optional[str] = None,
    user_profile_context: Optional[str] = None,
) -> str:
    """
    Build a prompt that answers strictly from conversation history.
    Used for recap/meta turns so the assistant does not invent topic summaries
    from external retrieval when the user is asking about the conversation itself.
    """
    parts = []

    parts.append(
        "You are an agricultural assistant for EU-FarmBook. "
        "The user is asking about the conversation itself. "
        "Answer strictly from the Previous Conversation section below."
    )

    parts.append(
        "Do not use outside knowledge, retrieved sources, or general agricultural background "
        "to fill gaps. If the previous conversation does not contain the requested information, "
        "say that plainly."
    )

    parts.append(
        "Language rule: respond in the same language as the user's latest message, "
        "unless the user explicitly asks for a different language."
    )

    parts.append(
        "Be concrete, brief, and faithful to what was actually said. "
        "If helpful, end with one short follow-up question. "
        "Do not force a follow-up question when a direct recap is enough."
    )

    if user_profile_context:
        parts.append(f"\nUser Profile:\n{user_profile_context}")

    if history:
        parts.append(f"\nPrevious Conversation:\n{history}")
    else:
        parts.append(
            "\nPrevious Conversation:\n"
            "No earlier conversation is available in the current session context."
        )

    parts.append(f"\nUser: {question}")
    parts.append("\nAssistant:")

    return "\n".join(parts)


def build_conversation_only_prompt(
    question: str,
    history: Optional[str] = None,
    user_profile_context: Optional[str] = None,
) -> str:
    """
    Build a prompt for conversational turns that should not trigger retrieval.
    This covers greetings, thanks, acknowledgements, small-talk-like control turns,
    and lightweight clarification turns that depend mainly on the current chat.
    """
    parts = []

    parts.append(
        "You are an agricultural assistant for EU-FarmBook. "
        "Handle this as a conversational turn, not a retrieval-heavy research answer."
    )

    parts.append(
        "Use the user's latest message and the Previous Conversation section for context. "
        "Do not invent external facts or pretend to have evidence that is not present."
    )

    parts.append(
        "If the user is greeting, thanking, confirming, or asking a lightweight conversational question, "
        "reply naturally and concisely."
    )

    parts.append(
        "If the user is asking for the next step but the subject is unclear, "
        "ask one specific clarifying question instead of guessing."
    )

    parts.append(
        "Language rule: respond in the same language as the user's latest message, "
        "unless the user explicitly asks for a different language."
    )

    parts.append(
        "If helpful, end with one short follow-up question. "
        "Do not force a follow-up question for simple greetings, thanks, confirmations, or closings."
    )

    if user_profile_context:
        parts.append(f"\nUser Profile:\n{user_profile_context}")

    if history:
        parts.append(f"\nPrevious Conversation:\n{history}")

    parts.append(f"\nUser: {question}")
    parts.append("\nAssistant:")

    return "\n".join(parts)


def build_capabilities_prompt(
    question: str,
    history: Optional[str] = None,
    user_profile_context: Optional[str] = None,
) -> str:
    """
    Build a prompt for questions about what the assistant can do.
    This should answer from the product's intended behavior rather than from retrieved sources.
    """
    parts = []

    parts.append(
        "You are an agricultural assistant for EU-FarmBook. "
        "The user is asking about your capabilities, how you can help, or what kinds of tasks you handle."
    )

    parts.append(
        "Answer from your intended product behavior, not from retrieved sources. "
        "Do not cite documents or present capability claims as if they were sourced facts."
    )

    parts.append(
        "Describe capabilities concretely and realistically. "
        "You can help with agriculture and EU-FarmBook-related questions, explain concepts, "
        "summarize or discuss uploaded PDF content when available, recap the conversation, "
        "compare options, and guide the user step by step on a farming or project-related topic."
    )

    parts.append(
        "Do not claim external actions you cannot perform. "
        "If a capability depends on the user providing more information or a document, say that clearly."
    )

    parts.append(
        "Language rule: respond in the same language as the user's latest message, "
        "unless the user explicitly asks for a different language."
    )

    parts.append(
        "Keep the answer concise, practical, and organized. "
        "If helpful, end with a short question offering 2-4 concrete next-step options."
    )

    if user_profile_context:
        parts.append(f"\nUser Profile:\n{user_profile_context}")

    if history:
        parts.append(f"\nPrevious Conversation:\n{history}")

    parts.append(f"\nUser: {question}")
    parts.append("\nAssistant:")

    return "\n".join(parts)


def build_summary_prompt(user_prompt: str, text: str) -> str:
    """Keep user's custom prompt authoritative."""
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
    """Build a small prompt for chat title generation."""
    base = (
        "Generate a short, specific chat title using 2-3 words only. "
        "No punctuation, no quotes, no emojis, no trailing period. Output ONLY the title text.\n\n"
        f"User's question: {question.strip()}\n"
    )
    if answer:
        base += f"Assistant's response: {answer.strip()[:200]}...\n"
    base += "\nTitle:"
    return base
