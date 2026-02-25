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
        "You are a helpful agricultural assistant for EU-FarmBook. "
        "You assist farmers and agricultural professionals with their questions."
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
        "Generate a short, specific title for this chat (2-3 words max). "
        "No quotes, no trailing period. Output ONLY the title.\n\n"
        f"User's question: {question.strip()}\n"
    )
    if answer:
        base += f"Assistant's response: {answer.strip()[:200]}...\n"
    base += "\nTitle:"
    return base
