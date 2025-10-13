# app/services/prompt_service.py

def build_prompt(contexts: list[str], question: str) -> str:
    joined = "\n\n".join(contexts)
    return (
        "System: You are a knowledgeable agricultural assistant. "
        "Read ONLY the sources provided below and answer the user's question clearly and concisely. "
        "Use complete sentences and, where helpful, brief reasoning. "
        "Cite sources inline with bracketed labels (e.g., [S1], [S2]) at the end of the sentence "
        "that uses information from that source. Only use S-labels that actually appear in the sources. "
        "Do NOT invent facts or citations. If the sources are insufficient, say you don’t know.\n\n"
        f"Sources:\n{joined}\n\n"
        f"User question: {question}\n\n"
        "Assistant:"
    )
