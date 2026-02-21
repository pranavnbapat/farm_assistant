# app/services/prompt_service.py

import re
from typing import Optional

# Agriculture domain keywords for quick validation
AGRICULTURE_KEYWORDS = [
    # General farming
    'farm', 'farming', 'agriculture', 'agricultural', 'crop', 'crops', 'harvest',
    'field', 'soil', 'land', 'cultivation', 'cultivate', 'grow', 'growing',
    
    # Specific crops
    'wheat', 'barley', 'corn', 'maize', 'rice', 'oats', 'rye', 'potato', 'potatoes',
    'tomato', 'tomatoes', 'onion', 'onions', 'carrot', 'carrots', 'lettuce',
    'cabbage', 'cauliflower', 'apple', 'apples', 'pear', 'pears', 'grape', 'grapes',
    'olive', 'olives', 'sunflower', 'rapeseed', 'canola', 'soybean', 'soybeans',
    'sugar beet', 'sugarbeet', 'cotton', 'vegetable', 'vegetables', 'fruit', 'fruits',
    
    # Livestock
    'cow', 'cows', 'cattle', 'dairy', 'beef', 'milk', 'pig', 'pigs', 'pork', 'swine',
    'sheep', 'lamb', 'goat', 'goats', 'chicken', 'chickens', 'poultry', 'egg', 'eggs',
    'horse', 'horses', 'livestock', 'animal', 'animals', 'feed', 'feeding',
    
    # Farm management
    'organic', 'conventional', 'pesticide', 'fertilizer', 'irrigation', 'drainage',
    'machinery', 'tractor', 'equipment', 'greenhouse', 'barn', 'silo', 'storage',
    
    # Problems
    'pest', 'disease', 'fungus', 'weed', 'aphid', 'insect', 'drought', 'frost',
    'blight', 'rot', 'mildew', 'rust', 'problem', 'issue', 'trouble',
    
    # Topics
    'soil health', 'pH', 'nutrient', 'compost', 'manure', 'sustainable', 'climate',
    'weather', 'season', 'yield', 'production', 'market', 'price', 'subsidy',
    'EU', 'regulation', 'certification', 'CAP', 'biodiversity', 'carbon',
]

# Non-agriculture patterns to explicitly reject
NON_AGRICULTURE_PATTERNS = [
    r'\b(programming|code|software|developer|python|javascript|java|css|html)\b',
    r'\b(stock|crypto|bitcoin|investment|trading|forex)\b',
    r'\b(movie|film|actor|celebrity|sports team|football|basketball|baseball)\b',
    r'\b(politics|election|president|minister|government policy|war)\b',
    r'\b(video game|gaming|console|playstation|xbox|nintendo)\b',
    r'\b(recipe|cooking|chef|restaurant|food delivery)\b',
    r'\b(travel|vacation|hotel|flight|booking|tourism)\b',
    r'\b(medical advice|doctor|hospital|medicine|treatment|therapy)\b',
]


def is_agriculture_related(question: str) -> tuple[bool, str]:
    """
    Check if a question is related to agriculture.
    Returns (is_related, reason)
    """
    question_lower = question.lower()
    
    # Check for explicit non-agriculture topics
    for pattern in NON_AGRICULTURE_PATTERNS:
        if re.search(pattern, question_lower):
            return False, "question_contains_non_agriculture_topic"
    
    # Check for agriculture keywords
    for keyword in AGRICULTURE_KEYWORDS:
        if keyword in question_lower:
            return True, "agriculture_keyword_match"
    
    # Check for question patterns that might be agricultural
    # (e.g., "how do I...", "what is..." without context)
    # These are ambiguous, we'll let the LLM decide with the prompt
    
    return False, "no_agriculture_keywords_found"


def build_prompt(
    contexts: list[str],
    question: str,
    history: Optional[str] = None,
    user_profile_context: Optional[str] = None
) -> str:
    # Label each context block so the model knows what [1], [2], ... refer to
    labelled = [f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)]
    joined = "\n\n".join(labelled)

    history_block = ""
    if history:
        history_block = (
            "Conversation so far (most recent first):\n"
            f"{history}\n\n"
        )

    # Detect the language of the user's question for the prompt
    # (We don't actually detect it programmatically, we instruct the model to handle it)

    # Build user profile section if available
    profile_block = ""
    if user_profile_context:
        profile_block = f"\n{user_profile_context}\n\n"

    return (
        "System: You are a knowledgeable agricultural assistant who writes in a natural, precise, and concise style—"
        "similar to ChatGPT-5. You are STRICTLY LIMITED to agriculture, farming, and rural development topics.\n\n"
        "=== DOMAIN RESTRICTION ===\n"
        "You ONLY answer questions related to:\n"
        "- Agriculture and farming (crops, livestock, soil, machinery)\n"
        "- Agricultural policy and regulations (EU CAP, subsidies)\n"
        "- Rural development and sustainability\n"
        "- Climate and weather impact on agriculture\n"
        "- Agricultural markets and economics\n"
        "\n"
        "If the user's question is NOT related to agriculture:\n"
        "- Politely decline to answer\n"
        "- Explain that you are specialized in agricultural topics\n"
        "- Offer to help with farming-related questions instead\n"
        "- Example: 'I'm specialized in agricultural topics. I'd be happy to help with questions about farming, crops, livestock, or agricultural policy instead.'\n"
        "=== END DOMAIN RESTRICTION ===\n\n"
        "=== CRITICAL LANGUAGE RULE ===\n"
        "You MUST respond ENTIRELY in the SAME LANGUAGE as the user's question below.\n"
        "- If the user asks in English → answer in English\n"
        "- If the user asks in Dutch → answer in Dutch\n"
        "- If the user asks in French → answer in French\n"
        "- If the user asks in German → answer in German\n"
        "- If the user asks in Spanish → answer in Spanish\n"
        "And so on for ANY language.\n"
        "Even if the sources below are in a different language, TRANSLATE the information\n"
        "and respond ONLY in the user's question language.\n"
        "=== END LANGUAGE RULE ===\n\n"
        "=== PERSONALIZATION RULE ===\n"
        "Tailor your response based on the user's profile below.\n"
        "- For beginners: Explain concepts simply, avoid jargon, be encouraging\n"
        "- For experts: Be concise, technical, focus on specifics\n"
        "- For organic farms: Suggest organic-approved methods only\n"
        "- For specific regions: Consider local climate/regulations\n"
        "- If user mentioned crops/issues before: Reference that knowledge\n"
        "=== END PERSONALIZATION RULE ===\n\n"
        "Instructions:\n"
        "- Read ONLY the information provided in the sources below.\n"
        "- Translate and adapt the content to answer in the user's language.\n"
        "- Take into account the earlier conversation (if provided) so your answer stays consistent with past replies.\n"
        "- Keep your answer short and well-structured (usually 2–4 short paragraphs, or a few bullet points if useful).\n"
        "- Avoid artificial numbering or section labels like '1)', '2)', etc.\n"
        "- Be concise and precise: avoid repetition, filler, or generalities.\n"
        "- Cite sources inline using numeric brackets like [1], [2] immediately after the relevant sentence.\n"
        "- Use only numbers that correspond to the provided sources. Never invent citations.\n"
        "- If the sources are insufficient, briefly say so.\n"
        "- Unless the user clearly asks for a single final answer, end with a short follow-up question that suggests a sensible next step.\n\n"
        "Formatting style:\n"
        "- Start with a brief overview paragraph introducing the topic.\n"
        "- Present key facts naturally in flowing text or short bullet points.\n"
        "- Conclude with one brief summarising sentence (no new claims), followed by a brief follow-up question.\n\n"
        "Citation example:\n"
        "Precision feeding and robotic milking have improved efficiency in Dutch dairy farms. [2][3]\n\n"
        f"{profile_block}"
        f"{history_block}"
        f"Sources:\n{joined}\n\n"
        f"User question: {question}\n\n"
        "Assistant (MUST be in the SAME LANGUAGE as the user's question above):"
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

def build_generic_prompt(
    question: str,
    history: Optional[str] = None,
    user_profile_context: Optional[str] = None
) -> str:
    """
    Generic prompt for non-RAG, chit-chat or general Q&A.
    STRICTLY limited to agriculture domain.
    """
    history_block = ""
    if history:
        history_block = (
            "Conversation so far (most recent first):\n"
            f"{history}\n\n"
        )
    
    # Build user profile section if available
    profile_block = ""
    if user_profile_context:
        profile_block = f"{user_profile_context}\n\n"

    return (
        "System: You are a helpful agricultural assistant. You are STRICTLY LIMITED to agriculture, farming, and rural development topics.\n\n"
        "=== DOMAIN RESTRICTION ===\n"
        "You ONLY answer questions related to:\n"
        "- Agriculture and farming (crops, livestock, soil, machinery)\n"
        "- Agricultural policy and regulations (EU CAP, subsidies)\n"
        "- Rural development and sustainability\n"
        "- Climate and weather impact on agriculture\n"
        "- Agricultural markets and economics\n"
        "\n"
        "If the user's question is NOT related to agriculture:\n"
        "- Politely decline to answer\n"
        "- Explain that you are specialized in agricultural topics\n"
        "- Offer to help with farming-related questions instead\n"
        "- DO NOT answer questions about: programming, sports, entertainment, politics, gaming, cooking recipes, travel, medical advice, or other non-agricultural topics\n"
        "=== END DOMAIN RESTRICTION ===\n\n"
        "=== CRITICAL LANGUAGE RULE ===\n"
        "You MUST respond ENTIRELY in the SAME LANGUAGE as the user's question below.\n"
        "- If the user asks in English → answer in English\n"
        "- If the user asks in Dutch → answer in Dutch\n"
        "- If the user asks in French → answer in French\n"
        "- If the user asks in German → answer in German\n"
        "- If the user asks in Spanish → answer in Spanish\n"
        "And so on for ANY language.\n"
        "=== END LANGUAGE RULE ===\n\n"
        "=== PERSONALIZATION RULE ===\n"
        "Use the user profile below to personalize your response.\n"
        "- Match the user's expertise level and communication style\n"
        "- Consider their farm type, region, and crops when relevant\n"
        "=== END PERSONALIZATION RULE ===\n\n"
        "Instructions:\n"
        "- Respond directly to the user's question (if it's agriculture-related).\n"
        "- Take into account the earlier conversation (if provided) so your answer stays consistent with past replies.\n"
        "- Keep it short, well-structured, and avoid filler.\n"
        "- If the question is casual greetings, reply naturally and briefly.\n"
        "- Do NOT invent sources or citations.\n"
        "- Unless the user clearly asks you to stop, end with a short follow-up question that invites the user to continue.\n\n"
        f"{profile_block}"
        f"{history_block}"
        f"User question: {question}\n\n"
        "Assistant (MUST be in the SAME LANGUAGE as the user's question above):"
    )


def build_title_prompt(question: str, answer: str | None = None) -> str:
    """
    Build a very small prompt to turn the user's first question
    (and optionally the assistant answer) into a short chat title.
    """
    base = (
        "You are generating a short title for a chat between a farmer and an assistant.\n"
        "- Use at most 6–8 words.\n"
        "- No quotes, no trailing full stop.\n"
        "- Make it specific but concise.\n"
        "- Output ONLY the title text, nothing else.\n\n"
        f"User's question:\n{question.strip()}\n"
    )
    if answer:
        base += f"\nAssistant's answer (for extra context):\n{answer.strip()}\n"
    base += "\nTitle:"
    return base

