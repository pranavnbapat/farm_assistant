# app/routers/ask.py

import asyncio
import base64
import hashlib
import json
import logging
import re as _re
import time

import httpx

from typing import Optional, List

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.clients.vllm_client import generate_once, stream_generate
from app.clients.opensearch_client import os_async_client, os_headers, os_auth
from app.config import get_settings
from app.services.chat_history import (
    load_chat_state,
    format_history,
    merge_messages,
    save_chat_state,
    CHAT_BACKEND_URL,
)
from app.services.context_service import (
    build_context_and_sources,
    estimate_retrieval_quality,
    estimate_semantic_quality,
    filter_items_by_min_score,
)
from app.services.image_service import (
    get_images_for_user,
    build_image_contexts,
    ensure_image_processed,
    upsert_image_attachment_to_backend,
    images_from_attachment_records,
)
from app.services.pdf_service import (
    get_docs_for_user,
    build_pdf_contexts,
    ensure_pdf_processed,
    upsert_attachment_to_backend,
    fetch_session_attachments_from_backend,
    docs_from_attachment_records,
)
from app.services.prompt_service import (
    build_capabilities_messages,
    build_platform_operation_messages,
    build_clarification_messages,
    build_general_knowledge_messages,
    build_messages,
    build_conversation_only_messages,
    build_history_only_messages,
    build_off_topic_messages,
    build_summary_prompt,
    build_title_prompt,
)
from app.services.scope_contract import (
    ROUTER_MODE_PROMPT_TEXT,
    ScopeRouteDecision,
    TurnMode,
    decision_for_mode,
    file_analysis_decision,
)
from app.services.search_service import build_search_payload, collect_os_items
from app.services.web_search_service import web_search_and_build_contexts
from app.utils.language_utils import detect_language, detect_language_confident, get_language_name
from app.services.user_profile_service import UserProfileService
from app.schemas import AskIn, ChatMessageStreamIn, ExportIntentIn, ExportIntentOut, FollowUpsIn, FollowUpsOut, SummariseIn, SummariseOut

logger = logging.getLogger("farm-assistant.router")
S = get_settings()
router = APIRouter()

_EMPTY_CITATION_RE = _re.compile(r"[ \t]*(?:\[(?:[ \t]|\[[ \t]*\])*\][ \t]*)+(?:-[ \t]*)?")

# A "lone marker" line: only whitespace (incl. non-breaking / zero-width spaces)
# around at most one stray bullet or dot. Models (notably Qwen and EuroLLM) emit
# these as empty list items between sections, which the Markdown renderer draws
# as orphan bullets. Built from code points so the source stays pure ASCII.
# Provider-agnostic - applied to every answer regardless of the LLM.
_LONE_MARKER_WS = "".join(chr(c) for c in (0x20, 0x09, 0xA0, 0x200B, 0x200C, 0x200D, 0xFEFF))
_LONE_MARKER_BULLETS = "".join(
    chr(c) for c in (0x2D, 0x2A, 0x2B, 0x2E, 0x2022, 0x00B7, 0x2023, 0x2043, 0x2219, 0x25CF, 0x25CB, 0x25E6)
)
_LONE_MARKER_RE = _re.compile(
    rf"(?m)^[{_re.escape(_LONE_MARKER_WS)}]*"
    rf"[{_re.escape(_LONE_MARKER_BULLETS)}]?"
    rf"[{_re.escape(_LONE_MARKER_WS)}]*$"
)

def _get_answer_language(question: str) -> str:
    """Resolve a deterministic language name from the latest message only."""
    language_code = detect_language_confident(question) or detect_language(question)
    language_name = get_language_name(language_code)
    return "English" if language_name == "Unknown" else language_name


async def _resolve_answer_language(question: str) -> str:
    """Detect the latest message language without consulting conversation history."""
    confident_code = detect_language_confident(question)
    if confident_code:
        return get_language_name(confident_code)

    prompt = (
        "Identify the language of the user message below. Consider only this message; "
        "do not infer language from any conversation history or user preference.\n"
        "Return JSON only as {\"language_code\":\"xx\"}, using a two-letter ISO 639-1 code.\n\n"
        f"User message:\n{question}"
    )
    try:
        raw = await asyncio.wait_for(
            generate_once(prompt, temperature=0.0, max_tokens=12),
            timeout=1.5,
        )
        match = _re.search(r"\{.*?\}", raw or "", flags=_re.DOTALL)
        if match:
            code = (json.loads(match.group(0)).get("language_code") or "").strip().lower()
            language_name = get_language_name(code)
            if language_name != "Unknown":
                return language_name
    except Exception:
        pass

    return _get_answer_language(question)


def _platform_operation_static_answer(answer_language: str) -> str:
    answers = {
        "english": (
            "I cannot confirm that from the available EU-FarmBook material. "
            "I should not assume that public upload access exists. "
            "In this chat, you can upload files for analysis, but uploading or publishing materials "
            "to EU-FarmBook itself would need to be confirmed through the official EU-FarmBook team or documentation."
        ),
        "dutch": (
            "Ik kan dat niet bevestigen op basis van het beschikbare EU-FarmBook-materiaal. "
            "Ik mag niet aannemen dat publieke uploadtoegang bestaat. "
            "In deze chat kun je bestanden uploaden voor analyse, maar materiaal uploaden of publiceren "
            "naar EU-FarmBook zelf moet worden bevestigd via het officiele EU-FarmBook-team of de documentatie."
        ),
        "french": (
            "Je ne peux pas le confirmer a partir des documents EU-FarmBook disponibles. "
            "Je ne dois pas supposer qu un acces public au televersement existe. "
            "Dans cette conversation, vous pouvez televerser des fichiers pour analyse, mais le televersement "
            "ou la publication de contenus sur EU-FarmBook doit etre confirme par l equipe ou la documentation officielle d EU-FarmBook."
        ),
        "german": (
            "Ich kann das anhand des verfuegbaren EU-FarmBook-Materials nicht bestaetigen. "
            "Ich sollte nicht annehmen, dass ein oeffentlicher Upload-Zugang existiert. "
            "In diesem Chat koennen Sie Dateien zur Analyse hochladen, aber das Hochladen oder Veroeffentlichen "
            "von Materialien in EU-FarmBook selbst muss ueber das offizielle EU-FarmBook-Team oder die Dokumentation bestaetigt werden."
        ),
        "spanish": (
            "No puedo confirmarlo con el material disponible de EU-FarmBook. "
            "No debo asumir que exista acceso publico para subir contenido. "
            "En este chat puede subir archivos para analizarlos, pero subir o publicar materiales "
            "en EU-FarmBook debe confirmarse con el equipo oficial o la documentacion oficial de EU-FarmBook."
        ),
        "portuguese": (
            "Nao posso confirmar isso com o material disponivel do EU-FarmBook. "
            "Nao devo presumir que exista acesso publico para upload. "
            "Neste chat, voce pode enviar arquivos para analise, mas enviar ou publicar materiais "
            "no proprio EU-FarmBook deve ser confirmado pela equipe oficial ou pela documentacao oficial do EU-FarmBook."
        ),
        "italian": (
            "Non posso confermarlo dal materiale EU-FarmBook disponibile. "
            "Non devo presumere che esista un accesso pubblico per il caricamento. "
            "In questa chat puoi caricare file per analisi, ma caricare o pubblicare materiali "
            "su EU-FarmBook deve essere confermato dal team ufficiale o dalla documentazione ufficiale di EU-FarmBook."
        ),
    }
    return answers.get((answer_language or "").strip().lower(), answers["english"])

def _extract_user_uuid_from_token(auth_token: str) -> Optional[str]:
    """Extract user UUID from JWT token string."""
    if not auth_token.startswith("Bearer "):
        return None
    
    token = auth_token[7:]
    
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        
        decoded = base64.urlsafe_b64decode(payload)
        token_data = json.loads(decoded)
        
        user_id = token_data.get("uuid") or token_data.get("user_id") or token_data.get("sub")
        return str(user_id) if user_id else None
    except Exception:
        return None


def _normalize_model(model: Optional[str]) -> str:
    """Normalize model parameter to use the configured vLLM model."""
    if model and ":" in model:
        logger.warning(f"Received Ollama-style model name '{model}', using VLLM_MODEL '{S.VLLM_MODEL}' instead")
        return S.VLLM_MODEL
    return model or S.VLLM_MODEL


def _sanitize_generated_markdown(text: str) -> str:
    """
    Light cleanup only. Frontend renders real Markdown (tables, headings, code blocks),
    so we no longer rewrite tables to bullets or strip <br>; just normalize line endings,
    remove empty citation placeholders, and collapse excessive blank lines.
    """
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _EMPTY_CITATION_RE.sub(" ", cleaned)
    cleaned = _LONE_MARKER_RE.sub("", cleaned)
    cleaned = _re.sub(r" +([.,;:!?])", r"\1", cleaned)
    cleaned = _re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_cited_numbers(text: str) -> set[int]:
    cited_nums: set[int] = set()

    for match in _re.finditer(r"\[\s*(\d+)\s*\]", text):
        cited_nums.add(int(match.group(1)))

    for match in _re.finditer(r"\[\s*([\d\s,–-]+)\s*\]", text):
        for token in _re.split(r"[,\s–-]+", match.group(1)):
            if token.isdigit():
                cited_nums.add(int(token))

    return cited_nums


def _strip_orphan_citations(text: str, valid_source_numbers: set[int]) -> str:
    if not valid_source_numbers:
        cleaned = _re.sub(
            r"(?m)^[ \t]*\[\s*[\d\s,–-]+\s*\][^\n]*(?:\n|$)",
            "",
            text,
        )
        cleaned = _re.sub(r"\s*\[\s*[\d\s,–-]+\s*\]", "", cleaned)
        return _re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    def _replace(match):
        tokens = [
            int(token)
            for token in _re.split(r"[,\s–-]+", match.group(1))
            if token.isdigit()
        ]
        valid_tokens = [token for token in tokens if token in valid_source_numbers]

        if not valid_tokens:
            return ""

        return "[" + ", ".join(str(token) for token in valid_tokens) + "]"

    cleaned = _re.sub(r"\[\s*([\d\s,–-]+)\s*\]", _replace, text)
    cleaned = _EMPTY_CITATION_RE.sub(" ", cleaned)
    cleaned = _LONE_MARKER_RE.sub("", cleaned)
    cleaned = _re.sub(r" +([.,;:!?])", r"\1", cleaned)
    cleaned = _re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _cache_params(_temp, _max, _model):
    return {
        "temperature": _temp,
        "max_tokens": _max,
        "model": _normalize_model(_model),
        "num_ctx": S.NUM_CTX,
        "top_p": S.TOP_P,
    }


def _history_scope(history_text: str) -> str:
    """Turn history into a short, stable hash for cache keys."""
    if not history_text:
        return ""
    return hashlib.sha256(history_text.encode("utf-8")).hexdigest()[:16]


async def _decide_turn_strategy(question: str, history_text: str) -> TurnMode:
    """
    Use the model to decide whether the current turn should be answered strictly
    from conversation history, handled as a conversational turn, answered as a capability question,
    or follow the normal retrieval flow.
    This keeps the routing behavior dynamic instead of hardcoding keyword rules.
    """
    prompt = (
        "You are routing a chat turn for an agricultural assistant for the EU-FarmBook platform.\n"
        "Choose one mode and return JSON only.\n\n"
        f"{ROUTER_MODE_PROMPT_TEXT}\n"
        f"User message:\n{question}\n\n"
        "Previous Conversation:\n"
        f"{history_text or 'No earlier conversation is available.'}\n\n"
        'Return exactly: {"mode":"off_topic"} or {"mode":"history_only"} or {"mode":"conversation_only"} or '
        '{"mode":"assistant_capabilities"} or {"mode":"platform_operation"} or {"mode":"general_knowledge"} or {"mode":"normal"}'
    )
    try:
        raw = await asyncio.wait_for(
            generate_once(prompt, temperature=0.0, max_tokens=20),
            timeout=1.5,
        )
        match = _re.search(r"\{.*?\}", raw or "", flags=_re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            mode = (data.get("mode") or "").strip().lower()
            if mode in {"off_topic", "history_only", "conversation_only", "assistant_capabilities", "platform_operation", "general_knowledge", "normal"}:
                return mode
    except Exception:
        pass
    return "normal"


async def _resolve_turn_context(
    question: str,
    history_text: str,
    last_assistant_question: str = "",
    followup_hint: str = "",
) -> dict:
    """
    Use the model to turn terse or elliptical user replies into a clean,
    standalone interpretation for the next assistant turn.
    This avoids brittle affirmation heuristics and prevents accidental
    continuation from trailing fragments of the prior answer.
    """
    if _is_affirmative_followup(question) and (last_assistant_question or followup_hint):
        accepted_offer = (last_assistant_question or followup_hint or "").strip()
        resolved = f"The user accepted the assistant's previous offer or question: {accepted_offer}"
        return {
            "resolved_user_message": resolved,
            "assistant_instruction": (
                f"The user answered affirmatively to this previous assistant offer or question: {accepted_offer}. "
                "Fulfill that accepted request using only the prior conversation and any available chat context."
            ),
        }

    prompt = (
        "You are interpreting the user's latest turn for an agricultural assistant.\n"
        "Return JSON only.\n\n"
        "Your task:\n"
        "1. Decide whether the latest user message is already a standalone request.\n"
        "2. If it is short, elliptical, or mainly confirms the assistant's previous question, "
        "rewrite it into a clear standalone intent grounded only in the conversation.\n"
        "3. Provide a prompt-ready instruction that helps the assistant answer the turn cleanly.\n\n"
        "Rules:\n"
        "- Do not invent facts not present in the conversation.\n"
        "- Do not continue or quote trailing fragments from the previous assistant answer.\n"
        "- If the user is agreeing to proceed after a prior assistant question, make that explicit.\n"
        "- Keep the resolved text concise and faithful.\n\n"
        f"Latest user message:\n{question}\n\n"
        f"Last assistant question:\n{last_assistant_question or 'None'}\n\n"
        f"Follow-up hint:\n{followup_hint or 'None'}\n\n"
        "Previous Conversation:\n"
        f"{history_text or 'No earlier conversation is available.'}\n\n"
        "Return exactly this JSON shape:\n"
        '{"resolved_user_message":"...","assistant_instruction":"..."}'
    )
    try:
        raw = await asyncio.wait_for(
            generate_once(prompt, temperature=0.0, max_tokens=180),
            timeout=2.0,
        )
        match = _re.search(r"\{.*\}", raw or "", flags=_re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            resolved = (data.get("resolved_user_message") or "").strip()
            instruction = (data.get("assistant_instruction") or "").strip()
            if resolved or instruction:
                return {
                    "resolved_user_message": resolved or question,
                    "assistant_instruction": instruction or resolved or question,
                }
    except Exception:
        pass
    return {
        "resolved_user_message": question,
        "assistant_instruction": question,
    }


SYSTEM_PROMPT_TAG = "farm-assistant-v2"


def _estimate_tokens(text: str) -> int:
    # Fast heuristic: ~4 chars per token for mixed English-like text.
    return max(1, len(text or "") // 4)


def _short_title_2_3_words(raw: str) -> str:
    cleaned = (raw or "").strip().strip(" \"'")
    if not cleaned:
        return ""
    words = [w for w in cleaned.split() if w]
    if not words:
        return ""
    # Keep at most 3 words; if model returns 1 word, keep it as-is.
    return " ".join(words[:3])[:120]


_ASSISTANT_OFFER_PATTERNS = (
    _re.compile(r"\b(?:let me know if|tell me if)\b", _re.IGNORECASE),
    _re.compile(r"\b(?:would you like|do you want|should i|shall i)\b", _re.IGNORECASE),
    _re.compile(r"\bi can\b.{0,120}\b(?:provide|prepare|create|make|turn|convert|summarize|summarise|translate|format|export)\b", _re.IGNORECASE),
)

_AFFIRMATIVE_FOLLOWUP_TERMS = {
    "yes", "yes please", "please", "sure", "sure please", "ok", "okay",
    "okay please", "yeah", "yep", "yup", "go ahead", "do it", "please do",
    "that would be great", "sounds good", "let's do it", "proceed",
}


def _is_affirmative_followup(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    q = _re.sub(r"[.!?]+$", "", q).strip()
    return q in _AFFIRMATIVE_FOLLOWUP_TERMS


def _extract_last_assistant_question(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if (msg.get("role") or "").lower() != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Take the last explicit question first.
        parts = _re.split(r"(?<=[\?\!\.])\s+", content)
        for p in reversed(parts):
            p = p.strip()
            if p.endswith("?"):
                return p[:300]
        # Some assistant continuations are phrased as offers, not questions:
        # "Let me know if you'd like...". Short affirmative replies should
        # inherit these instead of becoming generic conversation.
        for p in reversed(parts):
            p = p.strip()
            if p and any(pattern.search(p) for pattern in _ASSISTANT_OFFER_PATTERNS):
                return p[:300]
    return ""


def _is_file_handoff_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    canonical = {
        "here you go",
        "see this file",
        "see this image",
        "check this file",
        "check this image",
        "this file",
        "this image",
        "look at this",
        "use this file",
        "read this",
    }
    return t in canonical


def _mentions_file_or_document(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keys = {
        "file", "document", "pdf", "attachment", "this doc", "this file",
        "image", "photo", "picture", "screenshot", "this image", "this photo",
    }
    return any(k in t for k in keys)


def _attachment_summary_prompt(has_images: bool, has_pdfs: bool, count: int) -> str:
    if has_images and not has_pdfs:
        return "Please summarize the uploaded image." if count == 1 else "Please summarize the uploaded images."
    if has_pdfs and not has_images:
        return "Please summarize the uploaded PDF." if count == 1 else "Please summarize the uploaded PDFs."
    return "Please summarize the uploaded attachments."


def _should_skip_query_normalization(text: str) -> bool:
    # Avoid accidental translation for non-ASCII queries (e.g., Greek).
    return any(ord(ch) > 127 for ch in (text or ""))


def _looks_standalone(user_q: str) -> bool:
    """
    Decide whether to skip `_resolve_turn_context`. Uses only structural,
    language-agnostic signals (character count + whitespace-separated word
    count) so the optimization applies equally to any language. Short or
    sparse messages still go through resolve, where the LLM uses prior
    conversation context to expand them.
    """
    q = (user_q or "").strip()
    if len(q) < 40:
        return False
    words = [w for w in q.split() if w]
    return len(words) >= 7


def _is_meaningless_prompt(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return True
    if len(q) <= 4 and _re.fullmatch(r"[\W_]+", q):
        return True

    lowered = q.lower()
    trivial = {
        "?", "??", "???", ".", "..", "...",
        "huh", "hm", "hmm", "uh", "um", "ok?", "what?", "excuse me?",
    }
    return lowered in trivial


_AGRI_HINT_TERMS = {
    "agriculture", "agricultural", "farming", "farm", "farmer", "farmers",
    "crop", "crops", "soil", "livestock", "poultry", "cattle", "tractor",
    "machinery", "irrigation", "fertilizer", "fertiliser", "manure", "weed",
    "weeds", "pest", "pests", "crop rotation", "agri", "agri-tech",
    "food system", "food systems", "greenhouse", "horticulture", "aquaculture",
    "forest", "forestry", "sustainability", "climate adaptation", "post-harvest",
    "post harvest", "storage", "processing", "food safety", "value chain",
    "value chains", "farm business", "eufarmbook", "eu-farmbook",
}

_CONSUMER_TECH_OFFTOPIC_TERMS = {
    "iphone", "ipad", "macbook", "ios", "apple watch",
    "samsung", "galaxy", "android", "pixel", "google pixel",
    "oneplus", "xiaomi", "huawei", "oppo", "vivo",
    "smartphone", "phone", "mobile phone", "tablet", "laptop",
    "airpods", "earbuds", "smartwatch",
}

_PROMPT_INJECTION_TERMS = {
    "ignore previous instructions",
    "ignore all previous instructions",
    "you are now a general assistant",
    "you are now",
    "act as",
    "pretend to be",
    "system prompt",
    "developer prompt",
    "jailbreak",
}

_GENERAL_OFFTOPIC_TERMS = {
    "president", "prime minister", "king", "queen", "celebrity", "movie",
    "song", "lyrics", "football", "basketball", "politics", "france",
    "germany", "united states", "usa", "election",
}

_CULINARY_INTENT_TERMS = {
    "recipe", "recipes", "cook", "cooking", "bake", "baking",
    "fry", "frying", "boil", "grill", "roast", "kitchen",
    "homemade", "ingredients",
}

_CULINARY_FOOD_TERMS = {
    "lasagna", "lasagne", "pizza", "pasta", "burger", "cake",
    "cookie", "cookies", "bread", "soup", "salad", "sauce",
    "sandwich", "chips", "fries", "omelette", "pancake", "pancakes",
}

_FOOD_SYSTEM_ALLOWED_TERMS = {
    "farm", "farming", "farmer", "crop", "crops", "processing",
    "post-harvest", "post harvest", "storage", "shelf life", "food safety",
    "haccp", "packaging", "supply chain", "market", "business",
    "value chain", "agri", "agriculture", "agricultural", "production",
}


_HISTORY_TRANSFORM_TERMS = {
    "table", "tabular", "format", "formatted", "reformat", "rewrite",
    "summarize", "summary", "shorten", "simplify", "translate",
    "bullet", "bullets", "list", "compare", "organize", "structure",
}

_HISTORY_REFERENCE_TERMS = {
    "it", "this", "that", "above", "previous", "last", "answer",
    "response", "reply", "content", "all of this", "everything",
}


def _is_history_transform_query(user_q: str) -> bool:
    q = (user_q or "").strip().lower()
    if not q:
        return False
    has_transform = any(term in q for term in _HISTORY_TRANSFORM_TERMS)
    has_reference = any(term in q for term in _HISTORY_REFERENCE_TERMS)
    if has_transform and has_reference:
        return True
    return bool(_re.search(r"\b(?:give|put|make|turn|convert)\s+(?:it|this|that|the answer|the response)\b", q))


def _is_culinary_recipe_query(user_q: str) -> bool:
    q = (user_q or "").strip().lower()
    if not q:
        return False
    if any(term in q for term in _FOOD_SYSTEM_ALLOWED_TERMS):
        return False
    recipe_pattern = _re.search(
        r"\b(?:how\s+(?:do|can)\s+i|how\s+to|show\s+me\s+how\s+to)\s+(?:make|cook|bake|fry|prepare)\b",
        q,
    )
    has_culinary_term = any(term in q for term in _CULINARY_INTENT_TERMS)
    has_food_term = any(term in q for term in _CULINARY_FOOD_TERMS)
    return bool(recipe_pattern or has_food_term or (has_culinary_term and has_food_term))

_PLATFORM_OPERATION_TERMS = {
    "upload", "uploads", "uploaded", "uploading", "publish", "publishing",
    "submit", "submission", "submissions", "contribute", "contribution",
    "register", "registration", "sign up", "signup", "account", "dashboard",
    "portal", "access", "login", "log in", "import", "sync", "synchronise",
    "synchronize", "connect", "share",
}

_PLATFORM_TARGET_TERMS = {
    "euf", "eu-farmbook", "eufarmbook", "platform", "website", "portal",
    "dashboard", "materials", "material", "documents", "records", "data",
}


def _is_platform_operation_query(user_q: str) -> bool:
    q = (user_q or "").strip().lower()
    if not q:
        return False
    if not any(term in q for term in ("euf", "eu-farmbook", "eufarmbook")):
        return False
    has_operation = any(term in q for term in _PLATFORM_OPERATION_TERMS)
    has_target = any(term in q for term in _PLATFORM_TARGET_TERMS)
    return has_operation and has_target


def _hard_route_turn_mode(user_q: str) -> Optional[str]:
    """
    Deterministic guardrail for obviously off-topic standalone queries.
    This runs before the model router so repeated non-agriculture prompts do
    not drift into an answerable mode because of prior agricultural history.
    """
    q = (user_q or "").strip().lower()
    if not q:
        return "clarification_only"
    if _is_meaningless_prompt(q):
        return "clarification_only"
    if _is_platform_operation_query(q):
        return "platform_operation"
    if _is_culinary_recipe_query(q):
        return "off_topic"
    if _mentions_file_or_document(q):
        return None
    has_agri_hint = any(term in q for term in _AGRI_HINT_TERMS)
    if any(term in q for term in _PROMPT_INJECTION_TERMS):
        return "off_topic"
    if any(term in q for term in _CONSUMER_TECH_OFFTOPIC_TERMS):
        return "off_topic"
    if any(term in q for term in _GENERAL_OFFTOPIC_TERMS) and not has_agri_hint:
        return "off_topic"
    if has_agri_hint:
        return None
    return None


def _routing_history_for_query(user_q: str, history_text: str) -> str:
    """
    Keep history out of the lightweight router for clearly standalone queries.
    This reduces drift while preserving history-aware routing for actual follow-ups.
    """
    return "" if _looks_standalone(user_q) else history_text


def _has_offtopic_signal(user_q: str) -> bool:
    """Any term that explicitly marks the query as out-of-scope."""
    q = (user_q or "").strip().lower()
    if not q:
        return False
    if any(term in q for term in _CONSUMER_TECH_OFFTOPIC_TERMS):
        return True
    if any(term in q for term in _GENERAL_OFFTOPIC_TERMS):
        return True
    if any(term in q for term in _PROMPT_INJECTION_TERMS):
        return True
    return False


async def _route_turn_decision(
    *,
    user_q: str,
    prompt_q: str,
    history_text: str,
    has_uploaded_files: bool = False,
) -> ScopeRouteDecision:
    """
    Route the latest turn into a structured product-scope decision.

    The public/runtime API still works with `TurnMode`, but keeping intent,
    allowance, source requirements, and scope inheritance together prevents the
    prompt contract and router behavior from drifting apart.
    """
    q = (user_q or "").strip()
    if not q or _is_meaningless_prompt(q):
        return decision_for_mode(
            "clarification_only",
            reason="Input is empty, punctuation-only, or underspecified.",
        )

    if has_uploaded_files and (_is_file_handoff_query(user_q) or _mentions_file_or_document(user_q)):
        return file_analysis_decision(
            reason="User asks about uploaded PDF/image content before broad refusal checks.",
        )

    forced_mode = _hard_route_turn_mode(user_q)
    if forced_mode:
        return decision_for_mode(
            forced_mode,
            reason="Deterministic router guardrail matched.",
        )

    if history_text and _is_history_transform_query(user_q):
        return decision_for_mode(
            "history_only",
            reason="User asks to transform prior conversation content and inherits its scope.",
        )

    prompt_q_lower = (prompt_q or "").lower()
    resolved_affirmative_offer = (
        "accepted the assistant's previous offer or question" in prompt_q_lower
        or "answered affirmatively to this previous assistant offer or question" in prompt_q_lower
    )
    if history_text and _is_affirmative_followup(user_q) and resolved_affirmative_offer:
        return decision_for_mode(
            "history_only",
            reason="User affirmatively accepted the previous assistant offer or question.",
        )

    strategy_history = _routing_history_for_query(user_q, history_text)
    mode = await _decide_turn_strategy(prompt_q, strategy_history)

    # Safety net: the LLM classifier sometimes flags creative-but-on-topic
    # requests ("draw a farm in ASCII representing wheat or maize") as
    # off_topic because form (ASCII art / drawing) outweighs subject. If the
    # query has at least one agriculture hint and no explicit off-topic
    # markers, treat it as general_knowledge so the assistant answers within
    # scope without manufacturing citations.
    if mode == "off_topic":
        q_lower = q.lower()
        if q_lower and any(term in q_lower for term in _AGRI_HINT_TERMS) and not _has_offtopic_signal(user_q):
            return decision_for_mode(
                "general_knowledge",
                reason="Classifier returned off_topic, but agriculture terms keep the request in scope.",
            )

    return decision_for_mode(mode, reason="LLM turn-strategy classifier selected this mode.")


async def _route_turn_mode(
    *,
    user_q: str,
    prompt_q: str,
    history_text: str,
    has_uploaded_files: bool = False,
) -> TurnMode:
    decision = await _route_turn_decision(
        user_q=user_q,
        prompt_q=prompt_q,
        history_text=history_text,
        has_uploaded_files=has_uploaded_files,
    )
    return decision.mode


async def _normalize_query_for_retrieval(text: str) -> str:
    """
    Best-effort query cleanup for spelling/grammar to improve retrieval.
    Keeps meaning unchanged and falls back immediately on any error/timeout.
    """
    raw = (text or "").strip()
    if not raw:
        return raw

    prompt = (
        "Rewrite the user query with corrected spelling/grammar, preserving exact intent. "
        "Keep it concise, one line, no explanation.\n\n"
        f"Query: {raw}\n\n"
        "Rewritten query:"
    )
    try:
        rewritten = await asyncio.wait_for(
            generate_once(prompt, temperature=0.0, max_tokens=48),
            timeout=1.2,
        )
    except Exception:
        return raw

    line = (rewritten or "").strip().splitlines()
    if not line:
        return raw
    cleaned = line[0].strip(" \"'")
    return cleaned or raw


async def _maybe_update_session_title(
    session_id: str,
    question: str,
    answer: str | None = None,
    auth_token: str | None = None,
):
    if not session_id or not CHAT_BACKEND_URL:
        return

    title_prompt = build_title_prompt(question, answer)
    try:
        raw_title = await generate_once(
            title_prompt,
            temperature=0.2,
            max_tokens=16,
        )
    except httpx.HTTPError:
        return

    lines = (raw_title or "").strip().splitlines()
    if not lines:
        logger.warning(f"Title generation returned empty output for session {session_id[:8]}...")
        return

    title = _short_title_2_3_words(lines[0])
    if not title:
        return

    url = f"{CHAT_BACKEND_URL}/chat/sessions/{session_id}/"
    payload = {"title": title}

    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    headers = {}
    if auth_token:
        headers["Authorization"] = auth_token

    async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
        try:
            r = await client.patch(url, json=payload, headers=headers)
            if r.is_error:
                body_snippet = (r.text or "")[:300]
                logger.warning(
                    "Failed to update session title: "
                    f"HTTP {r.status_code}, session={session_id[:8]}..., body={body_snippet}"
                )
                return
            logger.info(f"Updated session title: {r.status_code}")
        except httpx.HTTPError:
            pass


@router.get("/chatbot/api/chats/message/stream", tags=["Chats"], summary="Stream a message without an existing chat")
@router.get("/chatbot/api/chats/{session_id}/message/stream", tags=["Chats"], summary="Stream a message for an existing chat")
@router.get("/ask/stream", include_in_schema=False)
async def ask_stream(
    q: str,
    page: int = 1,
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    followup_hint: Optional[str] = None,
    doc_ids: Optional[str] = None,
    client_history: Optional[str] = None,
    replace_history: bool = False,
    pause_personalization: bool = False,
    request: Request = None,
):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Natural conversation flow - sends full history to LLM, lets it handle context.
    """

    async def emit(event: str, data):
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        yield {"event": event, "data": data}

    async def emit_app_error(payload: dict):
        # Use a custom event name to avoid collision with EventSource transport "error".
        async for x in emit("app_error", payload):
            yield x

    normalized_model = _normalize_model(model)

    async def gen():
        requested_doc_ids = [d.strip() for d in (doc_ids or "").split(",") if d.strip()]

        t0 = time.perf_counter()
        
        # Extract auth info from the standard Authorization header.
        auth_token = request.headers.get("Authorization", "") if request else ""
        
        user_uuid = _extract_user_uuid_from_token(auth_token) if auth_token else None
        owner_scope = user_uuid or "anonymous"

        local_pdf_docs = get_docs_for_user(requested_doc_ids, owner_scope) if requested_doc_ids else []
        local_image_docs = get_images_for_user(requested_doc_ids, owner_scope) if requested_doc_ids else []

        user_q_was_empty = not bool((q or "").strip())
        raw_user_q = (q or "").strip()
        if not raw_user_q and (local_pdf_docs or local_image_docs):
            raw_user_q = _attachment_summary_prompt(
                has_images=bool(local_image_docs),
                has_pdfs=bool(local_pdf_docs),
                count=len(local_pdf_docs) + len(local_image_docs),
            )

        user_q = raw_user_q
        if not user_q:
            async for x in emit_app_error({"message": "Empty question"}):
                yield x
            return
        if _estimate_tokens(user_q) > S.MAX_USER_INPUT_TOKENS:
            async for x in emit_app_error({
                "message": (
                    f"Question is too long. Limit is ~{S.MAX_USER_INPUT_TOKENS} tokens per message."
                )
            }):
                yield x
            return
        
        # Load conversation state
        history_text: str = ""
        initial_llm_ctx: list[int] | None = None
        client_messages: list[dict] = []

        if client_history:
            try:
                parsed = json.loads(client_history)
                if isinstance(parsed, list):
                    client_messages = parsed
            except Exception:
                client_messages = []

        state = {"messages": [], "llm_context": None}
        if session_id and not replace_history:
            state = await load_chat_state(session_id, auth_token)

        # When the caller signals replace_history=true (regenerate / edit-and-resend),
        # the client_history is treated as authoritative and the persisted session is
        # ignored for THIS turn's prompt construction.
        if replace_history:
            merged_messages = list(client_messages)
        else:
            merged_messages = merge_messages(state.get("messages", []), client_messages)
        state["messages"] = merged_messages

        history_text = format_history(state.get("messages", []))
        initial_llm_ctx = state.get("llm_context")
        is_first_turn = not state.get("messages")
        prev_q = _extract_last_assistant_question(state.get("messages", []))
        if not prev_q and followup_hint:
            prev_q = (followup_hint or "").strip()[:300]

        # --- Pre-flight pipeline ---
        # We have three small LLM hops (turn-context resolve, query normalization,
        # turn-strategy router) and one HTTP-pair (user profile + facts) before the
        # main stream can start. Reduce wall time by:
        #   1. Skipping `_resolve_turn_context` when the message is plainly standalone.
        #   2. Skipping `_decide_turn_strategy` when a heuristic classifier is confident.
        #   3. Running whatever survives concurrently with profile loading.

        forced_turn_strategy = _hard_route_turn_mode(user_q)

        async def _profile_task() -> str:
            if forced_turn_strategy == "clarification_only":
                return ""
            if pause_personalization:
                # User asked to pause memory for this turn — don't load or inject.
                return ""
            if not (user_uuid and auth_token):
                return ""
            try:
                profile = await UserProfileService.get_or_create_profile(user_uuid, auth_token)
                facts, memory_notes = await asyncio.gather(
                    UserProfileService.get_facts(user_uuid, auth_token, limit=5),
                    UserProfileService.get_memory_notes(user_uuid, auth_token, limit=10),
                )
                return UserProfileService.build_profile_context(profile, facts, memory_notes)
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")
                return ""

        profile_future = asyncio.create_task(_profile_task())

        if _looks_standalone(user_q):
            effective_q = user_q
            prompt_q = user_q
        else:
            turn_context = await _resolve_turn_context(
                question=user_q,
                history_text=history_text,
                last_assistant_question=prev_q,
                followup_hint=followup_hint or "",
            )
            effective_q = (turn_context.get("resolved_user_message") or user_q).strip() or user_q
            prompt_q = (turn_context.get("assistant_instruction") or effective_q).strip() or effective_q

        async def _retrieval_q_task() -> str:
            if requested_doc_ids and _is_file_handoff_query(user_q):
                return (
                    "Summarize the uploaded attachment(s) with key points and practical takeaways, "
                    "then ask one focused follow-up question."
                )
            if _should_skip_query_normalization(effective_q):
                return effective_q
            return await _normalize_query_for_retrieval(effective_q)

        async def _strategy_task() -> ScopeRouteDecision:
            return await _route_turn_decision(
                user_q=user_q,
                prompt_q=prompt_q,
                history_text=history_text,
                has_uploaded_files=bool(requested_doc_ids or local_pdf_docs or local_image_docs),
            )

        retrieval_q, route_decision, profile_context, answer_language = await asyncio.gather(
            _retrieval_q_task(),
            _strategy_task(),
            profile_future,
            _resolve_answer_language(user_q),
        )
        turn_strategy = route_decision.mode

        # Concurrency gate. Arena requests pass X-Farm-Experiment-Backend, so one
        # Farm Assistant process can queue independently per LLM backend instead
        # of using one global generation slot pool for all variants.
        def _generation_queue_key() -> str:
            headers = request.headers if request is not None else {}
            backend_key = (headers.get("x-farm-experiment-backend") or "").strip()
            if backend_key:
                return backend_key
            provider = (getattr(S, "LLM_PROVIDER", "") or "vllm").strip().lower() or "vllm"
            if provider == "anthropic":
                return f"anthropic:{getattr(S, 'ANTHROPIC_MODEL', '') or 'default'}"
            return f"vllm:{S.VLLM_URL}:{normalized_model}"

        async def _get_generation_semaphore(queue_key: str):
            state = getattr(request.app, "state", None) if request is not None else None
            if state is None:
                return None

            semaphores = getattr(state, "gen_semaphores", None)
            if semaphores is None:
                return getattr(state, "gen_semaphore", None)

            limit = int(getattr(state, "gen_semaphore_limit", 3) or 3)
            lock = getattr(state, "gen_semaphores_lock", None)
            if lock is None:
                sem = semaphores.get(queue_key)
                if sem is None:
                    sem = asyncio.Semaphore(limit)
                    semaphores[queue_key] = sem
                return sem

            async with lock:
                sem = semaphores.get(queue_key)
                if sem is None:
                    sem = asyncio.Semaphore(limit)
                    semaphores[queue_key] = sem
                return sem

        queue_key = _generation_queue_key()
        sem = None

        async def _acquire_or_queue():
            nonlocal sem
            sem = await _get_generation_semaphore(queue_key)
            if sem is None:
                return
            if getattr(sem, "_value", 1) == 0:
                async for x in emit("status", {"stage": "Queue", "message": f"Waiting for a free {queue_key} slot..."}):
                    yield x
            await sem.acquire()

        async def _release():
            try:
                if sem is not None:
                    sem.release()
            except Exception:
                pass

        inp = AskIn(
            question=retrieval_q,
            page=page,
            k=k,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        contexts: list[str] = []
        sources: list = []
        attachment_contexts: list[str] = []
        attachment_sources: list = []
        retrieval_context_count = 0
        web_context_count = 0
        retrieval_quality = 0.0
        image_stats = {"total": 0, "agri_related": 0, "non_agri": 0}
        attachment_handoff = bool(requested_doc_ids) and (
            _is_file_handoff_query(user_q)
            or _mentions_file_or_document(user_q)
            or user_q_was_empty
        )
        t_search_start = time.perf_counter()
        t_search_end = t_search_start
        t_ctx_start = t_search_start
        t_ctx_end = t_search_start

        if turn_strategy == "clarification_only":
            async for x in emit("status", {"stage": "Clarify", "message": "Need a clearer question..."}):
                yield x
        elif turn_strategy == "off_topic":
            # Non-agriculture / chit-chat / quote / model-identity probe: skip retrieval
            # entirely and route to a polite refusal builder.
            async for x in emit("status", {"stage": "Scope", "message": "Off-topic question, skipping search..."}):
                yield x
        elif turn_strategy == "history_only":
            async for x in emit("status", {"stage": "History", "message": "Answering from conversation history..."}):
                yield x
        elif turn_strategy == "conversation_only":
            async for x in emit("status", {"stage": "Conversation", "message": "Answering from the current conversation..."}):
                yield x
        elif turn_strategy == "assistant_capabilities":
            async for x in emit("status", {"stage": "Capabilities", "message": "Answering about assistant capabilities..."}):
                yield x
        elif turn_strategy == "platform_operation":
            async for x in emit("status", {"stage": "Platform", "message": "Answering about EU-FarmBook platform scope..."}):
                yield x
        elif turn_strategy == "general_knowledge":
            # Skip OpenSearch: this question is answerable from common agricultural
            # knowledge alone. No `items`, no `contexts`, no citations.
            async for x in emit("status", {"stage": "Knowledge", "message": "Answering from general agricultural knowledge..."}):
                yield x
        else:
            # --- Normal retrieval flow ---
            async for x in emit("status", {"stage": "Search", "message": "Searching sources..."}):
                yield x

            search_payload = build_search_payload(inp)
            headers = os_headers()
            auth = os_auth()
            pages = [1]

            t_search_start = time.perf_counter()

            try:
                async with os_async_client(timeout=30.0) as client:
                    items = await collect_os_items(client, search_payload, pages, headers, auth)
            except httpx.HTTPStatusError as e:
                body = (e.response.text or "")[:400]
                async for x in emit_app_error({"stage": "search", "status": e.response.status_code, "body": body}):
                    yield x
                return

            t_search_end = time.perf_counter()

            filtered_items, score_filter_stats = filter_items_by_min_score(
                items,
                min_score=S.RETRIEVAL_MIN_SCORE,
            )
            if score_filter_stats["discarded_count"] > 0:
                logger.info(
                    "Score-filtered retrieval items: kept=%s discarded=%s threshold=%.3f",
                    score_filter_stats["kept_count"],
                    score_filter_stats["discarded_count"],
                    score_filter_stats["min_score_threshold"],
                )
            items = filtered_items

            # --- Build contexts ---
            async for x in emit("status", {"stage": "Context", "message": "Preparing context..."}):
                yield x

            t_ctx_start = time.perf_counter()
            t_k = inp.top_k if inp.top_k is not None else inp.k
            if not isinstance(t_k, int) or t_k <= 0:
                t_k = S.TOP_K
            contexts, sources = build_context_and_sources(
                items=items, question=retrieval_q, top_k=t_k, max_context_chars=S.MAX_CONTEXT_CHARS
            )

            # Relevance gate: cheap token-overlap check on the top retrieval items.
            # If overlap is very low, OpenSearch returned weakly-related results;
            # rather than ground the LLM in noise, drop them and treat as a
            # no-sources turn. The LLM prompt then says "no EU-FarmBook material
            # was found, give a cautious best-effort answer".
            # Relevance gate. Two modes produce a 0..1 `retrieval_quality` plus the
            # drop/web thresholds: lexical word-overlap (default) or scout's calibrated
            # semantic_score. Semantic mode is used only when items actually carry the
            # field; otherwise we fall back to word-overlap automatically.
            drop_threshold = S.RETRIEVAL_DROP_THRESHOLD
            web_threshold = S.WEB_FALLBACK_QUALITY_THRESHOLD
            semantic_quality = (
                estimate_semantic_quality(items, top_n=3)
                if (S.RELEVANCE_MODE or "").strip().lower() == "semantic"
                else None
            )
            if semantic_quality is not None:
                retrieval_quality = semantic_quality
                drop_threshold = S.SEMANTIC_DROP_THRESHOLD
                web_threshold = S.SEMANTIC_WEB_THRESHOLD
            elif items and contexts:
                retrieval_quality = estimate_retrieval_quality(retrieval_q, items, top_n=3)

            if contexts and retrieval_quality < drop_threshold:
                logger.info(
                    "Dropped %d retrieved contexts: quality=%.3f below drop=%.3f (mode=%s)",
                    len(contexts), retrieval_quality, drop_threshold,
                    "semantic" if semantic_quality is not None else "overlap",
                )
                contexts = []
                sources = []
            retrieval_context_count = len(contexts)

            # --- Trusted web-search fallback (gated, default OFF) ---
            # Fires when internal retrieval is empty OR weak-but-present. The backend
            # searches an allowlist and feeds extracted passages as additional cited
            # grounding; the model never browses. KO sources keep their citation
            # numbers; web sources are appended after them.
            if S.WEB_FALLBACK_ENABLED:
                needs_web = (not contexts) or (retrieval_quality < web_threshold)
                if needs_web:
                    async for x in emit("status", {"stage": "Web", "message": "Searching trusted external sources..."}):
                        yield x
                    try:
                        web_contexts, web_sources = await web_search_and_build_contexts(
                            retrieval_q,
                            max_results=S.WEB_FALLBACK_MAX_RESULTS,
                            max_chars=min(
                                S.WEB_FALLBACK_MAX_CHARS,
                                S.MAX_CONTEXT_CHARS - sum(len(c) for c in contexts),
                            ),
                            sid_offset=len(sources),
                        )
                    except Exception as e:
                        logger.warning(f"Web fallback failed: {e}")
                        web_contexts, web_sources = [], []
                    if web_contexts:
                        contexts.extend(web_contexts)
                        sources.extend(web_sources)
                        web_context_count = len(web_contexts)

            t_ctx_end = time.perf_counter()

        persisted_attachment_records = []
        if session_id and auth_token and (requested_doc_ids or _mentions_file_or_document(user_q)):
            persisted_attachment_records = await fetch_session_attachments_from_backend(
                chat_backend_url=CHAT_BACKEND_URL,
                verify_ssl=S.VERIFY_SSL,
                auth_token=auth_token,
                session_uuid=session_id,
            )

        if local_pdf_docs or local_image_docs or persisted_attachment_records:
            async for x in emit("status", {"stage": "Attachments", "message": "Preparing uploaded attachments..."}):
                yield x

            local_pdfs = list(local_pdf_docs)
            local_images = list(local_image_docs)

            if local_pdfs:
                for d in local_pdfs:
                    await ensure_pdf_processed(d)
                    if session_id and auth_token:
                        asyncio.create_task(
                            upsert_attachment_to_backend(
                                chat_backend_url=CHAT_BACKEND_URL,
                                verify_ssl=S.VERIFY_SSL,
                                auth_token=auth_token,
                                session_uuid=session_id,
                                doc=d,
                            )
                        )

            if local_images:
                for d in local_images:
                    await ensure_image_processed(d)
                    if session_id and auth_token:
                        asyncio.create_task(
                            upsert_image_attachment_to_backend(
                                chat_backend_url=CHAT_BACKEND_URL,
                                verify_ssl=S.VERIFY_SSL,
                                auth_token=auth_token,
                                session_uuid=session_id,
                                doc=d,
                            )
                        )

            persisted_pdfs = docs_from_attachment_records(
                persisted_attachment_records,
                owner_id=(user_uuid or "persisted"),
            ) if persisted_attachment_records else []
            persisted_images = images_from_attachment_records(
                persisted_attachment_records,
                owner_id=(user_uuid or "persisted"),
            ) if persisted_attachment_records else []

            seen_attachment_ids = {d.doc_id for d in local_pdfs + local_images}
            merged_pdfs = local_pdfs + [d for d in persisted_pdfs if d.doc_id not in seen_attachment_ids]
            merged_images = local_images + [d for d in persisted_images if d.doc_id not in seen_attachment_ids]

            remaining = max(2000, S.MAX_CONTEXT_CHARS - sum(len(c) for c in contexts))
            if merged_pdfs:
                pdf_contexts, pdf_sources = build_pdf_contexts(
                    merged_pdfs,
                    question=retrieval_q,
                    max_total_chars=remaining,
                )
                attachment_contexts.extend(pdf_contexts)
                remaining = max(1000, remaining - sum(len(c) for c in pdf_contexts))
                for s in pdf_sources:
                    attachment_sources.append(type("PdfSrc", (), {
                        "sid": None,
                        "id": s.get("id"),
                        "title": s.get("title"),
                        "display_url": None,
                        "url": None,
                        "license": None,
                    })())

            if merged_images:
                image_contexts, image_sources_raw, image_stats = build_image_contexts(
                    merged_images,
                    max_total_chars=remaining,
                )
                attachment_contexts.extend(image_contexts)
                for s in image_sources_raw:
                    attachment_sources.append(type("ImageSrc", (), {
                        "sid": None,
                        "id": s.get("id"),
                        "title": s.get("title"),
                        "display_url": None,
                        "url": None,
                        "license": None,
                    })())

            contexts.extend(attachment_contexts)
            sources.extend(attachment_sources)

            # Non-agri images used to force a hard off_topic refusal. That
            # silently dropped the vision summary and produced the generic
            # "I can't view or analyze images" reply, which is confusing
            # because FA *did* look at the image. Instead we let the LLM see
            # the (now-tagged) image context and rely on the prompt directive
            # to describe what was observed and steer back to agriculture.
            if attachment_handoff and contexts and turn_strategy in {"off_topic", "conversation_only", "general_knowledge"}:
                route_decision = file_analysis_decision(
                    reason="Uploaded attachment context became available; route switched to file analysis."
                )
                turn_strategy = route_decision.mode

        t_ctx_end = time.perf_counter()

        all_sources = [
            {
                "n": i + 1,
                "sid": getattr(s, "sid", None),
                "id": getattr(s, "id", None),
                "title": getattr(s, "title", None),
        # This value is resolved solely from the raw latest user message.
                "url": getattr(s, "url", None),
                "display_url": getattr(s, "display_url", None),
                "license": getattr(s, "license", None),
            }
            for i, s in enumerate(sources)
        ]
        if turn_strategy == "normal":
            if retrieval_context_count > 0 and web_context_count > 0:
                grounding_state = "euf_web_supported"
            elif web_context_count > 0:
                grounding_state = "web_supported"
            elif retrieval_context_count > 0:
                grounding_state = "euf_supported"
            elif attachment_contexts:
                grounding_state = "attachment_supported"
            else:
                grounding_state = "general_fallback"
        else:
            grounding_state = "euf_supported"
        if turn_strategy in {"clarification_only", "off_topic", "history_only", "conversation_only", "assistant_capabilities", "platform_operation", "general_knowledge"}:
            grounding_state = turn_strategy

        # --- Build prompt with conversation history ---
        _max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_OUTPUT_TOKENS
        if _max_tokens <= 0 or _max_tokens > S.MAX_OUTPUT_TOKENS:
            _max_tokens = S.MAX_OUTPUT_TOKENS
        _temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE
        
        # The answer language is resolved solely from the raw latest user message.
        history_messages_for_prompt = state.get("messages", [])
        if turn_strategy == "clarification_only":
            messages = build_clarification_messages(
                question=user_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=None,
                answer_language=answer_language,
            )
        elif turn_strategy == "off_topic":
            messages = build_off_topic_messages(
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                answer_language=answer_language,
            )
        elif turn_strategy == "history_only":
            messages = build_history_only_messages(
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                answer_language=answer_language,
            )
        elif turn_strategy == "conversation_only":
            messages = build_conversation_only_messages(
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                answer_language=answer_language,
            )
        elif turn_strategy == "assistant_capabilities":
            messages = build_capabilities_messages(
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                answer_language=answer_language,
            )
        elif turn_strategy == "platform_operation":
            messages = build_platform_operation_messages(
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                answer_language=answer_language,
            )
        elif turn_strategy == "general_knowledge":
            messages = build_general_knowledge_messages(
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                answer_language=answer_language,
            )
        else:
            messages = build_messages(
                contexts=contexts if contexts else [],
                question=prompt_q,
                history_messages=history_messages_for_prompt,
                user_profile_context=profile_context,
                has_relevant_sources=bool(contexts),
                answer_language=answer_language,
                has_web_sources=web_context_count > 0,
            )
        prompt_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        prompt_cap = min(
            S.MAX_INPUT_TOKENS,
            max(256, int(S.NUM_CTX) - int(_max_tokens) - 256),
        )
        if prompt_tokens > prompt_cap:
            async for x in emit_app_error({
                "message": (
                    f"Input context is too large (~{prompt_tokens} tokens). "
                    f"Please shorten your question or reduce attached content."
                )
            }):
                yield x
            return

        # --- LLM stream ---
        async for x in emit("grounding", {"mode": grounding_state}):
            yield x

        t_llm_start = time.perf_counter()
        ctx = initial_llm_ctx
        answer_chunks: List[str] = []
        llm_usage = None

        if turn_strategy == "platform_operation":
            async for x in emit("status", {"stage": "Platform", "message": "Applying EU-FarmBook platform scope..."}):
                yield x
            chunk = _platform_operation_static_answer(answer_language)
            answer_chunks.append(chunk)
            async for x in emit("token", chunk):
                yield x
            async for x in emit("stats", {"done": True, "deterministic": True}):
                yield x
        else:
            # Concurrency gate
            async for x in _acquire_or_queue():
                yield x

            async for x in emit("status", {"stage": "LLM", "message": "Generating response..."}):
                yield x
            try:
                # vLLM streams the whole answer in one pass; there is no Ollama-style
                # "context" continuation. If we ever need to handle done_reason="length"
                # (e.g. switch to a different backend later), reintroduce a continuation
                # loop here — but for now a single call is correct and clearer.
                async for obj in stream_generate(
                    "", _temperature, _max_tokens,
                    context=ctx, model=normalized_model, num_ctx=S.NUM_CTX,
                    messages=messages,
                ):
                    if "response" in obj and obj["response"]:
                        chunk = obj["response"]
                        answer_chunks.append(chunk)
                        async for x in emit("token", chunk):
                            yield x

                    if "context" in obj and obj["context"]:
                        ctx = obj["context"]

                    if obj.get("done"):
                        llm_usage = obj.get("usage") or llm_usage
                        stats = {k: v for k, v in obj.items() if k not in ("response", "prompt")}
                        async for x in emit("stats", stats):
                            yield x

            except httpx.ConnectError as e:
                logger.error(f"Cannot connect to LLM backend: {e}")
                async for x in emit_app_error({"stage": "LLM", "message": f"Cannot connect to LLM: {e}"}):
                    yield x
                return
            except httpx.HTTPStatusError as e:
                logger.error(f"LLM HTTP error {e.response.status_code}: {e.response.text[:500]}")
                async for x in emit_app_error({"stage": "LLM", "status": e.response.status_code}):
                    yield x
                return
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}")
                async for x in emit_app_error({"stage": "LLM", "message": f"Error: {str(e)}"}):
                    yield x
                return
            finally:
                await _release()

        t_llm_end = time.perf_counter()
        full_text = _sanitize_generated_markdown("".join(answer_chunks))

        # Persist state
        if session_id:
            await save_chat_state(session_id, ctx)
            if is_first_turn:
                asyncio.create_task(
                    _maybe_update_session_title(session_id, user_q, full_text, auth_token)
                )

        # Update profile (fire-and-forget) — only on substantive turns. Greetings,
        # acknowledgements, and capability questions don't carry profile signal,
        # and writing them anyway pollutes the profile (and burns a vLLM call).
        # Also skip when the user has paused personalization for this turn.
        if (
            user_uuid
            and session_id
            and auth_token
            and not pause_personalization
            and turn_strategy not in ("clarification_only", "off_topic", "conversation_only", "assistant_capabilities", "platform_operation")
        ):
            asyncio.create_task(
                UserProfileService.process_conversation_turn(
                    user_uuid, session_id, user_q, full_text, auth_token
                )
            )

        # Extract citations - handle formats like [1], [S1], [1, 2, 3], [S1, S3], (source: [1]), etc.
        norm_text = _re.sub(r"\(\s*source[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\s*\)", r"[\1]", full_text, flags=_re.IGNORECASE)
        norm_text = _re.sub(r"\bsource[s]?:?\s*\[?\s*(?:S)?(\d+)\s*\]?\b", r"[\1]", norm_text, flags=_re.IGNORECASE)
        # Convert [S1] -> [1], (S1) -> [1]
        norm_text = _re.sub(r"\[\s*[sS](\d+)\s*\]", r"[\1]", norm_text)
        norm_text = _re.sub(r"\(\s*[sS](\d+)\s*\)", r"[\1]", norm_text)
        # Handle comma-separated S-prefixed lists like [S1, S3, S4, S5] -> [1, 3, 4, 5]
        norm_text = _re.sub(r"\[\s*([sS]\d+(?:\s*[,–-]\s*[sS]?\d+)*)\s*\]", lambda m: "[" + _re.sub(r"[sS]", "", m.group(1)) + "]", norm_text)

        cited_nums = _extract_cited_numbers(norm_text)
        valid_source_numbers = {s["n"] for s in all_sources}
        full_text = _strip_orphan_citations(full_text, valid_source_numbers)
        norm_text = _strip_orphan_citations(norm_text, valid_source_numbers)
        cited_nums = _extract_cited_numbers(norm_text)

        # Token streaming is optimistic. Send the citation-sanitized final text so
        # clients can replace any orphan references the model emitted mid-stream.
        async for x in emit("final", {"text": full_text}):
            yield x

        if cited_nums:
            by_num = {s["n"]: s for s in all_sources}
            cited_sources = [by_num[n] for n in sorted(cited_nums) if n in by_num]
            if cited_sources:
                async for x in emit("sources", cited_sources):
                    yield x
        else:
            async for x in emit("sources", []):
                yield x

        # Timing
        total_ms = int((time.perf_counter() - t0) * 1000)
        search_ms = int((t_search_end - t_search_start) * 1000)
        context_ms = int((t_ctx_end - t_ctx_start) * 1000)
        llm_ms = int((t_llm_end - t_llm_start) * 1000)
        async for x in emit("timing", {
            "total_ms": total_ms,
            "search_ms": search_ms,
            "context_ms": context_ms,
            "llm_ms": llm_ms,
            "usage": llm_usage,
        }):
            yield x

        async for x in emit("done", {"message": "complete"}):
            yield x

    return EventSourceResponse(gen(), ping=10)


@router.post("/chatbot/api/chats/message", tags=["Chats"], summary="Send a message without an existing chat")
async def chat_message_stream_create(
    body: ChatMessageStreamIn,
    request: Request,
):
    return await ask_stream(
        q=body.q,
        page=body.page,
        k=body.k,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        model=body.model,
        session_id=None,
        followup_hint=body.followup_hint,
        doc_ids=",".join(body.doc_ids),
        client_history=json.dumps(body.client_history) if body.client_history else None,
        replace_history=body.replace_history,
        pause_personalization=body.pause_personalization,
        request=request,
    )


@router.post("/chatbot/api/chats/{session_id}/message", tags=["Chats"], summary="Send a message to an existing chat")
async def chat_message_stream_existing(
    session_id: str,
    body: ChatMessageStreamIn,
    request: Request,
):
    return await ask_stream(
        q=body.q,
        page=body.page,
        k=body.k,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        model=body.model,
        session_id=session_id,
        followup_hint=body.followup_hint,
        doc_ids=",".join(body.doc_ids),
        client_history=json.dumps(body.client_history) if body.client_history else None,
        replace_history=body.replace_history,
        pause_personalization=body.pause_personalization,
        request=request,
    )


@router.post("/summarise", response_model=SummariseOut)
async def summarise(inp: SummariseIn) -> SummariseOut:
    """Summarise a single text chunk according to a user-supplied prompt."""
    user_prompt = (inp.prompt or "").strip()
    text_chunk = (inp.text or "").strip()

    if not user_prompt:
        return SummariseOut(summary="", meta={"error": "Empty prompt"})
    if not text_chunk:
        return SummariseOut(summary="", meta={"error": "Empty text"})

    prompt = build_summary_prompt(user_prompt, text_chunk)

    max_tokens = inp.max_tokens if inp.max_tokens is not None else S.MAX_TOKENS
    temperature = inp.temperature if inp.temperature is not None else S.TEMPERATURE

    try:
        summary = await generate_once(prompt, temperature, max_tokens)
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:400]
        logger.error(f"LLM error {e.response.status_code}: {body}")
        return SummariseOut(
            summary="",
            meta={
                "upstream_status": e.response.status_code,
                "upstream_body_snippet": body,
            },
        )

    return SummariseOut(
        summary=summary,
        meta={
            "model": S.VLLM_MODEL,
            "num_ctx": S.NUM_CTX,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )


_FOLLOW_UPS_SYSTEM_PROMPT = (
    "You suggest evidence-aware follow-up questions for an agricultural assistant. "
    "Given the user's last question, the assistant's reply, and optional source labels, propose up to "
    "3 short, distinct follow-up questions the user might naturally ask next. Each follow-up must:\n"
    "- be a single question, phrased in the user's voice (\"How do I...\", \"What about...\")\n"
    "- stay strictly on the agricultural topic supported by the reply or source labels\n"
    "- never mention EU-FarmBook, platform features, accounts, registration, dashboards, uploads, "
    "data linking, imports, exports, synchronization, or database integrations\n"
    "- be no longer than 80 characters\n"
    "- be different from each other and from the user's prior question\n"
    "- be in the same language as the user's prior question\n\n"
    "Source labels identify relevant topics only; do not invent facts or capabilities from them. "
    "Return ONLY a JSON array of 0 to 3 strings. No prose, no markdown, no keys. "
    "If you cannot produce useful suggestions, return [].\n"
)

_FOLLOW_UPS_ALLOWED_MODES = {
    "attachment_supported",
    "euf_supported",
    "web_supported",
    "euf_web_supported",
}

_PROHIBITED_FOLLOW_UP_PATTERNS = (
    _re.compile(r"\beu[\s-]*farmbook\b", _re.IGNORECASE),
    _re.compile(r"\b(?:platform|dashboard|portal)\b", _re.IGNORECASE),
    _re.compile(r"\b(?:register|sign[\s-]*up|create)\b.{0,35}\b(?:account|farm|profile)\b", _re.IGNORECASE),
    _re.compile(r"\b(?:account|profile)\b.{0,35}\b(?:register|sign[\s-]*up|create|manage)\b", _re.IGNORECASE),
    _re.compile(
        r"\b(?:upload|import|export|link|connect|sync|synchroni[sz]e|share)\w*\b"
        r".{0,50}\b(?:data|database|record|system)\w*\b",
        _re.IGNORECASE,
    ),
    _re.compile(
        r"\b(?:data|database|record|system)\w*\b"
        r".{0,50}\b(?:upload|import|export|link|connect|sync|synchroni[sz]e|share)\w*\b",
        _re.IGNORECASE,
    ),
    _re.compile(r"\b(?:EU|European Union|agricultural)\b.{0,25}\bdatabase\b", _re.IGNORECASE),
)


def _is_prohibited_follow_up(candidate: str) -> bool:
    return any(pattern.search(candidate) for pattern in _PROHIBITED_FOLLOW_UP_PATTERNS)


_EXPORT_INTENT_SYSTEM_PROMPT = """Classify the latest user message in any language.
Return ONLY one JSON object with these keys:
- intent: normal_chat, export_previous, or generate_export
- format: pdf, docx, csv, xlsx, pptx, or null
- scope: previous_answer, conversation, or null
- confidence: number from 0 to 1

Use export_previous when the user asks to convert/download/save existing chat content.
Use scope=previous_answer when the user asks for this answer, that answer, or the preceding answer.
Use scope=conversation when the user asks for all of this, everything above, the whole chat, this conversation, or chat history.
Use generate_export when the user asks for new substantive content delivered as a file.
Use normal_chat for questions about file formats, unsupported formats, or ordinary conversation.
Never generate document content. Never add prose or markdown."""
_EXPORT_FORMATS = {"pdf", "docx", "csv", "xlsx", "pptx"}
_EXPORT_INTENTS = {"normal_chat", "export_previous", "generate_export"}
_EXPORT_SCOPES = {"previous_answer", "conversation"}


def _detect_export_format(text: str) -> str | None:
    q = (text or "").strip().lower()
    if not q:
        return None
    if "excel" in q or "xlsx" in q or "spreadsheet" in q:
        return "xlsx"
    for export_format in ("pdf", "docx", "csv", "pptx"):
        if _re.search(rf"\b{export_format}\b", q):
            return export_format
    if _re.search(r"\bpower\s*point\b", q):
        return "pptx"
    if _re.search(r"\bword\b", q):
        return "docx"
    return None


def _requests_conversation_export(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    return any(
        phrase in q
        for phrase in (
            "all of this",
            "all this",
            "everything above",
            "everything so far",
            "whole conversation",
            "entire conversation",
            "full conversation",
            "this conversation",
            "whole chat",
            "entire chat",
            "full chat",
            "chat history",
            "conversation history",
        )
    )


def _is_export_format_followup(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q or not _detect_export_format(q):
        return False
    return bool(_re.search(r"^(what about|and|also|same|again|as|in)\b", q))


def _deterministic_export_intent(
    query: str,
    *,
    previous_export_scope: str | None = None,
    has_conversation: bool = False,
) -> ExportIntentOut | None:
    export_format = _detect_export_format(query)
    if not export_format:
        return None

    if _requests_conversation_export(query):
        return ExportIntentOut(
            intent="export_previous",
            format=export_format,
            scope="conversation" if has_conversation else "previous_answer",
            confidence=1.0,
            meta={"reason": "deterministic_conversation_scope"},
        )

    if _is_export_format_followup(query) and previous_export_scope in _EXPORT_SCOPES:
        return ExportIntentOut(
            intent="export_previous",
            format=export_format,
            scope=previous_export_scope,
            confidence=1.0,
            meta={"reason": "deterministic_previous_export_scope"},
        )

    return None


def _parse_export_intent(raw: str) -> ExportIntentOut:
    if not raw:
        return ExportIntentOut(meta={"reason": "empty_response"})

    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ExportIntentOut(meta={"reason": "invalid_json"})

    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return ExportIntentOut(meta={"reason": "invalid_json"})

    intent = str(data.get("intent") or "").strip().lower()
    export_format = str(data.get("format") or "").strip().lower() or None
    scope = str(data.get("scope") or "").strip().lower() or None
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    if intent not in _EXPORT_INTENTS:
        return ExportIntentOut(meta={"reason": "invalid_intent"})
    if intent == "normal_chat":
        return ExportIntentOut(intent="normal_chat", confidence=confidence)
    if export_format not in _EXPORT_FORMATS:
        return ExportIntentOut(meta={"reason": "invalid_format"})
    if scope not in _EXPORT_SCOPES:
        scope = "previous_answer"
    if confidence < 0.75:
        return ExportIntentOut(confidence=confidence, meta={"reason": "low_confidence"})

    return ExportIntentOut(intent=intent, format=export_format, scope=scope, confidence=confidence)


def _parse_follow_ups(raw: str, enforce_policy: bool = True) -> list[str]:
    if not raw:
        return []

    text = raw.strip()
    # Strip any leading/trailing fences the model might emit despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate:
            continue
        if len(candidate) > 80:
            continue
        if enforce_policy and _is_prohibited_follow_up(candidate):
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(candidate)
        if len(cleaned) >= 3:
            break
    return cleaned


@router.post("/chatbot/api/export-intent", response_model=ExportIntentOut, tags=["Chats"], summary="Classify a document export request")
async def classify_export_intent(inp: ExportIntentIn) -> ExportIntentOut:
    query = (inp.query or "").strip()
    previous = (inp.previous_assistant_message or "").strip()
    previous_export_scope = (inp.previous_export_scope or "").strip().lower() or None
    has_conversation = bool(inp.conversation_messages)

    deterministic = _deterministic_export_intent(
        query,
        previous_export_scope=previous_export_scope,
        has_conversation=has_conversation,
    )
    if deterministic is not None:
        deterministic.meta["model"] = "deterministic"
        return deterministic

    previous_exists = "yes" if previous else "no"
    conversation_exists = "yes" if has_conversation else "no"
    previous_excerpt = previous[:1200] if previous else "(none)"
    conversation_excerpt = "\n".join(
        f"{m.role}: {m.content}"[:500]
        for m in inp.conversation_messages[:8]
    )
    if not conversation_excerpt:
        conversation_excerpt = "(none)"
    previous_scope_label = previous_export_scope or "(none)"
    user_content = (
        f"Previous assistant answer exists: {previous_exists}\n"
        f"Previous assistant answer excerpt:\n{previous_excerpt}\n\n"
        f"Conversation messages exist: {conversation_exists}\n"
        f"Conversation excerpt:\n{conversation_excerpt}\n\n"
        f"Previous export scope: {previous_scope_label}\n\n"
        f"Latest user message:\n{query[:1500]}\n\n"
        "Return JSON now."
    )

    try:
        raw = await generate_once(
            "",
            temperature=0.0,
            max_tokens=96,
            messages=[
                {"role": "system", "content": _EXPORT_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except httpx.HTTPStatusError as error:
        logger.error("Export-intent LLM error %s: %s", error.response.status_code, (error.response.text or "")[:300])
        return ExportIntentOut(meta={"reason": "upstream_error", "upstream_status": error.response.status_code})
    except Exception as error:
        logger.warning("Export-intent classification failed: %s", error)
        return ExportIntentOut(meta={"reason": "classifier_unavailable"})

    result = _parse_export_intent(raw)
    if result.intent != "normal_chat":
        if _requests_conversation_export(query):
            result.scope = "conversation" if has_conversation else "previous_answer"
            result.meta["scope_override"] = "conversation_phrase"
        elif _is_export_format_followup(query) and previous_export_scope in _EXPORT_SCOPES:
            result.scope = previous_export_scope
            result.meta["scope_override"] = "previous_export_scope"
        elif not result.scope:
            result.scope = "previous_answer"
    result.meta["model"] = S.VLLM_MODEL
    return result


@router.post("/chatbot/api/follow-ups", response_model=FollowUpsOut, tags=["Chats"], summary="Suggest follow-up questions for the last turn")
async def follow_ups(inp: FollowUpsIn) -> FollowUpsOut:
    user_msg = (inp.user_message or "").strip()
    assistant_msg = (inp.assistant_message or "").strip()

    if not user_msg or not assistant_msg:
        return FollowUpsOut(follow_ups=[], meta={"reason": "empty_input"})

    grounding_mode = (inp.grounding_mode or "").strip().lower()
    if grounding_mode not in _FOLLOW_UPS_ALLOWED_MODES:
        return FollowUpsOut(
            follow_ups=[],
            meta={"reason": "unsupported_grounding_mode", "grounding_mode": grounding_mode or "unknown"},
        )
    if not inp.sources:
        return FollowUpsOut(
            follow_ups=[],
            meta={"reason": "no_cited_sources", "grounding_mode": grounding_mode},
        )

    language_hint = ""
    if inp.language:
        language_hint = f"\nUser's language: {inp.language}\n"

    history_block = ""
    if inp.history:
        recent = inp.history[-4:]
        lines: list[str] = []
        for msg in recent:
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content or role not in {"user", "assistant"}:
                continue
            lines.append(f"{role.capitalize()}: {content[:600]}")
        if lines:
            history_block = "Earlier conversation (for topic anchoring only):\n" + "\n".join(lines) + "\n\n"

    source_block = ""
    if inp.sources:
        labels: list[str] = []
        for source in inp.sources[:5]:
            title = (source.title or "").strip()
            project = (source.project or "").strip()
            label = " - ".join(part for part in (title, project) if part)
            if label:
                labels.append(f"- {label[:240]}")
        if labels:
            source_block = "Cited source labels (topic anchors only):\n" + "\n".join(labels) + "\n\n"

    prompt = (
        f"{_FOLLOW_UPS_SYSTEM_PROMPT}"
        f"{language_hint}\n"
        f"Grounding mode: {grounding_mode or 'unknown'}\n\n"
        f"{history_block}"
        f"{source_block}"
        f"User question:\n{user_msg[:1500]}\n\n"
        f"Assistant reply:\n{assistant_msg[:3000]}\n\n"
        "Return JSON now."
    )

    try:
        raw = await generate_once(prompt, temperature=0.3, max_tokens=180)
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:300]
        logger.error(f"Follow-ups LLM error {e.response.status_code}: {body}")
        return FollowUpsOut(follow_ups=[], meta={"upstream_status": e.response.status_code})
    except Exception as e:
        logger.warning(f"Follow-ups generation failed: {e}")
        return FollowUpsOut(follow_ups=[], meta={"error": str(e)[:200]})

    suggestions = _parse_follow_ups(raw)
    unfiltered = _parse_follow_ups(raw, enforce_policy=False)
    meta = {
        "model": S.VLLM_MODEL,
        "grounding_mode": grounding_mode or "unknown",
    }

    if len(suggestions) < len(unfiltered):
        meta["reason"] = "policy_filtered"
        meta["filtered_count"] = len(unfiltered) - len(suggestions)
    elif not suggestions:

        meta["reason"] = "no_useful_suggestions"
    return FollowUpsOut(follow_ups=suggestions, meta=meta)
