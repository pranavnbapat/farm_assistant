"""
Language detection and normalization utilities.
Lightweight implementation using regex heuristics and common word detection.
For production, consider using fastText or langdetect library.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("farm-assistant.language")

# Common words for major EU languages (simplified detection).
LANGUAGE_MARKERS = {
    "en": {"the", "and", "is", "are", "farm", "what", "how", "help", "my", "give", "me", "some", "crop", "rotation", "techniques", "in", "tabular", "format", "tell", "about", "please", "can", "you"},
    "de": {"der", "die", "das", "und", "ist", "bauernhof", "mein", "wie", "hilfe"},
    "fr": {"le", "la", "les", "et", "est", "mon", "ferme", "comment", "aide"},
    "es": {"el", "la", "los", "las", "y", "es", "mi", "granja", "cómo", "ayuda"},
    "it": {"il", "la", "e", "è", "mia", "fattoria", "come", "aiuto"},
    "nl": {"de", "het", "en", "is", "mijn", "boerderij", "hoe", "help"},
    "pl": {"w", "z", "i", "jest", "moja", "farma", "jak", "pomoc"},
    "ro": {"în", "și", "este", "ferma", "mea", "cum", "ajutor"},
    "pt": {"o", "a", "e", "é", "minha", "fazenda", "como", "ajuda"},
    "el": {"το", "και", "είναι", "φάρμα", "μου", "πώς", "βοήθεια"},
    "hu": {"a", "és", "van", "farm", "saját", "hogyan", "segítség"},
    "cs": {"v", "a", "je", "farma", "moje", "jak", "pomoc"},
    "sv": {"och", "är", "min", "gård", "hur", "hjälp"},
    "bg": {"в", "и", "е", "ферма", "моя", "как", "помощ"},
    "hr": {"u", "i", "je", "farma", "moja", "kako", "pomoć"},
    "da": {"og", "er", "min", "gård", "hvordan", "hjælp"},
    "fi": {"ja", "on", "minun", "maatila", "miten", "apua"},
    "sk": {"a", "je", "moja", "farma", "ako", "pomoc"},
    "lt": {"ir", "yra", "mano", "ūkis", "kaip", "pagalba"},
    "sl": {"in", "je", "moja", "kmetija", "kako", "pomoč"},
    "lv": {"un", "ir", "mana", "saimniecība", "kā", "palīdzība"},
    "et": {"ja", "on", "minu", "talu", "kuidas", "abi"},
    "mt": {"u", "hu", "tieghi", "bidwi", "kif", "ghajnuna"},
    "ga": {"agus", "tá", "mo", "feirm", "conas", "cabhair"},
}

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "ro": "Romanian",
    "pt": "Portuguese", "el": "Greek", "hu": "Hungarian", "cs": "Czech",
    "sv": "Swedish", "bg": "Bulgarian", "hr": "Croatian", "da": "Danish",
    "fi": "Finnish", "sk": "Slovak", "lt": "Lithuanian", "sl": "Slovenian",
    "lv": "Latvian", "et": "Estonian", "mt": "Maltese", "ga": "Irish",
}

EU_LANGUAGE_CODES = set(LANGUAGE_NAMES.keys())


def detect_language_confident(text: str) -> Optional[str]:
    """Return a language only when marker evidence is meaningful and unambiguous."""
    if not text or not text.strip():
        return None

    words = set(re.findall(r"\b[a-zA-Zà-ÿÀ-Ÿ]+\b", text.lower()))
    if not words:
        return None

    informative_markers = {
        language: {marker for marker in markers if len(marker) > 1}
        for language, markers in LANGUAGE_MARKERS.items()
    }
    marker_languages: dict[str, set[str]] = {}
    for language, markers in informative_markers.items():
        for marker in markers:
            marker_languages.setdefault(marker, set()).add(language)

    overlaps = {
        language: words & markers
        for language, markers in informative_markers.items()
    }
    overlaps = {language: overlap for language, overlap in overlaps.items() if overlap}
    if not overlaps:
        return None

    best_score = max(len(overlap) for overlap in overlaps.values())
    leaders = [language for language, overlap in overlaps.items() if len(overlap) == best_score]
    if len(leaders) != 1:
        return None

    leader = leaders[0]
    if best_score >= 2:
        return leader

    marker = next(iter(overlaps[leader]))
    if len(marker) >= 3 and marker_languages.get(marker) == {leader}:
        return leader
    return None


def detect_language(text: str) -> str:
    """Detect a supported language, defaulting uncertain or ambiguous text to English."""
    return detect_language_confident(text) or "en"


def get_language_name(lang_code: str) -> str:
    """Get the English name for a language code."""
    return LANGUAGE_NAMES.get((lang_code or "").lower(), "Unknown")
def is_eu_language(lang_code: str) -> bool:
    """Check if the language code is one of the 24 EU official languages."""
    return lang_code.lower() in EU_LANGUAGE_CODES


def normalize_language_code(lang_code: str) -> str:
    """Normalize language code to ISO 639-1 (2 letters)."""
    if not lang_code:
        return "en"
    
    code = lang_code.lower().strip()
    
    # Handle common variants
    variants = {
        "eng": "en",
        "deu": "de", "ger": "de",
        "fra": "fr", "fre": "fr",
        "spa": "es",
        "ita": "it",
        "nld": "nl", "dut": "nl",
        "pol": "pl",
        "ron": "ro", "rum": "ro",
        "por": "pt",
        "ell": "el", "gre": "el",
        "hun": "hu",
        "ces": "cs", "cze": "cs",
        "swe": "sv",
        "bul": "bg",
        "hrv": "hr",
        "dan": "da",
        "fin": "fi",
        "slk": "sk", "slo": "sk",
        "lit": "lt",
        "slv": "sl",
        "lav": "lv",
        "est": "et",
        "mlt": "mt",
        "gle": "ga", "iri": "ga",
    }
    
    if code in variants:
        return variants[code]
    
    if code in EU_LANGUAGE_CODES:
        return code
    
    return "en"


def extract_language_preference(text: str) -> Optional[str]:
    """
    Extract if user explicitly requests a specific language.
    E.g., "answer in French", "responde en español"
    
    Returns language code or None.
    """
    text_lower = text.lower()
    
    # Pattern: "in/at [language]" or "en [language]"
    patterns = [
        r'(?:in|answer in|respond in|speak in)\s+([a-z]+)',
        r'(?:en|responde en|contesta en)\s+([a-z]+)',  # Spanish
        r'(?:auf|antworte auf)\s+([a-z]+)',  # German
        r'dans?\s+([a-z]+)',  # French
    ]
    
    language_keywords = {
        "english": "en", "eng": "en",
        "german": "de", "deutsch": "de", "allemand": "de", "alemán": "de",
        "french": "fr", "français": "fr", "francais": "fr", "frances": "fr",
        "spanish": "es", "español": "es", "espanol": "es",
        "italian": "it", "italiano": "it",
        "dutch": "nl", "nederlands": "nl", "holland": "nl",
        "polish": "pl", "polski": "pl",
        "romanian": "ro", "română": "ro", "romana": "ro",
        "portuguese": "pt", "português": "pt", "portugues": "pt",
        "greek": "el", "ελληνικά": "el",
        "hungarian": "hu", "magyar": "hu",
        "czech": "cs", "čeština": "cs", "cestina": "cs",
        "swedish": "sv", "svenska": "sv",
        "bulgarian": "bg", "български": "bg",
        "croatian": "hr", "hrvatski": "hr",
        "danish": "da", "dansk": "da",
        "finnish": "fi", "suomi": "fi",
        "slovak": "sk", "slovenčina": "sk", "slovencina": "sk",
        "lithuanian": "lt", "lietuvių": "lt", "lietuviu": "lt",
        "slovenian": "sl", "slovenščina": "sl", "slovenscina": "sl",
        "latvian": "lv", "latviešu": "lv", "latviesu": "lv",
        "estonian": "et", "eesti": "et",
        "maltese": "mt", "malti": "mt",
        "irish": "ga", "gaeilge": "ga",
    }
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            lang_keyword = match.group(1).strip()
            if lang_keyword in language_keywords:
                return language_keywords[lang_keyword]
    
    return None
