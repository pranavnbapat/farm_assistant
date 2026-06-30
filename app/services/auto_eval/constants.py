EU_LANGUAGES = [
    {"code": "bg", "name": "Bulgarian"},
    {"code": "hr", "name": "Croatian"},
    {"code": "cs", "name": "Czech"},
    {"code": "da", "name": "Danish"},
    {"code": "nl", "name": "Dutch"},
    {"code": "en", "name": "English"},
    {"code": "et", "name": "Estonian"},
    {"code": "fi", "name": "Finnish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "el", "name": "Greek"},
    {"code": "hu", "name": "Hungarian"},
    {"code": "ga", "name": "Irish"},
    {"code": "it", "name": "Italian"},
    {"code": "lv", "name": "Latvian"},
    {"code": "lt", "name": "Lithuanian"},
    {"code": "mt", "name": "Maltese"},
    {"code": "pl", "name": "Polish"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "ro", "name": "Romanian"},
    {"code": "sk", "name": "Slovak"},
    {"code": "sl", "name": "Slovenian"},
    {"code": "es", "name": "Spanish"},
    {"code": "sv", "name": "Swedish"},
]

EU_LANGUAGE_BY_CODE = {language["code"]: language for language in EU_LANGUAGES}
DEFAULT_VARIANTS = [
    {"id": "qwen3", "backend": "um_qwen3", "label": "Qwen3"},
    {"id": "mistral", "backend": "euf_chatbot_tnods", "label": "Mistral"},
    {"id": "eurollm", "backend": "eurollm", "label": "EuroLLM"},
]
CRITERIA = [
    "relevance",
    "trustworthiness",
    "clarity",
    "usefulness",
    "uncertainty_handling",
]
