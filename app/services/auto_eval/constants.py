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
# Single source of truth for the judging rubric. To change wording, tighten a
# boundary, or split a criterion into sub-criteria, edit the `definition` here —
# the judge prompt, the JSON schema hint, and the response parser are all built
# from this list. `key` is what the judge emits in JSON; `field` is the fixed
# JudgeResult/DB column the winner maps to. Adding a brand-new criterion with its
# own winner column also needs a JudgeResult field + a Django migration; adding
# sub-criteria that only live in the free-form `scores`/`rationales` JSON does not.
CRITERIA = [
    {
        "key": "relevance",
        "field": "relevant",
        "definition": (
            "does the answer actually address THIS question (its crop/livestock/policy/region and "
            "intent)? Off-topic, generic, or partial answers score low. For a deliberately "
            "non-agriculture question, correctly recognising it is out of scope is relevant."
        ),
    },
    {
        "key": "trustworthiness",
        "field": "most_trustworthy",
        "definition": (
            "are the claims factually correct and grounded in the answer's cited sources, with no "
            "fabricated facts, figures, citations, or invented EU rules? Unsupported or wrong claims score low."
        ),
    },
    {
        "key": "clarity",
        "field": "clearest",
        "definition": (
            "is it well-structured, unambiguous, and easy for a farmer or advisor to follow? "
            "Reward plain, organised explanations; penalise rambling or confusing text."
        ),
    },
    {
        "key": "usefulness",
        "field": "most_useful",
        "definition": (
            "how practically actionable is it for the asker — concrete, safe, applicable steps or "
            "guidance versus vague generalities?"
        ),
    },
    {
        "key": "uncertainty_handling",
        "field": "handled_uncertainty_best",
        "definition": (
            "when evidence is missing, conflicting, or the question rests on a false premise, does the "
            "answer acknowledge limits, avoid overclaiming, and ask for/flag what is needed? "
            "Confidently wrong answers score low."
        ),
    },
]

BEST_OVERALL_DEFINITION = (
    "taking all five criteria together, which single answer is best overall for this question."
)
