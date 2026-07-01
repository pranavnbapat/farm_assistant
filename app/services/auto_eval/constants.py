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

# Sub-topic pools that steer each base question to a DIFFERENT subject — without these the
# model defaults to the same go-to question every time. One hint is assigned per base question
# (sampled without immediate repeats), and injected into the generation prompt.
AGRICULTURE_SUBTOPICS = [
    "soil health and nutrient management", "crop rotation and cover crops",
    "integrated pest and disease management", "irrigation and water-use efficiency",
    "organic farming certification", "CAP subsidies and direct payments",
    "precision farming and sensors", "climate adaptation and drought resilience",
    "agroforestry and hedgerows", "livestock welfare and housing",
    "dairy herd management", "pasture and grazing systems",
    "fertiliser regulation and nitrate directives", "pollinators and biodiversity",
    "viticulture and wine production", "horticulture and greenhouses",
    "farm machinery and automation", "renewable energy on farms (solar, biogas)",
    "soil erosion and conservation tillage", "carbon sequestration and carbon farming",
    "food safety and traceability", "post-harvest storage and losses",
    "agri-environment schemes and rewilding", "young farmers and farm succession",
    "plant breeding and seed varieties", "aquaculture and inland fisheries",
    "forestry management and timber", "manure and slurry management",
    "weed control and herbicide reduction", "farm economics and risk management",
    "short supply chains and farmers' markets", "rural broadband and digital advisory",
]
NON_AGRICULTURE_TOPICS = [
    "world capital cities", "ancient Roman history", "rules of football",
    "famous movies and directors", "baking and dessert recipes",
    "Python programming basics", "the solar system and planets",
    "classical music composers", "symptoms of the common cold",
    "personal budgeting and saving", "tourist attractions in Asia",
    "Shakespeare's plays", "basic algebra and equations",
    "how smartphones work", "Olympic Games history",
    "popular video games", "famous painters and art movements",
    "world religions and festivals", "car engine maintenance", "chess strategy",
]

# A second dimension combined with the sub-topic to widen the question space (~8x) so a
# large batch (e.g. 1000) does not collapse to a few phrasings per sub-topic.
QUESTION_ANGLES = [
    "a practical how-to", "a specific EU regulation or directive", "a trade-off or comparison",
    "a troubleshooting scenario", "a recent policy or market change", "a cost or economic angle",
    "a sustainability or environmental angle", "a beginner-level definition",
    "a best-practice recommendation", "a data or measurement question",
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
