# app/config.py

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Environment selector ---
    FA_ENV: str = Field("local")  # "local" | "dev" | "prd"

    # --- App ---
    LOG_LEVEL: str = Field("INFO")
    APP_TITLE: str = "Farm Assistant RAG"
    APP_VERSION: str = "0.1.0"
    ENABLE_DOCS: bool = False
    ENABLE_REDOC: bool = False

    CHAT_BACKEND_URL: str = ""

    # --- OpenSearch proxy ---
    OPENSEARCH_API_URL: str
    OPENSEARCH_API_USR: str | None = None
    OPENSEARCH_API_PWD: str | None = None
    VERIFY_SSL: bool = True
    OS_API_PATH: str = "/neural_search_relevant"
    OS_RAG_API_PATH: str = "/llm_retrieve"

    # --- LLM / vLLM ---
    VLLM_URL: str = "http://localhost:8000"
    VLLM_MODEL: str = "qwen3-30b-a3b-awq"
    VLLM_API_KEY: str | None = None
    RUNPOD_VLLM_HOST: str = ""
    RUNPOD_VLLM_VISION_HOST: str = ""
    VLLM_VISION_MODEL: str = ""
    VLLM_VISION_API_KEY: str | None = None

    MAX_TOKENS: int = 768
    MAX_OUTPUT_TOKENS: int = 768
    MAX_INPUT_TOKENS: int = 3000
    MAX_USER_INPUT_TOKENS: int = 1200
    TEMPERATURE: float = 0.4
    NUM_CTX: int = 4096
    TOP_P: float = 0.9
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 24000
    RETRIEVAL_CANDIDATE_K: int = 10
    RETRIEVAL_MIN_SCORE: float = 1.0
    # Lexical-relevance floor (0..1 from estimate_retrieval_quality). Below this, the
    # retrieved KO contexts are dropped entirely and the turn is treated as "found
    # nothing" (then general fallback or, if enabled, web fallback).
    RETRIEVAL_DROP_THRESHOLD: float = 0.15

    # --- Retrieval relevance gate mode ---
    # "overlap" (default): judge KO sufficiency by lexical word-overlap (estimate_retrieval_quality).
    # "semantic": use scout's per-chunk `semantic_score` (content_embedding cosine) instead —
    #   calibrated and cross-query comparable. Requests scout with include_semantic_score=true.
    # Falls back to overlap automatically if items carry no semantic_score.
    RELEVANCE_MODE: str = "overlap"
    # Semantic-mode thresholds on scout's neural (1+cos)/2 scale. Calibrated on msmarco
    # over 9 probes: on-topic best-chunk scores >=0.91, off-topic <=0.87 (a ~0.04 dead
    # zone). These sit in/above that zone. NOTE: msmarco-specific and narrow-margin —
    # re-calibrate per model (e.g. minilm) and from ClickHouse /llm_retrieve logs.
    SEMANTIC_DROP_THRESHOLD: float = 0.88   # best chunk below this: drop all KO context
    SEMANTIC_WEB_THRESHOLD: float = 0.90    # best chunk below this: weak -> web fallback

    # --- LLM provider routing (additive) ---
    # "vllm" (default) keeps the existing OpenAI-compatible /v1/chat/completions
    # path unchanged — used by Qwen3, EuroLLM and the OpenAI API instance.
    # "anthropic" routes generation through the Anthropic Messages API instead.
    LLM_PROVIDER: str = Field("vllm")

    # --- Anthropic (only read when LLM_PROVIDER=anthropic) ---
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    ANTHROPIC_MAX_TOKENS: int = 1024

    # Force OpenAI GPT-5 param style (max_completion_tokens, no temperature/top_p).
    # Auto-detected for models named "gpt-5*"; set true to force it otherwise.
    OPENAI_GPT5_PARAM_STYLE: bool = False

    # --- Automated Arena evaluation ---
    AUTOMATION_ENABLED: bool = False
    AUTOMATION_TOKEN: str | None = None
    AUTOMATION_SERVICE_EMAIL: str = ""
    AUTOMATION_SERVICE_PASSWORD: str = ""
    AUTOMATION_EXPERIMENT_ID: str = "automated_eval"
    AUTOMATION_TOPIC_RATIO: str = "3:1"
    # One base question localized into all 24 EU languages = 24 comparison runs per cycle.
    AUTOMATION_BASE_QUESTION_COUNT: int = 1
    AUTOMATION_MIN_ANSWERS: int = 2
    AUTOMATION_MAX_CONCURRENCY: int = 3
    AUTOMATION_REQUEST_TIMEOUT: float = 180.0
    # Token budget for ONE base question localized into all requested languages.
    AUTOMATION_QUESTION_MAX_TOKENS: int = 3000
    FARM_ASSISTANT_UM_QWEN3_URL: str = "https://farm-assistant.nexavion.com"
    FARM_ASSISTANT_UM_QWEN3_MODEL_NAME: str = "qwen3-30b-a3b-awq"
    FARM_ASSISTANT_EUROLLM_URL: str = "http://127.0.0.1:18005"
    FARM_ASSISTANT_EUROLLM_MODEL_NAME: str = "utter-project/EuroLLM-9B-Instruct"
    EUF_CHATBOT_API_URL: str = "https://farm-assistant.tnods.nl"
    EUF_CHATBOT_API_KEY: str = ""
    EUF_CHATBOT_ARENA_UUID: str = "45b75f62-3fa3-4b18-8593-1411f110a98e"
    EUF_CHATBOT_MODEL_NAME: str = "azure_ai/mistral-small-2503"
    JUDGE_OPENAI_API_KEY: str | None = None
    JUDGE_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    JUDGE_OPENAI_MODEL: str = "gpt-5.4-mini"
    JUDGE_ANTHROPIC_API_KEY: str | None = None
    JUDGE_ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    JUDGE_MAX_TOKENS: int = 1400
    JUDGE_TEMPERATURE: float = 0.0

    # --- Web-search fallback (additive, default OFF) ---
    # When enabled, the `normal` retrieval turn searches a trusted allowlist of
    # agriculture/forestry web sources whenever internal OpenSearch retrieval is
    # empty or weak, and feeds the extracted passages to the LLM as cited grounding.
    # The backend performs the search/extraction — the model never browses directly.
    WEB_FALLBACK_ENABLED: bool = False
    # Ordered provider fallback chain, tried left to right. A provider is skipped
    # when its required API key is missing, and the chain advances to the next one
    # on any error (quota/rate-limit/network) or when it returns no allowlisted hits.
    # Supported: "tavily" (key), "staan" (key, EU-sovereign, see note in
    # web_search_service), "brave" (key), "duckduckgo" (free, keyless).
    WEB_SEARCH_PROVIDERS: str = "tavily,staan,brave,duckduckgo"
    STAAN_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    BRAVE_API_KEY: str | None = None
    # Internal-retrieval quality below this (token-overlap, 0..1) triggers a web
    # fallback even when some KO results exist. The pre-existing <0.15 hard-drop in
    # ask.py remains the "empty" trigger; this is the higher "weak" trigger.
    WEB_FALLBACK_QUALITY_THRESHOLD: float = 0.35
    WEB_FALLBACK_MAX_RESULTS: int = 4
    WEB_FALLBACK_MAX_CHARS: int = 6000   # sub-budget carved out of MAX_CONTEXT_CHARS
    WEB_FETCH_TIMEOUT: float = 6.0
    WEB_TRUSTED_DOMAINS: str = (
        "fao.org,europa.eu,ec.europa.eu,efsa.europa.eu,eppo.int,oecd.org,"
        "inrae.fr,wur.nl,teagasc.ie,ahdb.org.uk"
    )

    # pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore',
    )

    # light normalisation
    def normalise(self) -> "Settings":
        # Empty creds -> None
        if self.OPENSEARCH_API_USR == "":
            self.OPENSEARCH_API_USR = None
        if self.OPENSEARCH_API_PWD == "":
            self.OPENSEARCH_API_PWD = None

        backend_by_env = {
            "local": "http://127.0.0.1:8000",
            "dev": "https://backend-admin.dev.farmbook.ugent.be",
            "prd": "https://backend-admin.prd.farmbook.ugent.be",
        }

        if not self.CHAT_BACKEND_URL:
            env = (self.FA_ENV or "local").lower()
            self.CHAT_BACKEND_URL = backend_by_env.get(env, backend_by_env["local"])

        if self.CHAT_BACKEND_URL:
            self.CHAT_BACKEND_URL = self.CHAT_BACKEND_URL.rstrip("/")

        # Trim URLs
        if self.OPENSEARCH_API_URL:
            self.OPENSEARCH_API_URL = self.OPENSEARCH_API_URL.rstrip("/")

        if self.VLLM_URL == "http://localhost:8000" and self.RUNPOD_VLLM_HOST:
            self.VLLM_URL = self.RUNPOD_VLLM_HOST.rstrip("/")
        elif self.VLLM_URL:
            self.VLLM_URL = self.VLLM_URL.rstrip("/")

        if self.RUNPOD_VLLM_VISION_HOST:
            self.RUNPOD_VLLM_VISION_HOST = self.RUNPOD_VLLM_VISION_HOST.rstrip("/")

        return self

    def web_trusted_domains_list(self) -> list[str]:
        """Parse WEB_TRUSTED_DOMAINS (csv) into a clean, lowercased domain list."""
        raw = self.WEB_TRUSTED_DOMAINS or ""
        return [d.strip().lower().lstrip(".") for d in raw.split(",") if d.strip()]

    def web_search_providers_list(self) -> list[str]:
        """Parse WEB_SEARCH_PROVIDERS (csv) into an ordered, lowercased provider chain."""
        raw = self.WEB_SEARCH_PROVIDERS or ""
        return [p.strip().lower() for p in raw.split(",") if p.strip()]

@lru_cache()
def get_settings() -> Settings:
    return Settings().normalise()
