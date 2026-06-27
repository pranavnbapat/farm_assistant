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
    # web_search_service), "brave" (key), "duckduckgo" (free, keyless),
    # "wikipedia" (free, keyless institutional/foundational source).
    WEB_SEARCH_PROVIDERS: str = "tavily,staan,brave,duckduckgo,wikipedia"
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
        "inrae.fr,wur.nl,teagasc.ie,ahdb.org.uk,extension.org,wikipedia.org"
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
