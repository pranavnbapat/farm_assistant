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

    # --- LLM / vLLM ---
    VLLM_URL: str = "http://localhost:8000"
    VLLM_MODEL: str = "qwen3-30b-a3b-awq"
    VLLM_API_KEY: str | None = None
    RUNPOD_VLLM_HOST: str = ""

    MAX_TOKENS: int = 768
    MAX_OUTPUT_TOKENS: int = 768
    MAX_INPUT_TOKENS: int = 3000
    MAX_USER_INPUT_TOKENS: int = 1200
    TEMPERATURE: float = 0.4
    NUM_CTX: int = 4096
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 24000

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

        return self

@lru_cache()
def get_settings() -> Settings:
    return Settings().normalise()
