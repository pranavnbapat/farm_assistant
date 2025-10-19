from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# load_dotenv(ENV_PATH)

class Settings(BaseSettings):
    # --- App ---
    LOG_LEVEL: str = Field("INFO")
    APP_TITLE: str = "Farm Assistant RAG"
    APP_VERSION: str = "0.1.0"
    ENABLE_DOCS: bool = False
    ENABLE_REDOC: bool = False

    # --- OpenSearch proxy ---
    OPENSEARCH_API_URL: str
    OPENSEARCH_API_USR: str | None = None
    OPENSEARCH_API_PWD: str | None = None
    VERIFY_SSL: bool = True
    OS_API_PATH: str = "/neural_search_relevant"

    # --- LLM / Ollama ---
    OLLAMA_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "deepseek-llm:7b"
    MAX_TOKENS: int = -1
    TEMPERATURE: float = 0.4
    NUM_CTX: int = 4096
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 24000

    # --- Intent Router ---
    INTENT_ROUTER_URL: str = "https://intent-router.nexavion.com/intent-router"

    # --- Search defaults ---
    SEARCH_MODEL: str = "msmarco"
    SEARCH_INCLUDE_FULLTEXT: bool = True
    SEARCH_SORT_BY: str = "score_desc"
    SEARCH_PAGE: int = 1
    SEARCH_K: int = 0
    SEARCH_DEV: bool = False

    GARBAGE_OK_MAX: float = Field(0.10, description="Treat as clean if garbage_score below this")
    CONF_STRONG: float = Field(0.80, description="Treat as confident if above this")

    # pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # light normalisation
    def normalise(self) -> "Settings":
        # Empty creds -> None
        if self.OPENSEARCH_API_USR == "":
            self.OPENSEARCH_API_USR = None
        if self.OPENSEARCH_API_PWD == "":
            self.OPENSEARCH_API_PWD = None

        # Trim URLs
        if self.OPENSEARCH_API_URL:
            self.OPENSEARCH_API_URL = self.OPENSEARCH_API_URL.rstrip("/")
        if self.OLLAMA_URL:
            self.OLLAMA_URL = self.OLLAMA_URL.rstrip("/")

        # Clamp/validate knobs
        # self.TEMPERATURE = max(0.0, min(2.0, float(self.TEMPERATURE)))
        # if not (self.MAX_TOKENS == -1 or self.MAX_TOKENS >= 1):
        #     self.MAX_TOKENS = -1

        # self.GARBAGE_OK_MAX = max(0.0, min(1.0, float(self.GARBAGE_OK_MAX)))
        # self.CONF_STRONG = max(0.0, min(1.0, float(self.CONF_STRONG)))

        # Sanity: context sizes
        # self.NUM_CTX = max(256, int(self.NUM_CTX))
        # self.MAX_CONTEXT_CHARS = max(2000, int(self.MAX_CONTEXT_CHARS))

        return self

@lru_cache()
def get_settings() -> Settings:
    return Settings().normalise()
