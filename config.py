"""
Central configuration for the Financial Deep Research Agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ──────────────────────────────────────────────────────────────────
    # Supports OpenRouter (recommended) or direct OpenAI
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model to use via OpenRouter (change to any supported model)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "3000"))

    # ── Search ────────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # ── Financial Data ────────────────────────────────────────────────────────
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # ── RAG / Vector DB ───────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

    # ── Research Loop ─────────────────────────────────────────────────────────
    MIN_RESEARCH_STEPS: int = int(os.getenv("MIN_RESEARCH_STEPS", "5"))
    MAX_RESEARCH_STEPS: int = int(os.getenv("MAX_RESEARCH_STEPS", "15"))

    # ── Output ────────────────────────────────────────────────────────────────
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./reports")

    # ── Supported Sectors ─────────────────────────────────────────────────────
    SUPPORTED_SECTORS: list[str] = ["IT", "Pharma", "Finance", "Ecommerce", "Automotive", "Healthcare"]

    @classmethod
    def get_llm_api_key(cls) -> str:
        """Return whichever API key is configured."""
        return cls.OPENROUTER_API_KEY or cls.OPENAI_API_KEY

    @classmethod
    def get_llm_base_url(cls) -> str | None:
        """Return base URL for OpenRouter; None for direct OpenAI."""
        if cls.OPENROUTER_API_KEY:
            return cls.OPENROUTER_BASE_URL
        return None

    @classmethod
    def validate(cls) -> None:
        """Raise if critical config is missing."""
        if not cls.get_llm_api_key():
            raise EnvironmentError(
                "No LLM API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
            )
        if not cls.TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is required for web search. Set it in .env"
            )


config = Config()
