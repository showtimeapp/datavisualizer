"""
Configuration — supports OpenAI and Gemini as LLM providers.
Set via environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from typing import Literal, Optional


class Settings(BaseSettings):
    # --- LLM Provider ---
    LLM_PROVIDER: Literal["openai", "gemini"] = "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # --- App ---
    MAX_UPLOAD_SIZE_MB: int = 50
    DATASETS_DIR: str = "datasets"
    UPLOADS_DIR: str = "uploads"

    class Config:
        env_file = ".env"


settings = Settings()
