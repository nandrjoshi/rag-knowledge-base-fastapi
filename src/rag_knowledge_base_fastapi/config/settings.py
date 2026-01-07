from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized application configuration loaded from environment variables.

    For local development, values can be provided via a `.env` file.
    """

    # ---OpenAI ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPRNAI_CHAT_MODEL")
    openai_embed_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBED_MODEL")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/rag_kb",
        alias="DATABASE_URL",
    )

    # --- RAG Defaults ---
    rag_top_k_default: int = Field(default=5, alias="RAG_TOP_K_DEFAULT")
    chunk_size_chars: int = Field(default=1000, alias="CHUNK_SIZE_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP_CHARS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings: Settings = Settings()