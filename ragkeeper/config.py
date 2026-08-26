from functools import lru_cache

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str

    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "langchain_docs"

    langchain_repo_url: str = "https://github.com/langchain-ai/langchain.git"
    langchain_repo_path: str = "./data/langchain_repo"
    docs_root_subdir: str = "docs"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = [str(err["loc"][0]) for err in exc.errors() if err["type"] == "missing"]
        if missing:
            raise SystemExit(
                f"Missing required config: {', '.join(missing)}. "
                "Copy .env.example to .env and fill it in."
            ) from exc
        raise
