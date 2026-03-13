"""Application-level configuration for doc_intelligence."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DocIntelligenceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOC_INTEL_", env_file=".env", extra="ignore"
    )

    # Default models — overridable via DOC_INTEL_<PROVIDER>_DEFAULT_MODEL env vars
    openai_default_model: str = Field(default="gpt-4o-mini")
    anthropic_default_model: str = Field(default="claude-sonnet-4-20250514")
    gemini_default_model: str = Field(default="gemini-2.0-flash")
    ollama_default_model: str = Field(default="llama3.2")

    # Limits
    max_pdf_size_mb: float = Field(default=10.0)
    max_pdf_pages: int = Field(default=100)
    max_schema_depth: int = Field(default=5)

    # Async (used in Phase 4, declared here for single source of truth)
    max_concurrent_documents: int = Field(default=10)
    max_concurrent_pages: int = Field(default=4)
    max_concurrent_regions: int = Field(default=8)


settings = DocIntelligenceConfig()
