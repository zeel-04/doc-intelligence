"""Application-level configuration for doc_intelligence."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DocIntelligenceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOC_INTEL_", env_file=".env", extra="ignore"
    )

    # LLM defaults
    default_llm_model: str = "gpt-4o-mini"

    # Limits
    max_pdf_size_mb: float = Field(default=10.0)
    max_pdf_pages: int = Field(default=100)
    max_schema_depth: int = Field(default=5)

    # Async (used in Phase 4, declared here for single source of truth)
    max_concurrent_documents: int = Field(default=10)
    max_concurrent_pages: int = Field(default=4)
    max_concurrent_regions: int = Field(default=8)


settings = DocIntelligenceConfig()
