"""Schemas for PDF document structure and extraction configuration."""

from enum import Enum

from pydantic import BaseModel, Field

from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import (
    Document,
    ExtractionConfig,
    Page,
)


class PDF(BaseModel):
    """A PDF document consisting of pages."""

    pages: list[Page] = Field(default_factory=list)


class PDFDocument(Document):
    """A PDF document with optional parsed content and extraction state."""

    content: PDF | None = None
    extraction_mode: Enum = PDFExtractionMode.SINGLE_PASS
    page_numbers: list[int] | None = None
    pass1_result: BaseModel | None = None
    pass2_page_map: dict[str, list[int]] | None = None


class PDFExtractionConfig(ExtractionConfig):
    """Configuration for a PDF extraction run."""

    extraction_mode: Enum
    page_numbers: list[int] | None = None
