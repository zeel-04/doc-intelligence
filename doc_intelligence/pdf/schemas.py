"""Schemas for PDF document structure and extraction request."""

from pydantic import BaseModel, Field

from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import (
    Document,
    ExtractionRequest,
    Page,
)


class PDF(BaseModel):
    """A PDF document consisting of pages."""

    pages: list[Page] = Field(default_factory=list)


class PDFDocument(Document):
    """A PDF document with optional parsed content.

    Pure data container — holds only identity and parsed content.
    Extraction intent (mode, citations, page filtering) lives on
    :class:`PDFExtractionRequest`, not here.
    """

    content: PDF | None = None


class PDFExtractionRequest(ExtractionRequest):
    """PDF-specific extraction request.

    Extends :class:`ExtractionRequest` with PDF-specific options.

    Attributes:
        extraction_mode: ``SINGLE_PASS`` or ``MULTI_PASS``.
        page_numbers: Optional list of 0-indexed page numbers to
            restrict extraction to.
    """

    extraction_mode: PDFExtractionMode = PDFExtractionMode.SINGLE_PASS
    page_numbers: list[int] | None = None
