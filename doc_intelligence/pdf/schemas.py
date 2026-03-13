from enum import Enum

from pydantic import BaseModel, Field

from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import BoundingBox, Document, ExtractionConfig


class Line(BaseModel):
    text: str
    bounding_box: BoundingBox


class Page(BaseModel):
    lines: list[Line]
    width: int | float
    height: int | float


class PDF(BaseModel):
    pages: list[Page] = Field(default_factory=list)


class PDFDocument(Document):
    content: PDF | None = None
    extraction_mode: Enum = PDFExtractionMode.SINGLE_PASS
    pass1_result: BaseModel | None = None
    pass2_page_map: dict[str, list[int]] | None = None


class PDFExtractionConfig(ExtractionConfig):
    extraction_mode: Enum
    page_numbers: list[int] | None = None
