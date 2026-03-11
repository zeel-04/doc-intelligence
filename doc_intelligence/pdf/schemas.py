from enum import Enum

from pydantic import BaseModel, Field

from ..schemas.core import BoundingBox, Document, ExtractionConfig
from .types import PDFExtractionMode


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
    content: PDF | None = None  # type: ignore[override]
    extraction_mode: Enum = PDFExtractionMode.SINGLE_PASS


class PDFExtractionConfig(ExtractionConfig):
    extraction_mode: Enum
    page_numbers: list[int] | None = None
