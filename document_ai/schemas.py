from typing import TypeVar

from pydantic import BaseModel, Field, SerializeAsAny

Content = TypeVar("Content", bound=BaseModel)


class BoundingBox(BaseModel):
    x0: float
    top: float
    x1: float
    bottom: float


class Line(BaseModel):
    text: str
    bounding_box: BoundingBox


class Page(BaseModel):
    lines: list[Line]
    width: int | float
    height: int | float


class PDFDocument(BaseModel):
    pages: list[Page] = Field(default_factory=list)


class Document(BaseModel):
    document_type: str
    uri: str
    content: Content | None = None
    llm_input: list[str] | None = None
