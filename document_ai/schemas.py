from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


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


class PDF(BaseModel):
    pages: list[Page] = Field(default_factory=list)


# Generic Document schema
class Document(BaseModel):
    document_type: str
    uri: str
    content: PydanticModel | None = None
    llm_input: Any | None = None


class PDFDocument(Document):
    document_type: Literal["pdf"] = "pdf"
    content: PDF | None = None
    llm_input: list[str] | str | None = None

class Mode(BaseModel):
    paginated: bool
    include_line_numbers: bool