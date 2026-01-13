from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


# -------------------------------------
# Utility schemas
# -------------------------------------
class BoundingBox(BaseModel):
    x0: float
    top: float
    x1: float
    bottom: float


class Citation(TypedDict):
    """Citation dict for page and line number references."""

    page: int
    lines: list[int]


class Mode(BaseModel):
    include_line_numbers: bool = Field(default=True)


# -------------------------------------
# Generic Document schema
# -------------------------------------
class Document(BaseModel):
    document_type: str
    uri: str
    content: PydanticModel | None = None
    llm_input: Any | None = None


# -------------------------------------
# PDF schema
# -------------------------------------
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
    document_type: Literal["pdf"] = "pdf"
    content: PDF | None = None
    llm_input: list[str] | None = None
