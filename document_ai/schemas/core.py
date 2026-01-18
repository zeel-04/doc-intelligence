from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
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


# -------------------------------------
# Base Citation
# -------------------------------------
class BaseCitation(BaseModel):
    model_config = ConfigDict(title="Citation")


# -------------------------------------
# Generic Document schema
# -------------------------------------
class Document(BaseModel):
    document_type: str
    uri: str
    content: type[PydanticModel] | None = None
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
    document_type: str = "pdf"
    content: PDF | None = None
    llm_input: str | None = None
