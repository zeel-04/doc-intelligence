from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


# -------------------------------------
# Utility schemas
# -------------------------------------
class BoundingBox(BaseModel):
    x0: float
    top: float
    x1: float
    bottom: float


class Line(BaseModel):
    """A single line of text with its bounding box."""

    text: str
    bounding_box: BoundingBox


# -------------------------------------
# Base Citation
# -------------------------------------
class BaseCitation(BaseModel):
    model_config = ConfigDict(title="Citation")


# -------------------------------------
# Generic Document schema
# -------------------------------------
class Document(BaseModel):
    uri: str
    content: BaseModel | None = None
    include_citations: bool = True
    extraction_mode: Enum


# -------------------------------------
# Extraction result
# -------------------------------------
class ExtractionResult(BaseModel):
    """Typed result from an extraction pipeline.

    Attributes:
        data: The extracted Pydantic model instance.
        metadata: Citation metadata dict, or None if citations are disabled.
    """

    data: Any
    metadata: dict[str, Any] | None = None


# -------------------------------------
# Extraction config
# -------------------------------------
class ExtractionConfig(BaseModel):
    include_citations: bool
