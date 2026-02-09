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
    content: type[PydanticModel] | None = None
    include_citations: bool = True
    extraction_mode: Enum
    response: type[PydanticModel] | None = None
    response_metadata: dict[str, Any] | None = None


# -------------------------------------
# Extraction config
# -------------------------------------
class ExtractionConfig(BaseModel):
    include_citations: bool
