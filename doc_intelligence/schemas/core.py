"""Core schemas shared across all document types."""

from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

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


class Cell(BaseModel):
    """A single cell within a table."""

    text: str
    bounding_box: BoundingBox | None = None


# -------------------------------------
# Content blocks
# -------------------------------------
class TextBlock(BaseModel):
    """A contiguous region of text lines detected on a page."""

    block_type: Literal["text"] = "text"
    bounding_box: BoundingBox | None = None
    lines: list[Line]


class TableBlock(BaseModel):
    """A structured table region with rows of cells."""

    block_type: Literal["table"] = "table"
    bounding_box: BoundingBox | None = None
    rows: list[list[Cell]]


class ImageBlock(BaseModel):
    """An image region detected on a page.

    Skipped during formatting for now — placeholder for future VLM
    description or embedded image support.
    """

    block_type: Literal["image"] = "image"
    bounding_box: BoundingBox | None = None
    description: str | None = None
    image_uri: str | None = None


class ChartBlock(BaseModel):
    """A chart/figure region detected on a page.

    Skipped during formatting for now — placeholder for future chart
    data extraction or VLM description support.
    """

    block_type: Literal["chart"] = "chart"
    bounding_box: BoundingBox | None = None
    description: str | None = None
    data_table: list[list[Cell]] | None = None
    image_uri: str | None = None


ContentBlock = Annotated[
    TextBlock | TableBlock | ImageBlock | ChartBlock,
    Field(discriminator="block_type"),
]


# -------------------------------------
# Page
# -------------------------------------
class Page(BaseModel):
    """A single page containing an ordered list of content blocks."""

    blocks: list[ContentBlock]
    width: int | float
    height: int | float


# -------------------------------------
# Base Citation
# -------------------------------------
class BaseCitation(BaseModel):
    model_config = ConfigDict(title="Citation")


# -------------------------------------
# Generic Document schema
# -------------------------------------
class Document(BaseModel):
    """Base document — holds only identity and parsed content."""

    uri: str
    content: BaseModel | None = None


# -------------------------------------
# Extraction request
# -------------------------------------
class ExtractionRequest(BaseModel):
    """Base extraction request — carries caller intent, not document state.

    Attributes:
        uri: Path or URL of the document to process.
        response_format: Pydantic model class describing the expected schema.
        include_citations: Whether to include citation metadata.
        llm_config: Additional LLM configuration forwarded per-call.
    """

    uri: str
    response_format: type[BaseModel]
    include_citations: bool = True
    llm_config: dict[str, Any] | None = None


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
