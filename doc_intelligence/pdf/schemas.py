"""Schemas for PDF document structure and extraction configuration."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import (
    BoundingBox,
    Document,
    ExtractionConfig,
    Line,
)


class Cell(BaseModel):
    """A single cell within a table."""

    text: str
    bounding_box: BoundingBox | None = None


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


ContentBlock = Annotated[TextBlock | TableBlock, Field(discriminator="block_type")]


class Page(BaseModel):
    """A single page containing an ordered list of content blocks."""

    blocks: list[ContentBlock]
    width: int | float
    height: int | float


class PDF(BaseModel):
    """A PDF document consisting of pages."""

    pages: list[Page] = Field(default_factory=list)


class PDFDocument(Document):
    """A PDF document with optional parsed content and extraction state."""

    content: PDF | None = None
    extraction_mode: Enum = PDFExtractionMode.SINGLE_PASS
    page_numbers: list[int] | None = None
    pass1_result: BaseModel | None = None
    pass2_page_map: dict[str, list[int]] | None = None


class PDFExtractionConfig(ExtractionConfig):
    """Configuration for a PDF extraction run."""

    extraction_mode: Enum
    page_numbers: list[int] | None = None


# ---------------------------------------------------------------------------
# Block utilities
# ---------------------------------------------------------------------------


def table_block_to_text_block(table: TableBlock) -> TextBlock:
    """Convert a TableBlock to a TextBlock for LLM consumption.

    Each row is rendered as a pipe-delimited line: ``| cell1 | cell2 | cell3 |``.
    The bounding box for each line is taken from the first cell in that row
    that has a bbox, falling back to the table's own bbox, and finally to a
    full-page placeholder if neither is available.

    Args:
        table: The TableBlock to convert.

    Returns:
        A TextBlock with one Line per table row.
    """
    _fallback_bbox = table.bounding_box or BoundingBox(
        x0=0.0, top=0.0, x1=1.0, bottom=1.0
    )
    lines: list[Line] = []
    for row in table.rows:
        text = "| " + " | ".join(cell.text for cell in row) + " |"
        bbox = next(
            (cell.bounding_box for cell in row if cell.bounding_box is not None),
            _fallback_bbox,
        )
        lines.append(Line(text=text, bounding_box=bbox))
    return TextBlock(bounding_box=table.bounding_box, lines=lines)


def blocks_to_lines(blocks: list[ContentBlock]) -> list[Line]:
    """Flatten all content blocks into a single ordered list of Lines.

    TextBlocks contribute their lines directly. TableBlocks are first
    converted via :func:`table_block_to_text_block`.

    Args:
        blocks: Ordered list of ContentBlock instances on a page.

    Returns:
        Flat list of Line instances preserving block order.
    """
    lines: list[Line] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            lines.extend(block.lines)
        elif isinstance(block, TableBlock):
            lines.extend(table_block_to_text_block(block).lines)
    return lines
