"""Tests for pdf.schemas and core schema types."""

import pytest
from pydantic import BaseModel, ValidationError

from doc_intelligence.pdf.schemas import (
    PDF,
    PDFDocument,
    PDFExtractionRequest,
)
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import (
    BoundingBox,
    Cell,
    ChartBlock,
    ContentBlock,
    ImageBlock,
    Line,
    Page,
    TableBlock,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Line
# ---------------------------------------------------------------------------
class TestLine:
    def test_construction(self, sample_bbox: BoundingBox):
        line = Line(text="hello world", bounding_box=sample_bbox)
        assert line.text == "hello world"
        assert line.bounding_box == sample_bbox

    def test_missing_text_raises(self, sample_bbox: BoundingBox):
        with pytest.raises(ValidationError):
            Line(bounding_box=sample_bbox)  # type: ignore[call-arg]

    def test_missing_bbox_raises(self):
        with pytest.raises(ValidationError):
            Line(text="hello")  # type: ignore[call-arg]

    def test_model_dump(self, sample_bbox: BoundingBox):
        line = Line(text="hello", bounding_box=sample_bbox)
        data = line.model_dump()
        assert data["text"] == "hello"
        assert data["bounding_box"] == sample_bbox.model_dump()


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------
class TestCell:
    def test_construction_with_bbox(self, sample_bbox: BoundingBox):
        cell = Cell(text="value", bounding_box=sample_bbox)
        assert cell.text == "value"
        assert cell.bounding_box == sample_bbox

    def test_construction_without_bbox(self):
        cell = Cell(text="value")
        assert cell.text == "value"
        assert cell.bounding_box is None

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            Cell()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TextBlock
# ---------------------------------------------------------------------------
class TestTextBlock:
    def test_construction(self, sample_lines: list[Line]):
        block = TextBlock(lines=sample_lines)
        assert block.block_type == "text"
        assert block.lines == sample_lines
        assert block.bounding_box is None

    def test_with_bounding_box(
        self, sample_lines: list[Line], sample_bbox: BoundingBox
    ):
        block = TextBlock(lines=sample_lines, bounding_box=sample_bbox)
        assert block.bounding_box == sample_bbox

    def test_empty_lines(self):
        block = TextBlock(lines=[])
        assert block.lines == []

    def test_block_type_is_literal(self, sample_lines: list[Line]):
        block = TextBlock(lines=sample_lines)
        assert block.block_type == "text"

    def test_missing_lines_raises(self):
        with pytest.raises(ValidationError):
            TextBlock()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TableBlock
# ---------------------------------------------------------------------------
class TestTableBlock:
    def test_construction(self):
        rows = [[Cell(text="a"), Cell(text="b")], [Cell(text="c"), Cell(text="d")]]
        block = TableBlock(rows=rows)
        assert block.block_type == "table"
        assert len(block.rows) == 2
        assert block.bounding_box is None

    def test_with_bounding_box(self, sample_bbox: BoundingBox):
        block = TableBlock(rows=[], bounding_box=sample_bbox)
        assert block.bounding_box == sample_bbox

    def test_empty_rows(self):
        block = TableBlock(rows=[])
        assert block.rows == []

    def test_block_type_is_literal(self):
        block = TableBlock(rows=[])
        assert block.block_type == "table"

    def test_missing_rows_raises(self):
        with pytest.raises(ValidationError):
            TableBlock()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ImageBlock
# ---------------------------------------------------------------------------
class TestImageBlock:
    def test_construction_minimal(self):
        block = ImageBlock()
        assert block.block_type == "image"
        assert block.bounding_box is None
        assert block.description is None
        assert block.image_uri is None

    def test_with_all_fields(self, sample_bbox: BoundingBox):
        block = ImageBlock(
            bounding_box=sample_bbox,
            description="a photo",
            image_uri="/tmp/img.png",
        )
        assert block.bounding_box == sample_bbox
        assert block.description == "a photo"
        assert block.image_uri == "/tmp/img.png"


# ---------------------------------------------------------------------------
# ChartBlock
# ---------------------------------------------------------------------------
class TestChartBlock:
    def test_construction_minimal(self):
        block = ChartBlock()
        assert block.block_type == "chart"
        assert block.bounding_box is None
        assert block.description is None
        assert block.data_table is None
        assert block.image_uri is None

    def test_with_all_fields(self, sample_bbox: BoundingBox):
        block = ChartBlock(
            bounding_box=sample_bbox,
            description="bar chart",
            data_table=[[Cell(text="Q1"), Cell(text="100")]],
            image_uri="/tmp/chart.png",
        )
        assert block.bounding_box == sample_bbox
        assert block.description == "bar chart"
        assert block.data_table is not None and len(block.data_table) == 1


# ---------------------------------------------------------------------------
# ContentBlock discriminated union
# ---------------------------------------------------------------------------
class TestContentBlock:
    def test_text_block_discriminated(self, sample_lines: list[Line]):
        block: ContentBlock = TextBlock(lines=sample_lines)
        assert isinstance(block, TextBlock)

    def test_table_block_discriminated(self):
        block: ContentBlock = TableBlock(rows=[])
        assert isinstance(block, TableBlock)

    def test_image_block_discriminated(self):
        block: ContentBlock = ImageBlock()
        assert isinstance(block, ImageBlock)

    def test_chart_block_discriminated(self):
        block: ContentBlock = ChartBlock()
        assert isinstance(block, ChartBlock)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
class TestPage:
    def test_construction_with_text_block(self, sample_lines: list[Line]):
        page = Page(blocks=[TextBlock(lines=sample_lines)], width=612, height=792)
        assert len(page.blocks) == 1
        assert isinstance(page.blocks[0], TextBlock)
        assert page.width == 612
        assert page.height == 792

    def test_construction_with_table_block(self):
        rows = [[Cell(text="x")]]
        page = Page(blocks=[TableBlock(rows=rows)], width=612, height=792)
        assert len(page.blocks) == 1
        assert isinstance(page.blocks[0], TableBlock)

    def test_construction_with_mixed_blocks(self, sample_lines: list[Line]):
        blocks = [TextBlock(lines=sample_lines), TableBlock(rows=[[Cell(text="x")]])]
        page = Page(blocks=blocks, width=612, height=792)
        assert len(page.blocks) == 2

    def test_empty_blocks(self):
        page = Page(blocks=[], width=100, height=200)
        assert page.blocks == []

    def test_float_dimensions(self):
        page = Page(blocks=[], width=612.5, height=792.3)
        assert page.width == 612.5
        assert page.height == 792.3

    def test_missing_blocks_raises(self):
        with pytest.raises(ValidationError):
            Page(width=100, height=200)  # type: ignore[call-arg]

    def test_missing_dimensions_raises(self, sample_lines: list[Line]):
        with pytest.raises(ValidationError):
            Page(blocks=[TextBlock(lines=sample_lines)])  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
class TestPDF:
    def test_default_empty_pages(self):
        pdf = PDF()
        assert pdf.pages == []

    def test_construction_with_pages(self, sample_page: Page):
        pdf = PDF(pages=[sample_page])
        assert len(pdf.pages) == 1
        assert pdf.pages[0] == sample_page

    def test_multiple_pages(self, sample_page: Page):
        pdf = PDF(pages=[sample_page, sample_page, sample_page])
        assert len(pdf.pages) == 3


# ---------------------------------------------------------------------------
# PDFDocument
# ---------------------------------------------------------------------------
class TestPDFDocument:
    def test_minimal_construction(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.uri == "test.pdf"

    def test_default_content_is_none(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.content is None

    def test_with_content(self, sample_pdf: PDF):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        assert doc.content is sample_pdf
        assert len(doc.content.pages) == 2

    def test_missing_uri_raises(self):
        with pytest.raises(ValidationError):
            PDFDocument()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PDFExtractionRequest
# ---------------------------------------------------------------------------
class TestPDFExtractionRequest:
    def test_construction(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            include_citations=True,
            extraction_mode=PDFExtractionMode.SINGLE_PASS,
        )
        assert req.uri == "test.pdf"
        assert req.response_format is BaseModel
        assert req.include_citations is True
        assert req.extraction_mode is PDFExtractionMode.SINGLE_PASS

    def test_defaults(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
        )
        assert req.include_citations is True
        assert req.extraction_mode is PDFExtractionMode.SINGLE_PASS
        assert req.page_numbers is None
        assert req.llm_config is None

    def test_page_numbers_defaults_none(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
        )
        assert req.page_numbers is None

    def test_page_numbers_can_be_set(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            page_numbers=[0, 2, 4],
        )
        assert req.page_numbers == [0, 2, 4]

    def test_page_numbers_single_page(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            page_numbers=[3],
        )
        assert req.page_numbers == [3]

    def test_extraction_mode_default_single_pass(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
        )
        assert req.extraction_mode is PDFExtractionMode.SINGLE_PASS

    def test_extraction_mode_multi_pass(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        assert req.extraction_mode is PDFExtractionMode.MULTI_PASS

    def test_include_citations_default_true(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
        )
        assert req.include_citations is True

    def test_include_citations_false(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            include_citations=False,
        )
        assert req.include_citations is False

    def test_llm_config_defaults_none(self):
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
        )
        assert req.llm_config is None

    def test_llm_config_can_be_set(self):
        config = {"temperature": 0.5, "max_tokens": 1000}
        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            llm_config=config,
        )
        assert req.llm_config == config

    def test_response_format_accepts_custom_model(self):
        class Invoice(BaseModel):
            total: float
            vendor: str

        req = PDFExtractionRequest(
            uri="test.pdf",
            response_format=Invoice,
        )
        assert req.response_format is Invoice

    def test_missing_uri_raises(self):
        with pytest.raises(ValidationError):
            PDFExtractionRequest(response_format=BaseModel)  # type: ignore[call-arg]

    def test_missing_response_format_raises(self):
        with pytest.raises(ValidationError):
            PDFExtractionRequest(uri="test.pdf")  # type: ignore[call-arg]
