"""Tests for schemas.pdf module."""

import pytest
from pydantic import ValidationError

from doc_intelligence.pdf.schemas import (
    PDF,
    Cell,
    ContentBlock,
    Page,
    PDFDocument,
    PDFExtractionConfig,
    TableBlock,
    TextBlock,
    blocks_to_lines,
    table_block_to_text_block,
)
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import BoundingBox, Line


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
# ContentBlock discriminated union
# ---------------------------------------------------------------------------
class TestContentBlock:
    def test_text_block_discriminated(self, sample_lines: list[Line]):
        # Pydantic should resolve via block_type discriminator
        block: ContentBlock = TextBlock(lines=sample_lines)
        assert isinstance(block, TextBlock)

    def test_table_block_discriminated(self):
        block: ContentBlock = TableBlock(rows=[])
        assert isinstance(block, TableBlock)


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

    def test_default_extraction_mode(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.extraction_mode is PDFExtractionMode.SINGLE_PASS

    def test_custom_extraction_mode(self):
        doc = PDFDocument(uri="test.pdf", extraction_mode=PDFExtractionMode.MULTI_PASS)
        assert doc.extraction_mode is PDFExtractionMode.MULTI_PASS

    def test_with_content(self, sample_pdf: PDF):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        assert doc.content is sample_pdf
        assert len(doc.content.pages) == 2

    def test_inherits_document_defaults(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.include_citations is True

    def test_missing_uri_raises(self):
        with pytest.raises(ValidationError):
            PDFDocument()  # type: ignore[call-arg]

    def test_pass1_result_defaults_none(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.pass1_result is None

    def test_pass2_page_map_defaults_none(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.pass2_page_map is None

    def test_pass1_result_can_be_set(self):
        from pydantic import BaseModel as BM

        class Dummy(BM):
            x: int

        doc = PDFDocument(uri="test.pdf", pass1_result=Dummy(x=1))
        assert doc.pass1_result is not None
        assert doc.pass1_result.x == 1  # type: ignore[attr-defined]

    def test_pass2_page_map_can_be_set(self):
        doc = PDFDocument(uri="test.pdf", pass2_page_map={"name": [0, 1], "age": [0]})
        assert doc.pass2_page_map == {"name": [0, 1], "age": [0]}

    def test_page_numbers_defaults_none(self):
        doc = PDFDocument(uri="test.pdf")
        assert doc.page_numbers is None

    def test_page_numbers_can_be_set(self):
        doc = PDFDocument(uri="test.pdf", page_numbers=[0, 2, 4])
        assert doc.page_numbers == [0, 2, 4]

    def test_page_numbers_single_page(self):
        doc = PDFDocument(uri="test.pdf", page_numbers=[3])
        assert doc.page_numbers == [3]


# ---------------------------------------------------------------------------
# PDFExtractionConfig
# ---------------------------------------------------------------------------
class TestPDFExtractionConfig:
    def test_construction(self):
        cfg = PDFExtractionConfig(
            include_citations=True,
            extraction_mode=PDFExtractionMode.SINGLE_PASS,
        )
        assert cfg.include_citations is True
        assert cfg.extraction_mode is PDFExtractionMode.SINGLE_PASS

    def test_page_numbers_default_none(self):
        cfg = PDFExtractionConfig(
            include_citations=True,
            extraction_mode=PDFExtractionMode.SINGLE_PASS,
        )
        assert cfg.page_numbers is None

    def test_page_numbers_set(self):
        cfg = PDFExtractionConfig(
            include_citations=True,
            extraction_mode=PDFExtractionMode.SINGLE_PASS,
            page_numbers=[0, 1, 2],
        )
        assert cfg.page_numbers == [0, 1, 2]

    def test_missing_extraction_mode_raises(self):
        with pytest.raises(ValidationError):
            PDFExtractionConfig(include_citations=True)  # type: ignore[call-arg]

    def test_missing_include_citations_raises(self):
        with pytest.raises(ValidationError):
            PDFExtractionConfig(extraction_mode=PDFExtractionMode.SINGLE_PASS)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# table_block_to_text_block
# ---------------------------------------------------------------------------
class TestTableBlockToTextBlock:
    def test_single_row(self, sample_bbox: BoundingBox):
        table = TableBlock(
            rows=[[Cell(text="Name"), Cell(text="Age")]],
            bounding_box=sample_bbox,
        )
        result = table_block_to_text_block(table)
        assert isinstance(result, TextBlock)
        assert len(result.lines) == 1
        assert result.lines[0].text == "| Name | Age |"

    def test_multiple_rows(self):
        table = TableBlock(
            rows=[
                [Cell(text="a"), Cell(text="b")],
                [Cell(text="c"), Cell(text="d")],
            ]
        )
        result = table_block_to_text_block(table)
        assert len(result.lines) == 2
        assert result.lines[0].text == "| a | b |"
        assert result.lines[1].text == "| c | d |"

    def test_empty_table_produces_no_lines(self):
        result = table_block_to_text_block(TableBlock(rows=[]))
        assert result.lines == []

    def test_cell_bbox_used_when_available(self, sample_bbox: BoundingBox):
        cell_bbox = BoundingBox(x0=0.1, top=0.1, x1=0.9, bottom=0.2)
        table = TableBlock(rows=[[Cell(text="x", bounding_box=cell_bbox)]])
        result = table_block_to_text_block(table)
        assert result.lines[0].bounding_box == cell_bbox

    def test_table_bbox_used_as_fallback(self, sample_bbox: BoundingBox):
        table = TableBlock(rows=[[Cell(text="x")]], bounding_box=sample_bbox)
        result = table_block_to_text_block(table)
        assert result.lines[0].bounding_box == sample_bbox

    def test_zero_bbox_used_when_no_bbox_available(self):
        table = TableBlock(rows=[[Cell(text="x")]])
        result = table_block_to_text_block(table)
        bbox = result.lines[0].bounding_box
        assert bbox == BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=1.0)

    def test_block_type_is_text(self):
        result = table_block_to_text_block(TableBlock(rows=[]))
        assert result.block_type == "text"

    def test_bounding_box_preserved(self, sample_bbox: BoundingBox):
        table = TableBlock(rows=[], bounding_box=sample_bbox)
        result = table_block_to_text_block(table)
        assert result.bounding_box == sample_bbox


# ---------------------------------------------------------------------------
# blocks_to_lines
# ---------------------------------------------------------------------------
class TestBlocksToLines:
    def test_single_text_block(self, sample_lines: list[Line]):
        blocks: list[ContentBlock] = [TextBlock(lines=sample_lines)]
        result = blocks_to_lines(blocks)
        assert result == sample_lines

    def test_multiple_text_blocks(self, sample_lines: list[Line]):
        blocks: list[ContentBlock] = [
            TextBlock(lines=sample_lines[:1]),
            TextBlock(lines=sample_lines[1:]),
        ]
        result = blocks_to_lines(blocks)
        assert result == sample_lines

    def test_table_block_converted(self, sample_bbox: BoundingBox):
        table = TableBlock(
            rows=[[Cell(text="x"), Cell(text="y")]],
            bounding_box=sample_bbox,
        )
        result = blocks_to_lines([table])
        assert len(result) == 1
        assert result[0].text == "| x | y |"

    def test_mixed_blocks_preserve_order(
        self, sample_lines: list[Line], sample_bbox: BoundingBox
    ):
        text_block = TextBlock(lines=sample_lines[:1])
        table_block = TableBlock(
            rows=[[Cell(text="cell")]],
            bounding_box=sample_bbox,
        )
        result = blocks_to_lines([text_block, table_block])
        assert len(result) == 2
        assert result[0] == sample_lines[0]
        assert result[1].text == "| cell |"

    def test_empty_blocks_list(self):
        assert blocks_to_lines([]) == []

    def test_empty_text_block(self):
        result = blocks_to_lines([TextBlock(lines=[])])
        assert result == []
