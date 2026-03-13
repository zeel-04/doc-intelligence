"""Tests for schemas.pdf module."""

import pytest
from pydantic import ValidationError

from doc_intelligence.pdf.schemas import (
    PDF,
    Line,
    Page,
    PDFDocument,
    PDFExtractionConfig,
)
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import BoundingBox


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
# Page
# ---------------------------------------------------------------------------
class TestPage:
    def test_construction(self, sample_lines: list[Line]):
        page = Page(lines=sample_lines, width=612, height=792)
        assert len(page.lines) == 3
        assert page.width == 612
        assert page.height == 792

    def test_empty_lines(self):
        page = Page(lines=[], width=100, height=200)
        assert page.lines == []

    def test_float_dimensions(self):
        page = Page(lines=[], width=612.5, height=792.3)
        assert page.width == 612.5
        assert page.height == 792.3

    def test_missing_lines_raises(self):
        with pytest.raises(ValidationError):
            Page(width=100, height=200)  # type: ignore[call-arg]

    def test_missing_dimensions_raises(self, sample_lines: list[Line]):
        with pytest.raises(ValidationError):
            Page(lines=sample_lines)  # type: ignore[call-arg]


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
