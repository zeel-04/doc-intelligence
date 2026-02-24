"""Tests for formatter module."""

import pytest

from doc_intelligence.formatter import DigitalPDFFormatter
from doc_intelligence.schemas.pdf import PDF, PDFDocument
from doc_intelligence.types.pdf import PDFExtractionMode


@pytest.fixture
def formatter() -> DigitalPDFFormatter:
    return DigitalPDFFormatter()


# ---------------------------------------------------------------------------
# DigitalPDFFormatter
# ---------------------------------------------------------------------------
class TestDigitalPDFFormatter:
    # -- line-number formatting ---------------------------------------------

    def test_with_line_numbers_format(
        self, formatter: DigitalPDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert "0: First line of text" in result
        assert "1: Second line of text" in result
        assert "2: Third line of text" in result

    def test_without_line_numbers_format(
        self,
        formatter: DigitalPDFFormatter,
        sample_pdf_document_no_citations: PDFDocument,
    ):
        result = formatter.format_document_for_llm(sample_pdf_document_no_citations)
        assert "First line of text" in result
        assert "0:" not in result
        assert "1:" not in result

    # -- page tags ----------------------------------------------------------

    def test_page_tags(
        self, formatter: DigitalPDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert "<page number=0>" in result
        assert "</page>" in result

    def test_multiple_pages_joined(
        self, formatter: DigitalPDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert "<page number=0>" in result
        assert "<page number=1>" in result
        pages = result.split("\n\n")
        assert len(pages) == 2

    # -- citation / extraction mode routing ---------------------------------

    def test_citations_true_single_pass_uses_line_numbers(
        self, formatter: DigitalPDFFormatter, sample_pdf_document: PDFDocument
    ):
        assert sample_pdf_document.include_citations is True
        assert sample_pdf_document.extraction_mode == PDFExtractionMode.SINGLE_PASS
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert "0: First line of text" in result

    def test_citations_false_uses_no_line_numbers(
        self,
        formatter: DigitalPDFFormatter,
        sample_pdf_document_no_citations: PDFDocument,
    ):
        assert sample_pdf_document_no_citations.include_citations is False
        result = formatter.format_document_for_llm(sample_pdf_document_no_citations)
        assert "0:" not in result
        assert "First line of text\n" in result

    # -- error cases --------------------------------------------------------

    def test_multi_pass_raises(self, formatter: DigitalPDFFormatter, sample_pdf: PDF):
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        with pytest.raises(NotImplementedError, match="Multi-pass"):
            formatter.format_document_for_llm(doc)

    def test_none_content_raises(
        self,
        formatter: DigitalPDFFormatter,
        sample_pdf_document_unparsed: PDFDocument,
    ):
        with pytest.raises(ValueError, match="Document content is None"):
            formatter.format_document_for_llm(sample_pdf_document_unparsed)

    def test_empty_pages_raises(
        self, formatter: DigitalPDFFormatter, sample_pdf_empty: PDF
    ):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf_empty)
        with pytest.raises(ValueError, match="pages are not set"):
            formatter.format_document_for_llm(doc)

    # -- page_numbers filtering ---------------------------------------------

    def test_page_numbers_filter(
        self, formatter: DigitalPDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(
            sample_pdf_document, page_numbers=[0]
        )
        assert "<page number=0>" in result
        assert "<page number=1>" not in result

    def test_page_numbers_deduplication(
        self, formatter: DigitalPDFFormatter, sample_pdf: PDF
    ):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        result = formatter.format_document_for_llm(doc, page_numbers=[0, 0, 0])
        pages = result.split("\n\n")
        assert len(pages) == 1

    def test_page_numbers_sorting(
        self, formatter: DigitalPDFFormatter, sample_pdf: PDF
    ):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        result = formatter.format_document_for_llm(doc, page_numbers=[1, 0])
        first_page_pos = result.index("<page number=0>")
        second_page_pos = result.index("<page number=1>")
        assert first_page_pos < second_page_pos
