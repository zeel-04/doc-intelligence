"""Tests for formatter module."""

import pytest

from doc_intelligence.pdf.formatter import PDFFormatter
from doc_intelligence.pdf.schemas import PDF, PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode


@pytest.fixture
def formatter() -> PDFFormatter:
    return PDFFormatter()


# ---------------------------------------------------------------------------
# PDFFormatter
# ---------------------------------------------------------------------------
class TestPDFFormatter:
    # -- block-index formatting ------------------------------------------------

    def test_with_block_indices_format(
        self, formatter: PDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert '<block index="0" type="text">' in result
        assert "First line of text" in result
        assert '<block index="1" type="text">' in result
        assert "Second line of text" in result
        assert '<block index="2" type="text">' in result
        assert "Third line of text" in result

    def test_without_block_indices_format(
        self,
        formatter: PDFFormatter,
        sample_pdf_document_no_citations: PDFDocument,
    ):
        result = formatter.format_document_for_llm(sample_pdf_document_no_citations)
        assert "First line of text" in result
        assert "<block" not in result

    # -- page tags ----------------------------------------------------------

    def test_page_tags(self, formatter: PDFFormatter, sample_pdf_document: PDFDocument):
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert '<page number="0">' in result
        assert "</page>" in result

    def test_multiple_pages_joined(
        self, formatter: PDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert '<page number="0">' in result
        assert '<page number="1">' in result
        pages = result.split("\n\n")
        assert len(pages) == 2

    # -- citation / extraction mode routing ---------------------------------

    def test_citations_true_single_pass_uses_block_indices(
        self, formatter: PDFFormatter, sample_pdf_document: PDFDocument
    ):
        assert sample_pdf_document.include_citations is True
        assert sample_pdf_document.extraction_mode == PDFExtractionMode.SINGLE_PASS
        result = formatter.format_document_for_llm(sample_pdf_document)
        assert '<block index="0" type="text">' in result

    def test_citations_false_uses_no_block_tags(
        self,
        formatter: PDFFormatter,
        sample_pdf_document_no_citations: PDFDocument,
    ):
        assert sample_pdf_document_no_citations.include_citations is False
        result = formatter.format_document_for_llm(sample_pdf_document_no_citations)
        assert "<block" not in result
        assert "First line of text\n" in result

    # -- multi-pass routing -------------------------------------------------

    def test_multi_pass_with_citations_uses_block_indices(
        self, formatter: PDFFormatter, sample_pdf: PDF
    ):
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=True,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        result = formatter.format_document_for_llm(doc)
        assert '<block index="0" type="text">' in result

    def test_multi_pass_without_citations_uses_no_block_tags(
        self, formatter: PDFFormatter, sample_pdf: PDF
    ):
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=False,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        result = formatter.format_document_for_llm(doc)
        assert "First line of text\n" in result
        assert "<block" not in result

    def test_page_numbers_does_not_mutate_original_document(
        self, formatter: PDFFormatter, sample_pdf_document: PDFDocument
    ):
        original_page_count = len(sample_pdf_document.content.pages)  # type: ignore[union-attr]
        formatter.format_document_for_llm(sample_pdf_document, page_numbers=[0])
        assert len(sample_pdf_document.content.pages) == original_page_count  # type: ignore[union-attr]

    # -- error cases --------------------------------------------------------

    def test_none_content_raises(
        self,
        formatter: PDFFormatter,
        sample_pdf_document_unparsed: PDFDocument,
    ):
        with pytest.raises(ValueError, match="Document content is None"):
            formatter.format_document_for_llm(sample_pdf_document_unparsed)

    def test_empty_pages_raises(self, formatter: PDFFormatter, sample_pdf_empty: PDF):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf_empty)
        with pytest.raises(ValueError, match="pages are not set"):
            formatter.format_document_for_llm(doc)

    # -- page_numbers filtering ---------------------------------------------

    def test_page_numbers_filter(
        self, formatter: PDFFormatter, sample_pdf_document: PDFDocument
    ):
        result = formatter.format_document_for_llm(
            sample_pdf_document, page_numbers=[0]
        )
        assert '<page number="0">' in result
        assert '<page number="1">' not in result

    def test_page_numbers_deduplication(self, formatter: PDFFormatter, sample_pdf: PDF):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        result = formatter.format_document_for_llm(doc, page_numbers=[0, 0, 0])
        pages = result.split("\n\n")
        assert len(pages) == 1

    def test_page_numbers_sorting(self, formatter: PDFFormatter, sample_pdf: PDF):
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        result = formatter.format_document_for_llm(doc, page_numbers=[1, 0])
        first_page_pos = result.index('<page number="0">')
        second_page_pos = result.index('<page number="1">')
        assert first_page_pos < second_page_pos
