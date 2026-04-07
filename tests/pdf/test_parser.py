"""Tests for parser module."""

from io import BytesIO
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from doc_intelligence.pdf.parser import PDFParser
from doc_intelligence.pdf.schemas import PDFDocument
from doc_intelligence.pdf.types import ParseStrategy, ScannedPipelineType
from tests.conftest import FakeLLM


def _make_mock_page(
    width: int | float = 500,
    height: int | float = 800,
    lines: list[dict] | None = None,
):
    """Build a mock pdfplumber page with configurable text lines."""
    if lines is None:
        lines = [
            {"text": "Hello world", "x0": 50, "top": 100, "x1": 250, "bottom": 120},
            {"text": "Second line", "x0": 50, "top": 130, "x1": 250, "bottom": 150},
        ]
    page = MagicMock()
    page.width = width
    page.height = height
    page.extract_text_lines.return_value = lines
    return page


def _make_mock_pdf(pages=None):
    """Build a mock pdfplumber PDF context manager."""
    if pages is None:
        pages = [_make_mock_page()]
    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    return pdf


# ---------------------------------------------------------------------------
# PDFParser — strategy selection
# ---------------------------------------------------------------------------
class TestPDFParserStrategy:
    def test_default_strategy_is_digital(self):
        parser = PDFParser()
        assert parser._strategy == ParseStrategy.DIGITAL

    def test_default_scanned_pipeline_is_vlm(self):
        fake_llm = FakeLLM()
        parser = PDFParser(
            strategy=ParseStrategy.SCANNED,
            llm=fake_llm,
        )
        assert parser._scanned_pipeline == ScannedPipelineType.VLM

    def test_scanned_two_stage_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="TWO_STAGE.*not yet implemented"):
            PDFParser(
                strategy=ParseStrategy.SCANNED,
                scanned_pipeline=ScannedPipelineType.TWO_STAGE,
            )

    def test_scanned_vlm_without_llm_raises(self):
        with pytest.raises(ValueError, match="llm is required"):
            PDFParser(
                strategy=ParseStrategy.SCANNED,
                scanned_pipeline=ScannedPipelineType.VLM,
            )

    def test_scanned_vlm_stores_llm(self):
        fake_llm = FakeLLM()
        parser = PDFParser(
            strategy=ParseStrategy.SCANNED,
            scanned_pipeline=ScannedPipelineType.VLM,
            llm=fake_llm,
        )
        assert parser._scanned_pipeline == ScannedPipelineType.VLM
        assert parser._llm is fake_llm

    def test_scanned_vlm_stores_batch_size(self):
        fake_llm = FakeLLM()
        parser = PDFParser(
            strategy=ParseStrategy.SCANNED,
            scanned_pipeline=ScannedPipelineType.VLM,
            llm=fake_llm,
            vlm_batch_size=5,
        )
        assert parser._vlm_batch_size == 5

    def test_scanned_vlm_default_batch_size(self):
        fake_llm = FakeLLM()
        parser = PDFParser(
            strategy=ParseStrategy.SCANNED,
            scanned_pipeline=ScannedPipelineType.VLM,
            llm=fake_llm,
        )
        assert parser._vlm_batch_size == 1

    def test_unsupported_strategy(self):
        parser = PDFParser()
        parser._strategy = "invalid_strategy"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported parse strategy"):
            parser.parse("test.pdf")


# ---------------------------------------------------------------------------
# PDFParser — digital strategy
# ---------------------------------------------------------------------------
class TestPDFParserDigital:
    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_parse_local_file(self, mock_pdfplumber):
        mock_pdf = _make_mock_pdf()
        mock_pdfplumber.open.return_value = mock_pdf

        parser = PDFParser(strategy=ParseStrategy.DIGITAL)
        result = parser.parse("/path/to/test.pdf")

        mock_pdfplumber.open.assert_called_once_with("/path/to/test.pdf")
        assert isinstance(result, PDFDocument)
        assert result.content is not None
        assert len(result.content.pages) == 1
        assert len(result.content.pages[0].blocks) == 2
        assert result.content.pages[0].blocks[0].lines[0].text == "Hello world"  # type: ignore[union-attr]
        assert result.content.pages[0].blocks[1].lines[0].text == "Second line"  # type: ignore[union-attr]

    @patch("doc_intelligence.pdf.parser.requests")
    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_parse_http_url(self, mock_pdfplumber, mock_requests):
        mock_response = MagicMock()
        mock_response.content = b"fake-pdf-bytes"
        mock_requests.get.return_value = mock_response

        mock_pdf = _make_mock_pdf()
        mock_pdfplumber.open.return_value = mock_pdf

        parser = PDFParser(strategy=ParseStrategy.DIGITAL)
        result = parser.parse("http://example.com/test.pdf")

        mock_requests.get.assert_called_once_with("http://example.com/test.pdf")
        mock_response.raise_for_status.assert_called_once()
        call_arg = mock_pdfplumber.open.call_args[0][0]
        assert isinstance(call_arg, BytesIO)
        assert isinstance(result, PDFDocument)

    @patch("doc_intelligence.pdf.parser.requests")
    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_parse_https_url(self, mock_pdfplumber, mock_requests):
        mock_response = MagicMock()
        mock_response.content = b"fake-pdf-bytes"
        mock_requests.get.return_value = mock_response
        mock_pdfplumber.open.return_value = _make_mock_pdf()

        parser = PDFParser(strategy=ParseStrategy.DIGITAL)
        parser.parse("https://example.com/test.pdf")

        mock_requests.get.assert_called_once_with("https://example.com/test.pdf")

    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_bboxes_are_normalized(self, mock_pdfplumber):
        page = _make_mock_page(
            width=500,
            height=1000,
            lines=[
                {"text": "test", "x0": 100, "top": 200, "x1": 300, "bottom": 400},
            ],
        )
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = PDFParser()
        result = parser.parse("test.pdf")

        assert result.content is not None
        bbox = result.content.pages[0].blocks[0].lines[0].bounding_box  # type: ignore[union-attr]
        assert bbox.x0 == pytest.approx(0.2)
        assert bbox.top == pytest.approx(0.2)
        assert bbox.x1 == pytest.approx(0.6)
        assert bbox.bottom == pytest.approx(0.4)

    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_multiple_pages(self, mock_pdfplumber):
        pages = [_make_mock_page(), _make_mock_page(), _make_mock_page()]
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=pages)

        parser = PDFParser()
        result = parser.parse("test.pdf")

        assert result.content is not None
        assert len(result.content.pages) == 3

    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_empty_page(self, mock_pdfplumber):
        page = _make_mock_page(lines=[])
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = PDFParser()
        result = parser.parse("test.pdf")

        assert result.content is not None
        assert result.content.pages[0].blocks == []

    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_page_dimensions_preserved(self, mock_pdfplumber):
        page = _make_mock_page(width=612.5, height=792.0)
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = PDFParser()
        result = parser.parse("test.pdf")

        assert result.content is not None
        assert result.content.pages[0].width == 612.5
        assert result.content.pages[0].height == 792.0

    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_line_text_preserved(self, mock_pdfplumber):
        page = _make_mock_page(
            lines=[
                {
                    "text": "Special chars: é à ü",
                    "x0": 0,
                    "top": 0,
                    "x1": 1,
                    "bottom": 1,
                },
            ]
        )
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = PDFParser()
        result = parser.parse("test.pdf")

        assert result.content is not None
        assert result.content.pages[0].blocks[0].lines[0].text == "Special chars: é à ü"  # type: ignore[union-attr]

    @patch("doc_intelligence.pdf.parser.pdfplumber")
    def test_result_uri_matches_input(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = _make_mock_pdf()

        parser = PDFParser()
        result = parser.parse("/my/document.pdf")

        assert result.uri == "/my/document.pdf"

    @patch("doc_intelligence.pdf.parser.requests")
    def test_http_error_propagates(self, mock_requests):
        from requests.exceptions import HTTPError

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_requests.get.return_value = mock_response

        parser = PDFParser()
        with pytest.raises(HTTPError):
            parser.parse("http://example.com/missing.pdf")
