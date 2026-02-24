"""Tests for parser module."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from doc_intelligence.parser import DigitalPDFParser, PDFParser
from doc_intelligence.schemas.pdf import PDFDocument


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
# PDFParser ABC
# ---------------------------------------------------------------------------
class TestPDFParserABC:
    def test_not_instantiable(self):
        with pytest.raises(TypeError):
            PDFParser()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DigitalPDFParser
# ---------------------------------------------------------------------------
class TestDigitalPDFParser:
    @patch("doc_intelligence.parser.pdfplumber")
    def test_parse_local_file(self, mock_pdfplumber):
        mock_pdf = _make_mock_pdf()
        mock_pdfplumber.open.return_value = mock_pdf

        parser = DigitalPDFParser()
        doc = PDFDocument(uri="/path/to/test.pdf")
        result = parser.parse(doc)

        mock_pdfplumber.open.assert_called_once_with("/path/to/test.pdf")
        assert isinstance(result, PDFDocument)
        assert result.content is not None
        assert len(result.content.pages) == 1
        assert len(result.content.pages[0].lines) == 2

    @patch("doc_intelligence.parser.requests")
    @patch("doc_intelligence.parser.pdfplumber")
    def test_parse_http_url(self, mock_pdfplumber, mock_requests):
        mock_response = MagicMock()
        mock_response.content = b"fake-pdf-bytes"
        mock_requests.get.return_value = mock_response

        mock_pdf = _make_mock_pdf()
        mock_pdfplumber.open.return_value = mock_pdf

        parser = DigitalPDFParser()
        doc = PDFDocument(uri="http://example.com/test.pdf")
        result = parser.parse(doc)

        mock_requests.get.assert_called_once_with("http://example.com/test.pdf")
        mock_response.raise_for_status.assert_called_once()
        call_arg = mock_pdfplumber.open.call_args[0][0]
        assert isinstance(call_arg, BytesIO)
        assert isinstance(result, PDFDocument)

    @patch("doc_intelligence.parser.requests")
    @patch("doc_intelligence.parser.pdfplumber")
    def test_parse_https_url(self, mock_pdfplumber, mock_requests):
        mock_response = MagicMock()
        mock_response.content = b"fake-pdf-bytes"
        mock_requests.get.return_value = mock_response
        mock_pdfplumber.open.return_value = _make_mock_pdf()

        parser = DigitalPDFParser()
        doc = PDFDocument(uri="https://example.com/test.pdf")
        parser.parse(doc)

        mock_requests.get.assert_called_once_with("https://example.com/test.pdf")

    @patch("doc_intelligence.parser.pdfplumber")
    def test_bboxes_are_normalized(self, mock_pdfplumber):
        page = _make_mock_page(
            width=500,
            height=1000,
            lines=[
                {"text": "test", "x0": 100, "top": 200, "x1": 300, "bottom": 400},
            ],
        )
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = DigitalPDFParser()
        result = parser.parse(PDFDocument(uri="test.pdf"))

        bbox = result.content.pages[0].lines[0].bounding_box
        assert bbox.x0 == pytest.approx(0.2)
        assert bbox.top == pytest.approx(0.2)
        assert bbox.x1 == pytest.approx(0.6)
        assert bbox.bottom == pytest.approx(0.4)

    @patch("doc_intelligence.parser.pdfplumber")
    def test_multiple_pages(self, mock_pdfplumber):
        pages = [_make_mock_page(), _make_mock_page(), _make_mock_page()]
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=pages)

        parser = DigitalPDFParser()
        result = parser.parse(PDFDocument(uri="test.pdf"))

        assert len(result.content.pages) == 3

    @patch("doc_intelligence.parser.pdfplumber")
    def test_empty_page(self, mock_pdfplumber):
        page = _make_mock_page(lines=[])
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = DigitalPDFParser()
        result = parser.parse(PDFDocument(uri="test.pdf"))

        assert result.content.pages[0].lines == []

    @patch("doc_intelligence.parser.pdfplumber")
    def test_page_dimensions_preserved(self, mock_pdfplumber):
        page = _make_mock_page(width=612.5, height=792.0)
        mock_pdfplumber.open.return_value = _make_mock_pdf(pages=[page])

        parser = DigitalPDFParser()
        result = parser.parse(PDFDocument(uri="test.pdf"))

        assert result.content.pages[0].width == 612.5
        assert result.content.pages[0].height == 792.0

    @patch("doc_intelligence.parser.pdfplumber")
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

        parser = DigitalPDFParser()
        result = parser.parse(PDFDocument(uri="test.pdf"))

        assert result.content.pages[0].lines[0].text == "Special chars: é à ü"

    @patch("doc_intelligence.parser.pdfplumber")
    def test_result_uri_matches_input(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = _make_mock_pdf()

        parser = DigitalPDFParser()
        result = parser.parse(PDFDocument(uri="/my/document.pdf"))

        assert result.uri == "/my/document.pdf"

    @patch("doc_intelligence.parser.requests")
    def test_http_error_propagates(self, mock_requests):
        from requests.exceptions import HTTPError

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_requests.get.return_value = mock_response

        parser = DigitalPDFParser()
        with pytest.raises(HTTPError):
            parser.parse(PDFDocument(uri="http://example.com/missing.pdf"))
