"""Tests for PDFParser scanned helpers in pdf.parser module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from doc_intelligence.pdf.parser import (
    _render_pdf_to_images,
)


# ---------------------------------------------------------------------------
# _render_pdf_to_images
# ---------------------------------------------------------------------------
class TestRenderPdfToImages:
    def _mock_pdfium_page(self, h: int = 100, w: int = 80) -> MagicMock:
        """Return a mock pypdfium2 page that renders to a (h x w x 3) array."""
        from PIL import Image

        page = MagicMock()
        bitmap = MagicMock()
        bitmap.to_pil.return_value = Image.fromarray(
            np.zeros((h, w, 3), dtype=np.uint8)
        )
        page.render.return_value = bitmap
        return page

    def test_local_path_passed_directly_to_pdfium(self) -> None:
        mock_page = self._mock_pdfium_page()
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch("doc_intelligence.pdf.parser.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_pdf
            _render_pdf_to_images("/local/path/file.pdf", dpi=150)

        mock_pdfium.PdfDocument.assert_called_once_with("/local/path/file.pdf")

    def test_url_content_passed_to_pdfium(self) -> None:
        """HTTP URLs are downloaded; the response bytes are passed to pdfium."""
        mock_page = self._mock_pdfium_page()
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))

        pdf_bytes = b"%PDF-1.4 fake"
        mock_response = MagicMock()
        mock_response.content = pdf_bytes

        with (
            patch("doc_intelligence.pdf.parser.requests") as mock_requests,
            patch("doc_intelligence.pdf.parser.pdfium") as mock_pdfium,
        ):
            mock_requests.get.return_value = mock_response
            mock_pdfium.PdfDocument.return_value = mock_pdf
            _render_pdf_to_images("https://example.com/scan.pdf", dpi=150)

        mock_requests.get.assert_called_once_with("https://example.com/scan.pdf")
        mock_response.raise_for_status.assert_called_once()
        mock_pdfium.PdfDocument.assert_called_once_with(pdf_bytes)

    @pytest.mark.parametrize(
        "scheme", ["http://example.com/a.pdf", "https://example.com/a.pdf"]
    )
    def test_both_http_schemes_trigger_download(self, scheme: str) -> None:
        mock_page = self._mock_pdfium_page()
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))

        with (
            patch("doc_intelligence.pdf.parser.requests") as mock_requests,
            patch("doc_intelligence.pdf.parser.pdfium") as mock_pdfium,
        ):
            mock_requests.get.return_value = MagicMock(content=b"bytes")
            mock_pdfium.PdfDocument.return_value = mock_pdf
            _render_pdf_to_images(scheme, dpi=72)

        mock_requests.get.assert_called_once()

    def test_scale_uses_dpi_over_72(self) -> None:
        mock_page = self._mock_pdfium_page()
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch("doc_intelligence.pdf.parser.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_pdf
            _render_pdf_to_images("file.pdf", dpi=144)

        mock_page.render.assert_called_once_with(scale=144 / 72.0)

    def test_returns_one_array_per_page(self) -> None:
        pages = [self._mock_pdfium_page(100, 80), self._mock_pdfium_page(100, 80)]
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter(pages))

        with patch("doc_intelligence.pdf.parser.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_pdf
            result = _render_pdf_to_images("file.pdf", dpi=72)

        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_empty_pdf_returns_empty_list(self) -> None:
        mock_pdf = MagicMock()
        mock_pdf.__iter__ = MagicMock(return_value=iter([]))

        with patch("doc_intelligence.pdf.parser.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_pdf
            result = _render_pdf_to_images("empty.pdf", dpi=72)

        assert result == []

    def test_http_error_propagates(self) -> None:
        with patch("doc_intelligence.pdf.parser.requests") as mock_requests:
            mock_requests.get.return_value = MagicMock(
                **{"raise_for_status.side_effect": requests.HTTPError("404")}
            )
            with pytest.raises(requests.HTTPError):
                _render_pdf_to_images("https://example.com/missing.pdf", dpi=72)
