"""Tests for ScannedPDFParser in pdf.parser module."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.pdf.parser import (
    ScannedPDFParser,
    _crop,
    _render_pdf_to_images,
)
from doc_intelligence.pdf.schemas import (
    PDF,
    Page,
    PDFDocument,
    TableBlock,
    TextBlock,
)
from doc_intelligence.schemas.core import BoundingBox, Line


# ---------------------------------------------------------------------------
# Helpers / shared test data
# ---------------------------------------------------------------------------
def _make_image(h: int = 200, w: int = 150) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_bbox(
    x0: float = 10.0, top: float = 20.0, x1: float = 100.0, bottom: float = 80.0
) -> BoundingBox:
    return BoundingBox(x0=x0, top=top, x1=x1, bottom=bottom)


def _make_region(region_type: str = "text", confidence: float = 0.9) -> LayoutRegion:
    return LayoutRegion(
        bounding_box=_make_bbox(), region_type=region_type, confidence=confidence
    )


def _make_line(text: str = "hello") -> Line:
    return Line(
        text=text, bounding_box=BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=0.1)
    )


def _make_parser(
    regions: list[LayoutRegion] | None = None,
    lines: list[Line] | None = None,
    dpi: int = 150,
) -> tuple[ScannedPDFParser, MagicMock, MagicMock]:
    """Return (parser, layout_detector_mock, ocr_engine_mock)."""
    from tests.conftest import FakeLayoutDetector, FakeOCREngine

    layout_detector = FakeLayoutDetector(regions=regions or [])
    ocr_engine = FakeOCREngine(lines=lines or [])
    parser = ScannedPDFParser(
        layout_detector=layout_detector,
        ocr_engine=ocr_engine,
        dpi=dpi,
    )
    return parser, layout_detector, ocr_engine


def _patched_parse(
    parser: ScannedPDFParser,
    uri: str = "test.pdf",
    images: list[np.ndarray] | None = None,
) -> PDFDocument:
    """Run parser.parse() with _render_pdf_to_images patched to return *images*."""
    if images is None:
        images = [_make_image()]
    with patch(
        "doc_intelligence.pdf.parser._render_pdf_to_images",
        return_value=images,
    ):
        doc = PDFDocument(uri=uri)
        return parser.parse(doc)


# ---------------------------------------------------------------------------
# ScannedPDFParser — construction
# ---------------------------------------------------------------------------
class TestScannedPDFParserConstruction:
    def test_stores_layout_detector(self) -> None:
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        detector = FakeLayoutDetector()
        engine = FakeOCREngine()
        parser = ScannedPDFParser(layout_detector=detector, ocr_engine=engine)

        assert parser._layout_detector is detector

    def test_stores_ocr_engine(self) -> None:
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        detector = FakeLayoutDetector()
        engine = FakeOCREngine()
        parser = ScannedPDFParser(layout_detector=detector, ocr_engine=engine)

        assert parser._ocr_engine is engine

    def test_default_dpi_is_150(self) -> None:
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        parser = ScannedPDFParser(
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert parser._dpi == 150

    def test_custom_dpi(self) -> None:
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        parser = ScannedPDFParser(
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
            dpi=300,
        )
        assert parser._dpi == 300


# ---------------------------------------------------------------------------
# ScannedPDFParser — parse() output shape
# ---------------------------------------------------------------------------
class TestScannedPDFParserParse:
    def test_returns_pdf_document(self) -> None:
        parser, _, _ = _make_parser()
        result = _patched_parse(parser)
        assert isinstance(result, PDFDocument)

    def test_preserves_uri(self) -> None:
        parser, _, _ = _make_parser()
        result = _patched_parse(parser, uri="my_scan.pdf")
        assert result.uri == "my_scan.pdf"

    def test_content_is_pdf(self) -> None:
        parser, _, _ = _make_parser()
        result = _patched_parse(parser)
        assert isinstance(result.content, PDF)

    def test_empty_pdf_has_no_pages(self) -> None:
        parser, _, _ = _make_parser()
        result = _patched_parse(parser, images=[])
        assert result.content is not None
        assert result.content.pages == []

    def test_one_image_produces_one_page(self) -> None:
        parser, _, _ = _make_parser()
        result = _patched_parse(parser, images=[_make_image()])
        assert result.content is not None
        assert len(result.content.pages) == 1

    def test_three_images_produce_three_pages(self) -> None:
        parser, _, _ = _make_parser()
        result = _patched_parse(parser, images=[_make_image()] * 3)
        assert result.content is not None
        assert len(result.content.pages) == 3

    def test_page_dimensions_match_image(self) -> None:
        parser, _, _ = _make_parser()
        image = _make_image(h=300, w=200)
        result = _patched_parse(parser, images=[image])
        assert result.content is not None
        page = result.content.pages[0]
        assert page.width == 200
        assert page.height == 300

    def test_empty_page_has_no_blocks(self) -> None:
        """No layout regions → no blocks."""
        parser, _, _ = _make_parser(regions=[])
        result = _patched_parse(parser, images=[_make_image()])
        assert result.content is not None
        assert result.content.pages[0].blocks == []


# ---------------------------------------------------------------------------
# ScannedPDFParser — block assembly
# ---------------------------------------------------------------------------
class TestScannedPDFParserBlocks:
    def test_text_region_becomes_text_block(self) -> None:
        regions = [_make_region(region_type="text")]
        lines = [_make_line("extracted")]
        parser, _, _ = _make_parser(regions=regions, lines=lines)

        result = _patched_parse(parser)
        assert result.content is not None
        blocks = result.content.pages[0].blocks

        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)

    def test_title_region_becomes_text_block(self) -> None:
        regions = [_make_region(region_type="title")]
        parser, _, _ = _make_parser(regions=regions, lines=[_make_line()])

        result = _patched_parse(parser)
        assert result.content is not None
        assert isinstance(result.content.pages[0].blocks[0], TextBlock)

    def test_text_block_contains_ocr_lines(self) -> None:
        regions = [_make_region(region_type="text")]
        lines = [_make_line("first"), _make_line("second")]
        parser, _, _ = _make_parser(regions=regions, lines=lines)

        result = _patched_parse(parser)
        assert result.content is not None
        block = result.content.pages[0].blocks[0]
        assert isinstance(block, TextBlock)
        assert [l.text for l in block.lines] == ["first", "second"]

    def test_text_block_bbox_from_region(self) -> None:
        bbox = _make_bbox(x0=5.0, top=10.0, x1=120.0, bottom=90.0)
        regions = [LayoutRegion(bounding_box=bbox, region_type="text", confidence=0.9)]
        parser, _, _ = _make_parser(regions=regions, lines=[])

        result = _patched_parse(parser)
        assert result.content is not None
        block = result.content.pages[0].blocks[0]
        assert isinstance(block, TextBlock)
        assert block.bounding_box == bbox

    def test_table_region_becomes_table_block(self) -> None:
        regions = [_make_region(region_type="table")]
        lines = [_make_line("cell text")]
        parser, _, _ = _make_parser(regions=regions, lines=lines)

        result = _patched_parse(parser)
        assert result.content is not None
        block = result.content.pages[0].blocks[0]
        assert isinstance(block, TableBlock)

    def test_table_block_has_one_row_per_line(self) -> None:
        regions = [_make_region(region_type="table")]
        lines = [_make_line("row1"), _make_line("row2"), _make_line("row3")]
        parser, _, _ = _make_parser(regions=regions, lines=lines)

        result = _patched_parse(parser)
        assert result.content is not None
        block = result.content.pages[0].blocks[0]
        assert isinstance(block, TableBlock)
        assert len(block.rows) == 3
        assert [row[0].text for row in block.rows] == ["row1", "row2", "row3"]

    def test_table_block_bbox_from_region(self) -> None:
        bbox = _make_bbox(x0=0.0, top=50.0, x1=140.0, bottom=100.0)
        regions = [LayoutRegion(bounding_box=bbox, region_type="table", confidence=0.8)]
        parser, _, _ = _make_parser(regions=regions, lines=[_make_line()])

        result = _patched_parse(parser)
        assert result.content is not None
        block = result.content.pages[0].blocks[0]
        assert isinstance(block, TableBlock)
        assert block.bounding_box == bbox

    def test_multiple_regions_produce_multiple_blocks(self) -> None:
        regions = [
            _make_region(region_type="text"),
            _make_region(region_type="table"),
            _make_region(region_type="text"),
        ]
        parser, _, _ = _make_parser(regions=regions, lines=[_make_line()])

        result = _patched_parse(parser)
        assert result.content is not None
        blocks = result.content.pages[0].blocks
        assert len(blocks) == 3
        assert isinstance(blocks[0], TextBlock)
        assert isinstance(blocks[1], TableBlock)
        assert isinstance(blocks[2], TextBlock)

    def test_ocr_receives_cropped_image(self) -> None:
        """OCR engine must receive a crop, not the full page image."""
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        # Region covers a specific sub-rectangle of the page.
        bbox = BoundingBox(x0=10.0, top=20.0, x1=60.0, bottom=70.0)
        regions = [LayoutRegion(bounding_box=bbox, region_type="text", confidence=0.9)]

        ocr_engine = FakeOCREngine(lines=[])
        parser = ScannedPDFParser(
            layout_detector=FakeLayoutDetector(regions=regions),
            ocr_engine=ocr_engine,
        )
        full_image = _make_image(h=200, w=150)
        _patched_parse(parser, images=[full_image])

        assert ocr_engine.last_image is not None
        # Cropped region: rows 20:70, cols 10:60 → shape (50, 50, 3)
        assert ocr_engine.last_image.shape == (50, 50, 3)

    def test_layout_detector_receives_full_page_image(self) -> None:
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        detector = FakeLayoutDetector(regions=[])
        parser = ScannedPDFParser(
            layout_detector=detector,
            ocr_engine=FakeOCREngine(),
        )
        full_image = _make_image(h=300, w=200)
        _patched_parse(parser, images=[full_image])

        assert detector.last_image is not None
        assert detector.last_image.shape == (300, 200, 3)


# ---------------------------------------------------------------------------
# ScannedPDFParser — semaphore / parallelism
# ---------------------------------------------------------------------------
class TestScannedPDFParserParallelism:
    def test_all_regions_processed_within_semaphore_limit(self) -> None:
        """With 5 regions and max_concurrent_regions=2, all 5 still complete."""
        from tests.conftest import FakeLayoutDetector, FakeOCREngine

        n_regions = 5
        regions = [_make_region() for _ in range(n_regions)]
        ocr_engine = FakeOCREngine(lines=[_make_line()])
        parser = ScannedPDFParser(
            layout_detector=FakeLayoutDetector(regions=regions),
            ocr_engine=ocr_engine,
        )

        with patch("doc_intelligence.pdf.parser.settings") as mock_settings:
            mock_settings.max_concurrent_regions = 2
            with patch(
                "doc_intelligence.pdf.parser._render_pdf_to_images",
                return_value=[_make_image()],
            ):
                result = parser.parse(PDFDocument(uri="scan.pdf"))

        assert result.content is not None
        assert len(result.content.pages[0].blocks) == n_regions

    def test_max_concurrent_ocr_calls_respects_semaphore(self) -> None:
        """Concurrent OCR calls must never exceed max_concurrent_regions."""
        from tests.conftest import FakeLayoutDetector

        semaphore_limit = 2
        n_regions = 6
        concurrent_count = 0
        max_observed = 0
        lock = threading.Lock()

        class TrackingOCREngine(BaseOCREngine):
            def ocr(self, region_image: np.ndarray) -> list[Line]:
                nonlocal concurrent_count, max_observed
                with lock:
                    concurrent_count += 1
                    max_observed = max(max_observed, concurrent_count)
                time.sleep(0.02)  # hold slot briefly
                with lock:
                    concurrent_count -= 1
                return []

        regions = [_make_region() for _ in range(n_regions)]
        parser = ScannedPDFParser(
            layout_detector=FakeLayoutDetector(regions=regions),
            ocr_engine=TrackingOCREngine(),
        )

        with patch("doc_intelligence.pdf.parser.settings") as mock_settings:
            mock_settings.max_concurrent_regions = semaphore_limit
            with patch(
                "doc_intelligence.pdf.parser._render_pdf_to_images",
                return_value=[_make_image()],
            ):
                parser.parse(PDFDocument(uri="scan.pdf"))

        assert max_observed <= semaphore_limit


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


# ---------------------------------------------------------------------------
# _crop
# ---------------------------------------------------------------------------
class TestCrop:
    def test_crop_extracts_correct_subregion(self) -> None:
        image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
        bbox = BoundingBox(x0=2.0, top=3.0, x1=7.0, bottom=8.0)

        cropped = _crop(image, bbox)

        np.testing.assert_array_equal(cropped, image[3:8, 2:7])

    def test_crop_shape_matches_bbox(self) -> None:
        image = _make_image(h=200, w=150)
        bbox = BoundingBox(x0=10.0, top=20.0, x1=60.0, bottom=70.0)

        cropped = _crop(image, bbox)

        assert cropped.shape == (50, 50, 3)

    def test_crop_truncates_float_coords(self) -> None:
        image = _make_image(h=100, w=100)
        bbox = BoundingBox(x0=1.9, top=2.7, x1=51.9, bottom=52.7)

        cropped = _crop(image, bbox)

        # int(1.9)=1, int(2.7)=2, int(51.9)=51, int(52.7)=52
        assert cropped.shape == (50, 50, 3)

    def test_crop_full_image(self) -> None:
        image = _make_image(h=100, w=80)
        bbox = BoundingBox(x0=0.0, top=0.0, x1=80.0, bottom=100.0)

        cropped = _crop(image, bbox)

        np.testing.assert_array_equal(cropped, image)

    def test_crop_preserves_channels(self) -> None:
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[30:50, 20:40] = [10, 20, 30]
        bbox = BoundingBox(x0=20.0, top=30.0, x1=40.0, bottom=50.0)

        cropped = _crop(image, bbox)

        assert cropped.shape == (20, 20, 3)
        np.testing.assert_array_equal(cropped[0, 0], [10, 20, 30])
