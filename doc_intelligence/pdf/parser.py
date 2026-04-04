"""PDF parsers for digital and scanned documents.

``DigitalPDFParser`` uses pdfplumber to extract text from native digital PDFs.

``ScannedPDFParser`` renders each page to a numpy image, runs a layout
detector to segment the page into typed regions, then runs OCR on every
region concurrently within each page.  Pages are processed sequentially.
Both parsers produce the same ``PDFDocument`` schema, so the existing
formatter and extractor work unchanged.
"""

import asyncio
from abc import abstractmethod
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import pdfplumber
import pypdfium2 as pdfium
import requests

from doc_intelligence.base import BaseParser
from doc_intelligence.config import settings
from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.pdf.schemas import (
    PDF,
    Cell,
    ContentBlock,
    Page,
    PDFDocument,
    TableBlock,
    TextBlock,
)
from doc_intelligence.schemas.core import BoundingBox, Line
from doc_intelligence.utils import normalize_bounding_box


class PDFParser(BaseParser[PDFDocument]):
    @abstractmethod
    def parse(self, document: PDFDocument) -> PDFDocument:
        pass


# ---------------------------------------------------------------------------
# DigitalPDFParser
# ---------------------------------------------------------------------------


class DigitalPDFParser(PDFParser):
    def parse(self, document: PDFDocument) -> PDFDocument:
        pages = []

        # Check if URI is a URL or local path
        parsed = urlparse(document.uri)
        if parsed.scheme in ("http", "https"):
            # Download the PDF from URL
            response = requests.get(document.uri)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
        else:
            # Use local file path
            pdf_file = document.uri

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                lines = []
                for line in page.extract_text_lines(return_chars=False):
                    bbox = normalize_bounding_box(
                        BoundingBox(
                            x0=line["x0"],
                            top=line["top"],
                            x1=line["x1"],
                            bottom=line["bottom"],
                        ),
                        page.width,
                        page.height,
                    )
                    lines.append(Line(text=line["text"], bounding_box=bbox))
                blocks = [TextBlock(lines=lines)] if lines else []
                pages.append(Page(blocks=blocks, width=page.width, height=page.height))
        return PDFDocument(uri=document.uri, content=PDF(pages=pages))


# ---------------------------------------------------------------------------
# ScannedPDFParser
# ---------------------------------------------------------------------------


class ScannedPDFParser(BaseParser[PDFDocument]):
    """Parser for scanned PDFs using layout detection and OCR.

    Renders each page to a numpy image, detects layout regions, then OCR's
    every region concurrently (bounded by ``settings.max_concurrent_regions``).
    Produces the same ``PDFDocument`` schema as ``DigitalPDFParser``.

    Args:
        layout_detector: Detects layout regions in page images.
        ocr_engine: Runs OCR on cropped region images.
        dpi: Rendering resolution in dots per inch (default 150).
    """

    def __init__(
        self,
        layout_detector: BaseLayoutDetector,
        ocr_engine: BaseOCREngine,
        dpi: int = 150,
    ) -> None:
        self._layout_detector = layout_detector
        self._ocr_engine = ocr_engine
        self._dpi = dpi

    def parse(self, document: PDFDocument) -> PDFDocument:
        """Parse a scanned PDF into a PDFDocument with populated content.

        Args:
            document: A ``PDFDocument`` whose ``uri`` is a local path or
                HTTP(S) URL of the scanned PDF.

        Returns:
            A new ``PDFDocument`` with ``content`` populated from OCR results.
        """
        images = _render_pdf_to_images(document.uri, self._dpi)
        pages = asyncio.run(self._parse_all_pages(images))
        return PDFDocument(uri=document.uri, content=PDF(pages=pages))

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _parse_all_pages(self, images: list[np.ndarray]) -> list[Page]:
        """Process pages sequentially, with per-page region parallelism.

        Args:
            images: Ordered list of page images.

        Returns:
            Ordered list of assembled ``Page`` objects.
        """
        pages: list[Page] = []
        for image in images:
            pages.append(await self._parse_page(image))
        return pages

    async def _parse_page(self, image: np.ndarray) -> Page:
        """Detect regions and run OCR concurrently within a single page.

        Args:
            image: HxWxC uint8 numpy array of the full page.

        Returns:
            A ``Page`` with blocks assembled from OCR output.
        """
        h, w = image.shape[:2]
        regions = self._layout_detector.detect(image)
        semaphore = asyncio.Semaphore(settings.max_concurrent_regions)
        blocks: list[ContentBlock] = list(
            await asyncio.gather(
                *[self._process_region(image, r, semaphore) for r in regions]
            )
        )
        return Page(blocks=blocks, width=w, height=h)

    async def _process_region(
        self,
        image: np.ndarray,
        region: LayoutRegion,
        semaphore: asyncio.Semaphore,
    ) -> ContentBlock:
        """Crop, OCR, and assemble one layout region into a ContentBlock.

        Text regions become ``TextBlock``; table regions become ``TableBlock``
        where each OCR line is stored as a single-cell row.

        Args:
            image: Full page image.
            region: Layout region with pixel-coordinate bounding box.
            semaphore: Shared semaphore capping concurrent OCR calls.

        Returns:
            A ``TextBlock`` or ``TableBlock``.
        """
        async with semaphore:
            cropped = _crop(image, region.bounding_box)
            lines = await asyncio.to_thread(self._ocr_engine.ocr, cropped)

        if region.region_type == "table":
            rows = [
                [Cell(text=line.text, bounding_box=line.bounding_box)] for line in lines
            ]
            return TableBlock(bounding_box=region.bounding_box, rows=rows)
        return TextBlock(bounding_box=region.bounding_box, lines=lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _render_pdf_to_images(uri: str, dpi: int) -> list[np.ndarray]:
    """Render each page of a PDF to a numpy array.

    Downloads the PDF first if ``uri`` is an HTTP(S) URL.

    Args:
        uri: Local file path or HTTP(S) URL of the PDF.
        dpi: Rendering resolution in dots per inch.

    Returns:
        Ordered list of HxWxC uint8 numpy arrays, one per page.

    Raises:
        requests.HTTPError: If ``uri`` is a URL and the request fails.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        response = requests.get(uri)
        response.raise_for_status()
        source: str | bytes = response.content
    else:
        source = uri

    scale = dpi / 72.0  # PDF uses 72 points per inch
    pdf = pdfium.PdfDocument(source)
    images: list[np.ndarray] = []
    for page in pdf:
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        images.append(np.array(pil_image))
    return images


def _crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crop a region from a page image using a pixel-coordinate BoundingBox.

    Args:
        image: HxWxC numpy array of the full page.
        bbox: Pixel-coordinate bounding box of the region to crop.

    Returns:
        Cropped sub-image as a numpy array.
    """
    return image[int(bbox.top) : int(bbox.bottom), int(bbox.x0) : int(bbox.x1)]
