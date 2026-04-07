"""PDF parser with pluggable strategy (digital or scanned).

``PDFParser`` uses a :class:`ParseStrategy` to decide how to extract content:

- **DIGITAL** — uses pdfplumber to extract text from native digital PDFs.
  Each text line becomes its own ``TextBlock`` so that block-level citation
  addressing gives line-level precision for digital documents.
- **SCANNED** — supports two sub-pipelines selected via
  :class:`ScannedPipelineType`:

  - **TWO_STAGE** (Type 1) — *not yet implemented*. Will render each page
    to a numpy image, run a layout detector to segment into typed regions,
    then run OCR on every region concurrently.
  - **VLM** (Type 2) — sends page images to a vision-capable LLM that
    performs layout detection and OCR in a single call per page batch.

Both strategies produce the same ``PDFDocument`` schema, so the existing
formatter and extractor work unchanged.
"""

import base64
import json
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import pdfplumber
import pypdfium2 as pdfium
import requests
from loguru import logger

from doc_intelligence.base import BaseLLM, BaseParser
from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine
from doc_intelligence.pdf.schemas import (
    PDF,
    PDFDocument,
)
from doc_intelligence.pdf.types import ParseStrategy, ScannedPipelineType
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
from doc_intelligence.utils import normalize_bounding_box

# Region types that map to ImageBlock or ChartBlock and are skipped during
# formatting.  Layout detectors may use varying label vocabularies — expand
# these sets as needed.
_IMAGE_REGION_TYPES = frozenset({"image", "figure", "picture", "photo"})
_CHART_REGION_TYPES = frozenset({"chart", "diagram", "plot", "graph"})


class PDFParser(BaseParser[PDFDocument]):
    """Unified PDF parser with strategy selection.

    Picks the right parsing path (digital text extraction or OCR-based
    scanning) based on the ``strategy`` argument.  When ``strategy`` is
    ``SCANNED``, the ``scanned_pipeline`` parameter selects which
    sub-pipeline to use:

    - ``TWO_STAGE`` — *not yet implemented*. Reserved for a future
      layout-detection + OCR pipeline.
    - ``VLM`` (default) — requires ``llm`` (a vision-capable
      :class:`BaseLLM`).

    Args:
        strategy: Parsing strategy — ``DIGITAL`` (default) or ``SCANNED``.
        scanned_pipeline: Sub-pipeline for scanned strategy —
            ``VLM`` (default) or ``TWO_STAGE`` (not yet implemented).
        layout_detector: A :class:`BaseLayoutDetector` implementation.
            Reserved for ``TWO_STAGE`` (not yet implemented).
        ocr_engine: A :class:`BaseOCREngine` implementation.
            Reserved for ``TWO_STAGE`` (not yet implemented).
        llm: A vision-capable :class:`BaseLLM` instance.
            Required when ``scanned_pipeline`` is ``VLM``.
        dpi: Page rendering resolution in dots per inch (default 150).
            Only used for scanned PDFs.
        vlm_batch_size: Number of pages per VLM call (default 1).
            Only used when ``scanned_pipeline`` is ``VLM``.

    Raises:
        ValueError: If required arguments for the selected scanned
            sub-pipeline are missing.
        NotImplementedError: If ``scanned_pipeline`` is ``TWO_STAGE``.
    """

    def __init__(
        self,
        strategy: ParseStrategy = ParseStrategy.DIGITAL,
        scanned_pipeline: ScannedPipelineType = ScannedPipelineType.VLM,
        layout_detector: BaseLayoutDetector | None = None,
        ocr_engine: BaseOCREngine | None = None,
        llm: BaseLLM | None = None,
        dpi: int = 150,
        vlm_batch_size: int = 1,
    ) -> None:
        self._strategy = strategy
        self._scanned_pipeline = scanned_pipeline
        self._layout_detector = layout_detector
        self._ocr_engine = ocr_engine
        self._llm = llm
        self._dpi = dpi
        self._vlm_batch_size = vlm_batch_size

        if strategy == ParseStrategy.SCANNED:
            if scanned_pipeline == ScannedPipelineType.TWO_STAGE:
                raise NotImplementedError(
                    "TWO_STAGE scanned pipeline is not yet implemented. "
                    "Use ScannedPipelineType.VLM instead."
                )
            elif scanned_pipeline == ScannedPipelineType.VLM:
                if llm is None:
                    raise ValueError(
                        "llm is required for scanned PDFs with VLM pipeline "
                        "— supply a vision-capable BaseLLM instance."
                    )

    def parse(self, uri: str) -> PDFDocument:
        """Parse a PDF into a structured PDFDocument.

        Args:
            uri: Local file path or HTTP(S) URL of the PDF.

        Returns:
            A ``PDFDocument`` with ``content`` populated.
        """
        if self._strategy == ParseStrategy.DIGITAL:
            return self._parse_digital(uri)
        elif self._strategy == ParseStrategy.SCANNED:
            return self._parse_scanned(uri)
        else:
            raise ValueError(f"Unsupported parse strategy: {self._strategy}")

    # ------------------------------------------------------------------
    # Digital parsing (pdfplumber)
    # ------------------------------------------------------------------

    def _parse_digital(self, uri: str) -> PDFDocument:
        """Extract text from a native digital PDF using pdfplumber."""
        pages = []

        parsed = urlparse(uri)
        if parsed.scheme in ("http", "https"):
            response = requests.get(uri)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
        else:
            pdf_file = uri

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                blocks: list[ContentBlock] = []
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
                    blocks.append(
                        TextBlock(
                            lines=[Line(text=line["text"], bounding_box=bbox)],
                            bounding_box=bbox,
                        )
                    )
                pages.append(Page(blocks=blocks, width=page.width, height=page.height))
        return PDFDocument(uri=uri, content=PDF(pages=pages))

    # ------------------------------------------------------------------
    # Scanned parsing (dispatches to sub-pipeline)
    # ------------------------------------------------------------------

    def _parse_scanned(self, uri: str) -> PDFDocument:
        """Dispatch to the appropriate scanned sub-pipeline."""
        if self._scanned_pipeline == ScannedPipelineType.VLM:
            return self._parse_scanned_vlm(uri)
        return self._parse_scanned_two_stage(uri)

    # ------------------------------------------------------------------
    # Scanned: Type 1 — Two-stage (layout detection + OCR)
    # ------------------------------------------------------------------

    def _parse_scanned_two_stage(self, uri: str) -> PDFDocument:
        """Parse a scanned PDF using layout detection and OCR.

        .. note:: Not yet implemented. Raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "TWO_STAGE scanned pipeline is not yet implemented. "
            "Use ScannedPipelineType.VLM instead."
        )

    # ------------------------------------------------------------------
    # Scanned: Type 2 — VLM (single-stage vision LLM)
    # ------------------------------------------------------------------

    def _parse_scanned_vlm(self, uri: str) -> PDFDocument:
        """Parse a scanned PDF using a vision-capable LLM.

        Sends page images to the LLM in batches and parses the structured
        JSON response into ``Page`` objects.
        """
        assert self._llm is not None

        images = _render_pdf_to_images(uri, self._dpi)
        if not images:
            return PDFDocument(uri=uri, content=PDF(pages=[]))

        pages: list[Page] = []
        for batch_start in range(0, len(images), self._vlm_batch_size):
            batch_images = images[batch_start : batch_start + self._vlm_batch_size]
            batch_data_urls = [_encode_image_to_data_url(img) for img in batch_images]
            batch_dimensions = [(img.shape[1], img.shape[0]) for img in batch_images]

            schema_text = json.dumps(_VLM_RESPONSE_SCHEMA, indent=2)
            user_prompt = _VLM_USER_PROMPT.format(
                schema=schema_text,
                num_pages=len(batch_images),
            )

            logger.debug(
                f"VLM parse: batch starting at page {batch_start}, "
                f"{len(batch_images)} page(s)"
            )
            raw_response = self._llm.generate(
                system_prompt=_VLM_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=batch_data_urls,
            )
            batch_pages = _parse_vlm_response(raw_response, batch_dimensions)
            pages.extend(batch_pages)

        return PDFDocument(uri=uri, content=PDF(pages=pages))


# ---------------------------------------------------------------------------
# VLM prompts and response parsing
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = """\
You are a document layout analysis and OCR engine. Given image(s) of \
document pages, detect every content region and return structured JSON.

For each region, provide:
- block_type: one of [text, table, figure, header, footer, page_number, \
caption, list, formula]
- bbox: bounding box as {x0, y0, x1, y1} in pixel coordinates
- text: the OCR'd text content of that region. For tables, use markdown \
pipe-delimited format. For figures, describe the figure briefly.

Rules:
- Preserve reading order (top-to-bottom, left-to-right).
- Every piece of visible text must belong to exactly one block.
- Do NOT merge unrelated regions. Headers, footers, and page numbers \
are separate blocks.
- For multi-column layouts, read left column fully before right column.
- Be thorough: do not skip any content on the page.\
"""

_VLM_USER_PROMPT = """\
Analyse {num_pages} document page(s). Detect all layout regions and OCR \
the text. Return JSON matching this schema exactly:

{schema}

Return ONLY the JSON object, no markdown fences or explanation.\
"""

_VLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_index": {"type": "integer"},
                    "blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "block_type": {
                                    "type": "string",
                                    "enum": [
                                        "text",
                                        "table",
                                        "figure",
                                        "header",
                                        "footer",
                                        "page_number",
                                        "caption",
                                        "list",
                                        "formula",
                                    ],
                                },
                                "bbox": {
                                    "type": "object",
                                    "properties": {
                                        "x0": {"type": "integer"},
                                        "y0": {"type": "integer"},
                                        "x1": {"type": "integer"},
                                        "y1": {"type": "integer"},
                                    },
                                    "required": ["x0", "y0", "x1", "y1"],
                                },
                                "text": {"type": "string"},
                            },
                            "required": ["block_type", "bbox", "text"],
                        },
                    },
                },
                "required": ["page_index", "blocks"],
            },
        },
    },
    "required": ["pages"],
}

# VLM block_type → region type mapping for reuse of existing constants
_VLM_TABLE_TYPES = frozenset({"table"})
_VLM_TEXT_TYPES = frozenset(
    {"text", "header", "footer", "page_number", "caption", "list", "formula"}
)


def _parse_vlm_response(
    raw_json: str,
    page_dimensions: list[tuple[int, int]],
) -> list[Page]:
    """Parse a VLM JSON response into a list of ``Page`` objects.

    Args:
        raw_json: Raw JSON string from the VLM.
        page_dimensions: ``(width, height)`` for each page in the batch.

    Returns:
        Ordered list of ``Page`` objects.
    """
    # Strip markdown fences if the model wrapped the JSON
    text = raw_json.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)

    # Handle varying response shapes
    if isinstance(data, list):
        pages_data = data
    elif "pages" in data:
        pages_data = data["pages"]
    elif "blocks" in data:
        # Single page returned without wrapper
        pages_data = [{"page_index": 0, "blocks": data["blocks"]}]
    else:
        pages_data = []

    pages: list[Page] = []
    for i, page_data in enumerate(pages_data):
        w, h = page_dimensions[i] if i < len(page_dimensions) else (0, 0)
        blocks: list[ContentBlock] = []

        for block_data in page_data.get("blocks", []):
            btype = block_data.get("block_type", "text").lower()
            bbox_data = block_data.get("bbox", {})
            bbox = BoundingBox(
                x0=bbox_data.get("x0", 0),
                top=bbox_data.get("y0", 0),
                x1=bbox_data.get("x1", 0),
                bottom=bbox_data.get("y1", 0),
            )
            if w > 0 and h > 0:
                bbox = normalize_bounding_box(bbox, w, h)
            text = block_data.get("text", "")

            if btype in _IMAGE_REGION_TYPES:
                blocks.append(ImageBlock(bounding_box=bbox, description=text))
            elif btype in _CHART_REGION_TYPES:
                blocks.append(ChartBlock(bounding_box=bbox, description=text))
            elif btype in _VLM_TABLE_TYPES:
                # Parse markdown table into rows
                rows: list[list[Cell]] = []
                for line_text in text.split("\n"):
                    line_text = line_text.strip()
                    if (
                        not line_text
                        or line_text.startswith("---")
                        or line_text.startswith("|-")
                    ):
                        continue
                    cells = [
                        Cell(text=c.strip()) for c in line_text.strip("|").split("|")
                    ]
                    rows.append(cells)
                blocks.append(TableBlock(bounding_box=bbox, rows=rows))
            else:
                # Text-like types: split into lines
                # VLM provides one bbox per block — assign it to each line
                lines = [
                    Line(text=ln, bounding_box=bbox)
                    for ln in text.split("\n")
                    if ln.strip()
                ]
                if lines:
                    blocks.append(TextBlock(bounding_box=bbox, lines=lines))

        pages.append(Page(blocks=blocks, width=w, height=h))

    return pages


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


def _encode_image_to_data_url(image: np.ndarray) -> str:
    """Encode a numpy image array as a base64 PNG data URL.

    Args:
        image: HxWxC uint8 numpy array.

    Returns:
        A ``data:image/png;base64,...`` string.
    """
    from PIL import Image

    pil_image = Image.fromarray(image)
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
