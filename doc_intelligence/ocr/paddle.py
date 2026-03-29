"""PaddleOCR implementations of BaseLayoutDetector and BaseOCREngine.

Both classes use deferred imports so the module is importable without PaddleOCR
installed. Install the ``ocr`` optional dependency group to use them:

    uv sync --extra ocr
"""

from typing import Any

import numpy as np

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.pdf.schemas import Line
from doc_intelligence.schemas.core import BoundingBox


class PaddleLayoutDetector(BaseLayoutDetector):
    """Layout detector backed by PaddleOCR's PPStructure.

    Segments a page image into typed regions (text, table, figure, etc.) using
    PaddleOCR's document layout analysis model.  Bounding boxes are returned in
    pixel coordinates relative to the input page image.

    Args:
        **kwargs: Extra keyword arguments forwarded to ``PPStructure()``.
    """

    def __init__(self, **kwargs: Any) -> None:
        from paddleocr import (
            PPStructure,  # type: ignore[missing-import]  # noqa: PLC0415
        )

        self._engine = PPStructure(
            layout=True,
            table=False,
            ocr=False,
            show_log=False,
            **kwargs,
        )

    def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
        """Detect layout regions in a page image.

        Args:
            page_image: An HxWxC uint8 numpy array representing the full page.

        Returns:
            A list of detected regions with pixel-coordinate bounding boxes,
            type labels, and confidence scores.
        """
        results: list[dict[str, Any]] = self._engine(page_image)
        return [self._to_layout_region(r) for r in results]

    def _to_layout_region(self, raw: dict[str, Any]) -> LayoutRegion:
        """Convert a single PPStructure result dict to a LayoutRegion.

        Args:
            raw: A PPStructure result dict with keys ``bbox``, ``type``, and
                ``score``.

        Returns:
            A ``LayoutRegion`` with pixel-coordinate bounding box.
        """
        x0, y0, x1, y1 = raw["bbox"]
        return LayoutRegion(
            bounding_box=BoundingBox(
                x0=float(x0),
                top=float(y0),
                x1=float(x1),
                bottom=float(y1),
            ),
            region_type=raw["type"],
            confidence=float(raw["score"]),
        )


class PaddleOCREngine(BaseOCREngine):
    """OCR engine backed by PaddleOCR.

    Reads text from a single cropped region image and returns structured lines
    with bounding boxes normalized to [0, 1] relative to the region dimensions.

    Args:
        lang: Language code passed to ``PaddleOCR()`` (default ``"en"``).
        **kwargs: Extra keyword arguments forwarded to ``PaddleOCR()``.
    """

    def __init__(self, lang: str = "en", **kwargs: Any) -> None:
        from paddleocr import PaddleOCR  # type: ignore[missing-import]  # noqa: PLC0415

        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            **kwargs,
        )

    def ocr(self, region_image: np.ndarray) -> list[Line]:
        """Run OCR on a single cropped region image.

        Args:
            region_image: An HxWxC uint8 numpy array of a cropped page region.

        Returns:
            A list of lines with text and bounding boxes normalized to [0, 1]
            relative to the region image dimensions.  Returns an empty list
            when no text is detected.
        """
        h, w = region_image.shape[:2]
        results: list[Any] | None = self._engine.ocr(region_image, cls=True)
        if not results or results[0] is None:
            return []
        return [self._to_line(item, w, h) for item in results[0]]

    def _to_line(self, raw: Any, width: int, height: int) -> Line:
        """Convert a single PaddleOCR result item to a Line.

        PaddleOCR represents each text line as a 2-element sequence:
        ``[polygon_points, (text, confidence)]`` where ``polygon_points`` is
        ``[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]`` in pixel coordinates.

        The four-point polygon is converted to an axis-aligned bounding box,
        then normalized by the region image dimensions.

        Args:
            raw: A PaddleOCR result item.
            width: Width of the region image in pixels.
            height: Height of the region image in pixels.

        Returns:
            A ``Line`` with normalized bounding box.
        """
        polygon, (text, _confidence) = raw
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        return Line(
            text=text,
            bounding_box=BoundingBox(
                x0=min(xs) / width,
                top=min(ys) / height,
                x1=max(xs) / width,
                bottom=max(ys) / height,
            ),
        )
