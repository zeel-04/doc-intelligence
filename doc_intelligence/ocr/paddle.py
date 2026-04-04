"""PaddleOCR implementations of BaseLayoutDetector and BaseOCREngine.

Both classes use deferred imports so the module is importable without PaddleOCR
installed. Install the ``ocr`` optional dependency group to use them:

    uv sync --extra ocr

Compatible with PaddleOCR v3.x which uses the ``predict()`` API,
``LayoutDetection`` for layout analysis, and ``PaddleOCR`` for text recognition.
"""

from typing import Any

import numpy as np

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.schemas.core import BoundingBox, Line


class PaddleLayoutDetector(BaseLayoutDetector):
    """Layout detector backed by PaddleOCR's ``LayoutDetection`` model.

    Segments a page image into typed regions (text, table, figure, etc.) using
    PaddleOCR's document layout analysis model.  Bounding boxes are returned in
    pixel coordinates relative to the input page image.

    Args:
        model_name: PaddleOCR layout model name
            (default ``"PP-DocLayout_plus-L"``).
        **kwargs: Extra keyword arguments forwarded to ``LayoutDetection()``.
    """

    def __init__(
        self,
        model_name: str = "PP-DocLayout_plus-L",
        **kwargs: Any,
    ) -> None:
        from paddleocr import (
            LayoutDetection,  # type: ignore[missing-import]  # noqa: PLC0415
        )

        self._engine = LayoutDetection(model_name=model_name, **kwargs)

    def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
        """Detect layout regions in a page image.

        Args:
            page_image: An HxWxC uint8 numpy array representing the full page.

        Returns:
            A list of detected regions with pixel-coordinate bounding boxes,
            type labels, and confidence scores.
        """
        regions: list[LayoutRegion] = []
        for res in self._engine.predict(page_image, batch_size=1):
            boxes = res.json["res"]["boxes"]
            regions.extend(self._to_layout_region(box) for box in boxes)
        return regions

    def _to_layout_region(self, raw: dict[str, Any]) -> LayoutRegion:
        """Convert a single LayoutDetection result dict to a LayoutRegion.

        Args:
            raw: A result dict with keys ``coordinate``, ``label``, and
                ``score``.

        Returns:
            A ``LayoutRegion`` with pixel-coordinate bounding box.
        """
        x0, y0, x1, y1 = raw["coordinate"]
        return LayoutRegion(
            bounding_box=BoundingBox(
                x0=float(x0),
                top=float(y0),
                x1=float(x1),
                bottom=float(y1),
            ),
            region_type=raw["label"],
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

        self._engine = PaddleOCR(lang=lang, **kwargs)

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
        lines: list[Line] = []
        for res in self._engine.predict(region_image):
            inner = res.json["res"]
            rec_texts = inner.get("rec_texts", [])
            rec_boxes = inner.get("rec_boxes", [])
            if not rec_texts:
                return []
            for text, box in zip(rec_texts, rec_boxes):
                lines.append(self._to_line(text, box, w, h))
        return lines

    def _to_line(self, text: str, box: list[int], width: int, height: int) -> Line:
        """Convert a PaddleOCR v3 result item to a Line.

        PaddleOCR v3 returns ``rec_boxes`` as ``[x_min, y_min, x_max, y_max]``
        in pixel coordinates.  These are normalized by the region image
        dimensions.

        Args:
            text: Recognised text string.
            box: Bounding box as ``[x_min, y_min, x_max, y_max]`` pixels.
            width: Width of the region image in pixels.
            height: Height of the region image in pixels.

        Returns:
            A ``Line`` with normalized bounding box.
        """
        x0, y0, x1, y1 = box
        return Line(
            text=text,
            bounding_box=BoundingBox(
                x0=float(x0) / width,
                top=float(y0) / height,
                x1=float(x1) / width,
                bottom=float(y1) / height,
            ),
        )
