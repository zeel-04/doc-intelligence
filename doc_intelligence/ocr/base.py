"""Abstract base classes and shared schemas for OCR components."""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from doc_intelligence.pdf.schemas import Line
from doc_intelligence.schemas.core import BoundingBox


class LayoutRegion(BaseModel):
    """A detected region on a page image.

    Attributes:
        bounding_box: Pixel coordinates of the region within the page image.
        region_type: Label describing the region content (e.g. "text", "table",
            "header", "figure"). Plain string so third-party detectors can use
            their own label vocabulary.
        confidence: Detection confidence score in the range [0, 1].
    """

    bounding_box: BoundingBox
    region_type: str
    confidence: float


class BaseLayoutDetector(ABC):
    """Abstract base for page layout detectors.

    A layout detector segments a page image into typed regions (text, table,
    figure, etc.) before OCR is applied per region.
    """

    @abstractmethod
    def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
        """Detect regions in a page image.

        Args:
            page_image: An HxWxC uint8 numpy array representing the full page.

        Returns:
            A list of detected regions with bounding boxes, type labels, and
            confidence scores.
        """


class BaseOCREngine(ABC):
    """Abstract base for OCR engines.

    An OCR engine reads text from a single cropped region image and returns
    structured lines with normalized bounding boxes.
    """

    @abstractmethod
    def ocr(self, region_image: np.ndarray) -> list[Line]:
        """Run OCR on a single cropped region image.

        Args:
            region_image: An HxWxC uint8 numpy array of a cropped page region.

        Returns:
            A list of lines with text and bounding boxes normalized to [0, 1]
            relative to the region image dimensions.
        """
