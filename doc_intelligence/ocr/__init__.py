"""OCR package: layout detection and text recognition engines.

Public surface:

- :class:`BaseLayoutDetector` / :class:`BaseOCREngine` — ABCs for custom implementations
- :class:`LayoutRegion` — schema returned by layout detectors
- :class:`PaddleLayoutDetector` / :class:`PaddleOCREngine` — default implementations
  (require the ``ocr`` optional dependency group)
"""

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.ocr.paddle import PaddleLayoutDetector, PaddleOCREngine

__all__ = [
    # Abstract bases
    "BaseLayoutDetector",
    "BaseOCREngine",
    "LayoutRegion",
    # Default implementations
    "PaddleLayoutDetector",
    "PaddleOCREngine",
]
