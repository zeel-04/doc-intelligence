"""OCR package for scanned PDF processing.

Public surface:

- :class:`BaseLayoutDetector` / :class:`BaseOCREngine` — ABCs for custom implementations
- :class:`LayoutRegion` — schema returned by layout detectors
- :class:`PaddleLayoutDetector` / :class:`PaddleOCREngine` — default implementations
  (require the ``ocr`` optional dependency group)
- :class:`~doc_intelligence.pdf.ocr_parser.ScannedPDFParser` — parser that wires
  layout detection + OCR into a ``PDFDocument``
"""

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.ocr.paddle import PaddleLayoutDetector, PaddleOCREngine
from doc_intelligence.pdf.ocr_parser import ScannedPDFParser

__all__ = [
    # Abstract bases
    "BaseLayoutDetector",
    "BaseOCREngine",
    "LayoutRegion",
    # Default implementations
    "PaddleLayoutDetector",
    "PaddleOCREngine",
    # Parser
    "ScannedPDFParser",
]
