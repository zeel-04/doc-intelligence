"""OCR package: layout detection and text recognition engines.

Public surface:

- :class:`BaseLayoutDetector` / :class:`BaseOCREngine` — ABCs for custom implementations
- :class:`LayoutRegion` — schema returned by layout detectors
"""

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion

__all__ = [
    "BaseLayoutDetector",
    "BaseOCREngine",
    "LayoutRegion",
]
