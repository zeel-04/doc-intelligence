"""Tests for ocr.base module."""

import numpy as np
import pytest

from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.schemas.core import BoundingBox, Line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image(h: int = 100, w: int = 80) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_bbox() -> BoundingBox:
    return BoundingBox(x0=0.1, top=0.2, x1=0.5, bottom=0.4)


def _make_line(text: str = "hello") -> Line:
    return Line(text=text, bounding_box=_make_bbox())


# ---------------------------------------------------------------------------
# LayoutRegion
# ---------------------------------------------------------------------------
class TestLayoutRegion:
    def test_fields_are_stored(self) -> None:
        bbox = _make_bbox()
        region = LayoutRegion(bounding_box=bbox, region_type="text", confidence=0.95)

        assert region.bounding_box == bbox
        assert region.region_type == "text"
        assert region.confidence == 0.95

    def test_arbitrary_region_type(self) -> None:
        region = LayoutRegion(
            bounding_box=_make_bbox(), region_type="custom_label", confidence=0.5
        )
        assert region.region_type == "custom_label"

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_boundary_values(self, confidence: float) -> None:
        region = LayoutRegion(
            bounding_box=_make_bbox(), region_type="text", confidence=confidence
        )
        assert region.confidence == confidence


# ---------------------------------------------------------------------------
# BaseLayoutDetector
# ---------------------------------------------------------------------------
class TestBaseLayoutDetector:
    def test_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLayoutDetector()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self) -> None:
        class ConcreteDetector(BaseLayoutDetector):
            def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
                return []

        detector = ConcreteDetector()
        assert isinstance(detector, BaseLayoutDetector)

    def test_detect_receives_image_and_returns_regions(self) -> None:
        expected = [
            LayoutRegion(bounding_box=_make_bbox(), region_type="text", confidence=0.9)
        ]

        class ConcreteDetector(BaseLayoutDetector):
            def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
                return expected

        image = _make_image()
        detector = ConcreteDetector()
        result = detector.detect(image)

        assert result == expected

    def test_detect_called_with_correct_image(self) -> None:
        received: list[np.ndarray] = []

        class ConcreteDetector(BaseLayoutDetector):
            def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
                received.append(page_image)
                return []

        image = _make_image(200, 150)
        ConcreteDetector().detect(image)

        assert len(received) == 1
        assert received[0].shape == (200, 150, 3)

    def test_missing_detect_raises_type_error(self) -> None:
        class IncompleteDetector(BaseLayoutDetector):
            pass

        with pytest.raises(TypeError):
            IncompleteDetector()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# BaseOCREngine
# ---------------------------------------------------------------------------
class TestBaseOCREngine:
    def test_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseOCREngine()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self) -> None:
        class ConcreteEngine(BaseOCREngine):
            def ocr(self, region_image: np.ndarray) -> list[Line]:
                return []

        engine = ConcreteEngine()
        assert isinstance(engine, BaseOCREngine)

    def test_ocr_receives_image_and_returns_lines(self) -> None:
        expected = [_make_line("first"), _make_line("second")]

        class ConcreteEngine(BaseOCREngine):
            def ocr(self, region_image: np.ndarray) -> list[Line]:
                return expected

        image = _make_image()
        result = ConcreteEngine().ocr(image)

        assert result == expected

    def test_ocr_called_with_correct_image(self) -> None:
        received: list[np.ndarray] = []

        class ConcreteEngine(BaseOCREngine):
            def ocr(self, region_image: np.ndarray) -> list[Line]:
                received.append(region_image)
                return []

        image = _make_image(50, 60)
        ConcreteEngine().ocr(image)

        assert len(received) == 1
        assert received[0].shape == (50, 60, 3)

    def test_missing_ocr_raises_type_error(self) -> None:
        class IncompleteEngine(BaseOCREngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FakeLayoutDetector / FakeOCREngine (conftest fakes)
# ---------------------------------------------------------------------------
class TestConfTestFakes:
    def test_fake_layout_detector_tracks_calls(
        self,
        fake_layout_detector: BaseLayoutDetector,
        sample_layout_region: LayoutRegion,
    ) -> None:
        from tests.conftest import FakeLayoutDetector

        assert isinstance(fake_layout_detector, FakeLayoutDetector)
        image = _make_image()
        result = fake_layout_detector.detect(image)

        assert result == [sample_layout_region]
        assert fake_layout_detector.call_count == 1  # type: ignore[attr-defined]
        assert fake_layout_detector.last_image is image  # type: ignore[attr-defined]

    def test_fake_ocr_engine_tracks_calls(
        self, fake_ocr_engine: BaseOCREngine, sample_lines: list[Line]
    ) -> None:
        from tests.conftest import FakeOCREngine

        assert isinstance(fake_ocr_engine, FakeOCREngine)
        image = _make_image()
        result = fake_ocr_engine.ocr(image)

        assert result == sample_lines
        assert fake_ocr_engine.call_count == 1  # type: ignore[attr-defined]
        assert fake_ocr_engine.last_image is image  # type: ignore[attr-defined]

    def test_fake_layout_detector_empty_by_default(self) -> None:
        from tests.conftest import FakeLayoutDetector

        detector = FakeLayoutDetector()
        assert detector.detect(_make_image()) == []

    def test_fake_ocr_engine_empty_by_default(self) -> None:
        from tests.conftest import FakeOCREngine

        engine = FakeOCREngine()
        assert engine.ocr(_make_image()) == []
