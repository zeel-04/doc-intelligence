"""Tests for ocr.paddle module."""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from doc_intelligence.schemas.core import BoundingBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image(h: int = 200, w: int = 150) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_paddle_module() -> MagicMock:
    """Return a mock ``paddleocr`` module with PPStructure and PaddleOCR stubs."""
    mock = MagicMock()
    mock.PPStructure.return_value = MagicMock()
    mock.PaddleOCR.return_value = MagicMock()
    return mock


def _make_ppstructure_result(
    bbox: list[float],
    region_type: str,
    score: float,
) -> dict[str, Any]:
    """Build a PPStructure result dict."""
    return {"bbox": bbox, "type": region_type, "score": score}


def _make_paddle_ocr_result(
    polygon: list[list[float]],
    text: str,
    confidence: float,
) -> list[Any]:
    """Build a single PaddleOCR text-line result."""
    return [polygon, (text, confidence)]


# ---------------------------------------------------------------------------
# Module import (deferred-import contract)
# ---------------------------------------------------------------------------
class TestDeferredImport:
    def test_module_importable_without_paddleocr(self) -> None:
        """Importing the paddle module must not require PaddleOCR to be installed."""
        # Temporarily hide paddleocr from sys.modules if it happens to be present.
        saved = sys.modules.pop("paddleocr", None)
        try:
            import importlib

            import doc_intelligence.ocr.paddle as paddle_mod

            importlib.reload(paddle_mod)  # re-import without paddleocr available
        finally:
            if saved is not None:
                sys.modules["paddleocr"] = saved

    def test_instantiation_without_paddleocr_raises(self) -> None:
        """Instantiating PaddleLayoutDetector without PaddleOCR installed must raise."""
        saved = sys.modules.pop("paddleocr", None)
        try:
            from doc_intelligence.ocr.paddle import PaddleLayoutDetector

            with pytest.raises((ImportError, ModuleNotFoundError)):
                PaddleLayoutDetector()
        finally:
            if saved is not None:
                sys.modules["paddleocr"] = saved


# ---------------------------------------------------------------------------
# PaddleLayoutDetector
# ---------------------------------------------------------------------------
class TestPaddleLayoutDetector:
    # --- fixtures / helpers ---

    @staticmethod
    def _make_detector() -> tuple[Any, MagicMock]:
        """Return (detector_instance, ppstructure_mock_instance)."""
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        mock_module = _make_paddle_module()
        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            detector = PaddleLayoutDetector()
        return detector, mock_module.PPStructure.return_value

    # --- construction ---

    def test_ppstructure_instantiated_with_correct_args(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleLayoutDetector()

        mock_module.PPStructure.assert_called_once_with(
            layout=True,
            table=False,
            ocr=False,
            show_log=False,
        )

    def test_extra_kwargs_forwarded_to_ppstructure(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleLayoutDetector(lang="en")

        _, kwargs = mock_module.PPStructure.call_args
        assert kwargs["lang"] == "en"

    # --- detect() ---

    def test_detect_returns_empty_list_for_no_results(self) -> None:
        detector, engine_mock = self._make_detector()
        engine_mock.return_value = []
        image = _make_image()

        result = detector.detect(image)

        assert result == []

    def test_detect_converts_single_region(self) -> None:
        from doc_intelligence.ocr.base import LayoutRegion

        detector, engine_mock = self._make_detector()
        engine_mock.return_value = [
            _make_ppstructure_result([10.0, 20.0, 110.0, 80.0], "text", 0.95)
        ]

        regions = detector.detect(_make_image())

        assert len(regions) == 1
        r = regions[0]
        assert isinstance(r, LayoutRegion)
        assert r.region_type == "text"
        assert r.confidence == pytest.approx(0.95)
        assert r.bounding_box == BoundingBox(x0=10.0, top=20.0, x1=110.0, bottom=80.0)

    def test_detect_converts_multiple_regions(self) -> None:
        detector, engine_mock = self._make_detector()
        engine_mock.return_value = [
            _make_ppstructure_result([0.0, 0.0, 50.0, 30.0], "title", 0.99),
            _make_ppstructure_result([0.0, 40.0, 200.0, 180.0], "text", 0.87),
            _make_ppstructure_result([0.0, 190.0, 200.0, 300.0], "table", 0.75),
        ]

        regions = detector.detect(_make_image())

        assert len(regions) == 3
        assert regions[0].region_type == "title"
        assert regions[1].region_type == "text"
        assert regions[2].region_type == "table"

    def test_detect_passes_image_to_engine(self) -> None:
        detector, engine_mock = self._make_detector()
        engine_mock.return_value = []
        image = _make_image(300, 200)

        detector.detect(image)

        engine_mock.assert_called_once_with(image)

    # --- _to_layout_region() ---

    def test_to_layout_region_float_conversion(self) -> None:
        detector, _ = self._make_detector()
        raw = _make_ppstructure_result([10, 20, 110, 80], "figure", 0.6)

        region = detector._to_layout_region(raw)

        assert isinstance(region.bounding_box.x0, float)
        assert isinstance(region.bounding_box.top, float)
        assert isinstance(region.bounding_box.x1, float)
        assert isinstance(region.bounding_box.bottom, float)

    def test_to_layout_region_maps_all_fields(self) -> None:
        from doc_intelligence.ocr.base import LayoutRegion

        detector, _ = self._make_detector()
        raw = _make_ppstructure_result([5.5, 10.0, 200.0, 150.5], "header", 0.88)

        region = detector._to_layout_region(raw)

        assert isinstance(region, LayoutRegion)
        assert region.bounding_box == BoundingBox(
            x0=5.5, top=10.0, x1=200.0, bottom=150.5
        )
        assert region.region_type == "header"
        assert region.confidence == pytest.approx(0.88)

    @pytest.mark.parametrize(
        "region_type", ["text", "table", "figure", "title", "list"]
    )
    def test_to_layout_region_accepts_any_region_type(self, region_type: str) -> None:
        detector, _ = self._make_detector()
        raw = _make_ppstructure_result([0.0, 0.0, 100.0, 50.0], region_type, 0.9)

        region = detector._to_layout_region(raw)

        assert region.region_type == region_type


# ---------------------------------------------------------------------------
# PaddleOCREngine
# ---------------------------------------------------------------------------
class TestPaddleOCREngine:
    # --- fixtures / helpers ---

    @staticmethod
    def _make_engine() -> tuple[Any, MagicMock]:
        """Return (engine_instance, paddleocr_mock_instance)."""
        from doc_intelligence.ocr.paddle import PaddleOCREngine

        mock_module = _make_paddle_module()
        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            engine = PaddleOCREngine()
        return engine, mock_module.PaddleOCR.return_value

    # --- construction ---

    def test_paddleocr_instantiated_with_correct_args(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleOCREngine

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleOCREngine()

        mock_module.PaddleOCR.assert_called_once_with(
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )

    def test_custom_lang_forwarded(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleOCREngine

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleOCREngine(lang="ch")

        _, kwargs = mock_module.PaddleOCR.call_args
        assert kwargs["lang"] == "ch"

    def test_extra_kwargs_forwarded_to_paddleocr(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleOCREngine

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleOCREngine(use_gpu=True)

        _, kwargs = mock_module.PaddleOCR.call_args
        assert kwargs["use_gpu"] is True

    # --- ocr() — empty / None results ---

    def test_ocr_returns_empty_list_when_results_is_none(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.ocr.return_value = None
        assert engine.ocr(_make_image()) == []

    def test_ocr_returns_empty_list_when_results_is_empty_list(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.ocr.return_value = []
        assert engine.ocr(_make_image()) == []

    def test_ocr_returns_empty_list_when_page_result_is_none(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.ocr.return_value = [None]
        assert engine.ocr(_make_image()) == []

    def test_ocr_returns_empty_list_when_page_result_is_empty(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.ocr.return_value = [[]]
        assert engine.ocr(_make_image()) == []

    # --- ocr() — single line ---

    def test_ocr_single_line_text(self) -> None:
        engine, paddle_mock = self._make_engine()
        polygon = [[10.0, 20.0], [90.0, 20.0], [90.0, 40.0], [10.0, 40.0]]
        paddle_mock.ocr.return_value = [
            [_make_paddle_ocr_result(polygon, "hello world", 0.98)]
        ]

        lines = engine.ocr(_make_image(h=100, w=100))

        assert len(lines) == 1
        assert lines[0].text == "hello world"

    def test_ocr_single_line_bbox_is_normalized(self) -> None:
        engine, paddle_mock = self._make_engine()
        # image: 200h x 100w; polygon occupies 10–90 in x, 20–40 in y
        polygon = [[10.0, 20.0], [90.0, 20.0], [90.0, 40.0], [10.0, 40.0]]
        paddle_mock.ocr.return_value = [[_make_paddle_ocr_result(polygon, "text", 0.9)]]

        lines = engine.ocr(_make_image(h=200, w=100))

        bbox = lines[0].bounding_box
        assert bbox.x0 == pytest.approx(0.10)  # 10/100
        assert bbox.x1 == pytest.approx(0.90)  # 90/100
        assert bbox.top == pytest.approx(0.10)  # 20/200
        assert bbox.bottom == pytest.approx(0.20)  # 40/200

    # --- ocr() — multiple lines ---

    def test_ocr_multiple_lines(self) -> None:
        engine, paddle_mock = self._make_engine()
        poly1 = [[0.0, 0.0], [100.0, 0.0], [100.0, 10.0], [0.0, 10.0]]
        poly2 = [[0.0, 20.0], [100.0, 20.0], [100.0, 30.0], [0.0, 30.0]]
        poly3 = [[0.0, 40.0], [100.0, 40.0], [100.0, 50.0], [0.0, 50.0]]
        paddle_mock.ocr.return_value = [
            [
                _make_paddle_ocr_result(poly1, "first", 0.99),
                _make_paddle_ocr_result(poly2, "second", 0.95),
                _make_paddle_ocr_result(poly3, "third", 0.88),
            ]
        ]

        lines = engine.ocr(_make_image(h=100, w=100))

        assert len(lines) == 3
        assert [l.text for l in lines] == ["first", "second", "third"]

    def test_ocr_passes_image_to_engine(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.ocr.return_value = [[]]
        image = _make_image(300, 200)

        engine.ocr(image)

        paddle_mock.ocr.assert_called_once_with(image, cls=True)

    # --- _to_line() ---

    def test_to_line_uses_axis_aligned_bbox(self) -> None:
        """Non-rectangular polygon: bbox must wrap the extremes."""
        engine, _ = self._make_engine()
        # Skewed quadrilateral
        polygon = [[5.0, 10.0], [80.0, 8.0], [85.0, 30.0], [2.0, 32.0]]
        raw = _make_paddle_ocr_result(polygon, "skewed", 0.9)

        line = engine._to_line(raw, width=100, height=100)

        assert line.bounding_box.x0 == pytest.approx(0.02)  # min(5,80,85,2)/100
        assert line.bounding_box.x1 == pytest.approx(0.85)  # max(5,80,85,2)/100
        assert line.bounding_box.top == pytest.approx(0.08)  # min(10,8,30,32)/100
        assert line.bounding_box.bottom == pytest.approx(0.32)  # max(10,8,30,32)/100

    def test_to_line_extracts_text(self) -> None:
        engine, _ = self._make_engine()
        polygon = [[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]]
        raw = _make_paddle_ocr_result(polygon, "extracted text", 0.77)

        line = engine._to_line(raw, width=100, height=100)

        assert line.text == "extracted text"

    def test_to_line_ignores_confidence(self) -> None:
        """Confidence is present in input but not stored on Line."""
        engine, _ = self._make_engine()
        polygon = [[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]]

        line1 = engine._to_line(
            _make_paddle_ocr_result(polygon, "same text", 0.1),
            width=100,
            height=100,
        )
        line2 = engine._to_line(
            _make_paddle_ocr_result(polygon, "same text", 0.99),
            width=100,
            height=100,
        )

        assert line1 == line2

    def test_to_line_normalizes_by_region_dimensions(self) -> None:
        """Changing width/height changes bbox values proportionally."""
        engine, _ = self._make_engine()
        polygon = [[10.0, 20.0], [90.0, 20.0], [90.0, 40.0], [10.0, 40.0]]

        line_100 = engine._to_line(
            _make_paddle_ocr_result(polygon, "t", 0.9),
            width=100,
            height=100,
        )
        line_200 = engine._to_line(
            _make_paddle_ocr_result(polygon, "t", 0.9),
            width=200,
            height=200,
        )

        assert line_100.bounding_box.x0 == pytest.approx(line_200.bounding_box.x0 * 2)
        assert line_100.bounding_box.top == pytest.approx(line_200.bounding_box.top * 2)
