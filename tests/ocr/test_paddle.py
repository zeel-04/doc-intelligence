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
    """Return a mock ``paddleocr`` module with LayoutDetection and PaddleOCR stubs."""
    mock = MagicMock()
    mock.LayoutDetection.return_value = MagicMock()
    mock.PaddleOCR.return_value = MagicMock()
    return mock


def _make_layout_result(
    coordinate: list[float],
    label: str,
    score: float,
    cls_id: int = 0,
) -> dict[str, Any]:
    """Build a LayoutDetection box result dict (v3 format)."""
    return {
        "cls_id": cls_id,
        "label": label,
        "score": score,
        "coordinate": coordinate,
    }


def _make_layout_predict_result(boxes: list[dict[str, Any]]) -> MagicMock:
    """Build a mock result object from LayoutDetection.predict()."""
    res = MagicMock()
    res.json = {"res": {"input_path": None, "page_index": None, "boxes": boxes}}
    return res


def _make_ocr_predict_result(
    rec_texts: list[str],
    rec_boxes: list[list[int]],
) -> MagicMock:
    """Build a mock result object from PaddleOCR.predict()."""
    res = MagicMock()
    res.json = {
        "res": {
            "input_path": None,
            "page_index": None,
            "rec_texts": rec_texts,
            "rec_boxes": rec_boxes,
        }
    }
    return res


# ---------------------------------------------------------------------------
# Module import (deferred-import contract)
# ---------------------------------------------------------------------------
class TestDeferredImport:
    def test_module_importable_without_paddleocr(self) -> None:
        """Importing the paddle module must not require PaddleOCR to be installed."""
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
        import importlib

        import doc_intelligence.ocr.paddle as paddle_mod

        # Save and block all paddleocr-related modules
        saved: dict[str, Any] = {}
        to_remove = [
            k for k in sys.modules if k == "paddleocr" or k.startswith("paddleocr.")
        ]
        for k in to_remove:
            saved[k] = sys.modules.pop(k)

        # Insert a sentinel that makes `import paddleocr` raise
        sys.modules["paddleocr"] = None  # type: ignore[assignment]
        try:
            importlib.reload(paddle_mod)
            with pytest.raises((ImportError, ModuleNotFoundError)):
                paddle_mod.PaddleLayoutDetector()
        finally:
            sys.modules.pop("paddleocr", None)
            sys.modules.update(saved)
            importlib.reload(paddle_mod)


# ---------------------------------------------------------------------------
# PaddleLayoutDetector
# ---------------------------------------------------------------------------
class TestPaddleLayoutDetector:
    # --- fixtures / helpers ---

    @staticmethod
    def _make_detector() -> tuple[Any, MagicMock]:
        """Return (detector_instance, layout_detection_mock_instance)."""
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        mock_module = _make_paddle_module()
        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            detector = PaddleLayoutDetector()
        return detector, mock_module.LayoutDetection.return_value

    # --- construction ---

    def test_layout_detection_instantiated_with_correct_args(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleLayoutDetector()

        mock_module.LayoutDetection.assert_called_once_with(
            model_name="PP-DocLayout_plus-L",
        )

    def test_custom_model_name_forwarded(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleLayoutDetector(model_name="PP-DocLayout-S")

        mock_module.LayoutDetection.assert_called_once_with(
            model_name="PP-DocLayout-S",
        )

    def test_extra_kwargs_forwarded_to_layout_detection(self) -> None:
        mock_module = _make_paddle_module()
        from doc_intelligence.ocr.paddle import PaddleLayoutDetector

        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            PaddleLayoutDetector(device="gpu:0")

        _, kwargs = mock_module.LayoutDetection.call_args
        assert kwargs["device"] == "gpu:0"

    # --- detect() ---

    def test_detect_returns_empty_list_for_no_results(self) -> None:
        detector, engine_mock = self._make_detector()
        engine_mock.predict.return_value = [_make_layout_predict_result([])]
        image = _make_image()

        result = detector.detect(image)

        assert result == []

    def test_detect_converts_single_region(self) -> None:
        from doc_intelligence.ocr.base import LayoutRegion

        detector, engine_mock = self._make_detector()
        engine_mock.predict.return_value = [
            _make_layout_predict_result(
                [_make_layout_result([10.0, 20.0, 110.0, 80.0], "text", 0.95)]
            )
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
        engine_mock.predict.return_value = [
            _make_layout_predict_result(
                [
                    _make_layout_result([0.0, 0.0, 50.0, 30.0], "doc_title", 0.99),
                    _make_layout_result([0.0, 40.0, 200.0, 180.0], "text", 0.87),
                    _make_layout_result([0.0, 190.0, 200.0, 300.0], "table", 0.75),
                ]
            )
        ]

        regions = detector.detect(_make_image())

        assert len(regions) == 3
        assert regions[0].region_type == "doc_title"
        assert regions[1].region_type == "text"
        assert regions[2].region_type == "table"

    def test_detect_passes_image_to_engine(self) -> None:
        detector, engine_mock = self._make_detector()
        engine_mock.predict.return_value = [_make_layout_predict_result([])]
        image = _make_image(300, 200)

        detector.detect(image)

        engine_mock.predict.assert_called_once_with(image, batch_size=1)

    # --- _to_layout_region() ---

    def test_to_layout_region_float_conversion(self) -> None:
        detector, _ = self._make_detector()
        raw = _make_layout_result([10, 20, 110, 80], "figure", 0.6)

        region = detector._to_layout_region(raw)

        assert isinstance(region.bounding_box.x0, float)
        assert isinstance(region.bounding_box.top, float)
        assert isinstance(region.bounding_box.x1, float)
        assert isinstance(region.bounding_box.bottom, float)

    def test_to_layout_region_maps_all_fields(self) -> None:
        from doc_intelligence.ocr.base import LayoutRegion

        detector, _ = self._make_detector()
        raw = _make_layout_result([5.5, 10.0, 200.0, 150.5], "header", 0.88)

        region = detector._to_layout_region(raw)

        assert isinstance(region, LayoutRegion)
        assert region.bounding_box == BoundingBox(
            x0=5.5, top=10.0, x1=200.0, bottom=150.5
        )
        assert region.region_type == "header"
        assert region.confidence == pytest.approx(0.88)

    @pytest.mark.parametrize(
        "region_type", ["text", "table", "figure", "doc_title", "list"]
    )
    def test_to_layout_region_accepts_any_region_type(self, region_type: str) -> None:
        detector, _ = self._make_detector()
        raw = _make_layout_result([0.0, 0.0, 100.0, 50.0], region_type, 0.9)

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

        mock_module.PaddleOCR.assert_called_once_with(lang="en")

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
            PaddleOCREngine(use_doc_orientation_classify=True)

        _, kwargs = mock_module.PaddleOCR.call_args
        assert kwargs["use_doc_orientation_classify"] is True

    # --- ocr() — empty results ---

    def test_ocr_returns_empty_list_when_no_texts(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.predict.return_value = [_make_ocr_predict_result([], [])]
        assert engine.ocr(_make_image()) == []

    # --- ocr() — single line ---

    def test_ocr_single_line_text(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.predict.return_value = [
            _make_ocr_predict_result(
                rec_texts=["hello world"],
                rec_boxes=[[10, 20, 90, 40]],
            )
        ]

        lines = engine.ocr(_make_image(h=100, w=100))

        assert len(lines) == 1
        assert lines[0].text == "hello world"

    def test_ocr_single_line_bbox_is_normalized(self) -> None:
        engine, paddle_mock = self._make_engine()
        # image: 200h x 100w; box occupies 10–90 in x, 20–40 in y
        paddle_mock.predict.return_value = [
            _make_ocr_predict_result(
                rec_texts=["text"],
                rec_boxes=[[10, 20, 90, 40]],
            )
        ]

        lines = engine.ocr(_make_image(h=200, w=100))

        bbox = lines[0].bounding_box
        assert bbox.x0 == pytest.approx(0.10)  # 10/100
        assert bbox.x1 == pytest.approx(0.90)  # 90/100
        assert bbox.top == pytest.approx(0.10)  # 20/200
        assert bbox.bottom == pytest.approx(0.20)  # 40/200

    # --- ocr() — multiple lines ---

    def test_ocr_multiple_lines(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.predict.return_value = [
            _make_ocr_predict_result(
                rec_texts=["first", "second", "third"],
                rec_boxes=[
                    [0, 0, 100, 10],
                    [0, 20, 100, 30],
                    [0, 40, 100, 50],
                ],
            )
        ]

        lines = engine.ocr(_make_image(h=100, w=100))

        assert len(lines) == 3
        assert [line.text for line in lines] == ["first", "second", "third"]

    def test_ocr_passes_image_to_engine(self) -> None:
        engine, paddle_mock = self._make_engine()
        paddle_mock.predict.return_value = [_make_ocr_predict_result([], [])]
        image = _make_image(300, 200)

        engine.ocr(image)

        paddle_mock.predict.assert_called_once_with(image)

    # --- _to_line() ---

    def test_to_line_normalizes_by_region_dimensions(self) -> None:
        """Changing width/height changes bbox values proportionally."""
        engine, _ = self._make_engine()

        line_100 = engine._to_line("t", [10, 20, 90, 40], width=100, height=100)
        line_200 = engine._to_line("t", [10, 20, 90, 40], width=200, height=200)

        assert line_100.bounding_box.x0 == pytest.approx(line_200.bounding_box.x0 * 2)
        assert line_100.bounding_box.top == pytest.approx(line_200.bounding_box.top * 2)

    def test_to_line_extracts_text(self) -> None:
        engine, _ = self._make_engine()

        line = engine._to_line("extracted text", [0, 0, 50, 10], width=100, height=100)

        assert line.text == "extracted text"

    def test_to_line_float_conversion(self) -> None:
        """Box coordinates (ints from PaddleOCR) are converted to float."""
        engine, _ = self._make_engine()

        line = engine._to_line("t", [10, 20, 90, 40], width=100, height=100)

        assert isinstance(line.bounding_box.x0, float)
        assert isinstance(line.bounding_box.top, float)
        assert isinstance(line.bounding_box.x1, float)
        assert isinstance(line.bounding_box.bottom, float)
