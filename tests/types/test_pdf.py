"""Tests for types.pdf module."""

import pytest

from doc_intelligence.types.pdf import PDFExtractionMode


# ---------------------------------------------------------------------------
# PDFExtractionMode
# ---------------------------------------------------------------------------
class TestPDFExtractionMode:
    def test_single_pass_value(self):
        assert PDFExtractionMode.SINGLE_PASS.value == "single_pass"

    def test_multi_pass_value(self):
        assert PDFExtractionMode.MULTI_PASS.value == "multi_pass"

    def test_member_count(self):
        assert len(PDFExtractionMode) == 2

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("single_pass", PDFExtractionMode.SINGLE_PASS),
            ("multi_pass", PDFExtractionMode.MULTI_PASS),
        ],
    )
    def test_from_value(self, value, expected):
        assert PDFExtractionMode(value) is expected

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            PDFExtractionMode("invalid")

    def test_name_attribute(self):
        assert PDFExtractionMode.SINGLE_PASS.name == "SINGLE_PASS"
        assert PDFExtractionMode.MULTI_PASS.name == "MULTI_PASS"
