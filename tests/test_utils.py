"""Tests for utils module."""

from typing import Any

import pytest

from doc_intelligence.schemas.core import BoundingBox
from doc_intelligence.utils import (
    normalize_bounding_box,
    strip_citations,
)


# ---------------------------------------------------------------------------
# normalize_bounding_box
# ---------------------------------------------------------------------------
class TestNormalizeBoundingBox:
    def test_basic_normalization(self):
        bbox = BoundingBox(x0=100.0, top=200.0, x1=300.0, bottom=400.0)
        result = normalize_bounding_box(bbox, page_width=1000, page_height=1000)
        assert result.x0 == pytest.approx(0.1)
        assert result.top == pytest.approx(0.2)
        assert result.x1 == pytest.approx(0.3)
        assert result.bottom == pytest.approx(0.4)

    def test_identity_when_dimensions_are_one(self):
        bbox = BoundingBox(x0=0.5, top=0.6, x1=0.7, bottom=0.8)
        result = normalize_bounding_box(bbox, page_width=1, page_height=1)
        assert result == bbox

    def test_float_dimensions(self):
        bbox = BoundingBox(x0=50.0, top=100.0, x1=250.0, bottom=125.0)
        result = normalize_bounding_box(bbox, page_width=500.0, page_height=1000.0)
        assert result.x0 == pytest.approx(0.1)
        assert result.top == pytest.approx(0.1)
        assert result.x1 == pytest.approx(0.5)
        assert result.bottom == pytest.approx(0.125)

    @pytest.mark.parametrize(
        "bbox, width, height, expected_x0",
        [
            (BoundingBox(x0=100, top=0, x1=0, bottom=0), 1000, 1, 0.1),
            (BoundingBox(x0=0, top=0, x1=0, bottom=0), 500, 800, 0.0),
            (BoundingBox(x0=250, top=400, x1=250, bottom=400), 500, 800, 0.5),
        ],
    )
    def test_various_inputs(self, bbox, width, height, expected_x0):
        result = normalize_bounding_box(bbox, width, height)
        assert result.x0 == pytest.approx(expected_x0)

    @pytest.mark.parametrize(
        "width, height", [(0, 100), (100, 0), (0, 0), (-1, 100), (100, -5)]
    )
    def test_zero_or_negative_dimensions_raise(self, width, height):
        bbox = BoundingBox(x0=10, top=20, x1=30, bottom=40)
        with pytest.raises(ValueError, match="Page dimensions must be positive"):
            normalize_bounding_box(bbox, width, height)


# ---------------------------------------------------------------------------
# strip_citations
# ---------------------------------------------------------------------------
class TestStripCitations:
    def test_simple_unwrap(self, citation_response_simple: dict[str, Any]):
        result = strip_citations(citation_response_simple)
        assert result == {"name": "Alice"}

    def test_nested_unwrap(self, citation_response_nested: dict[str, Any]):
        result = strip_citations(citation_response_nested)
        assert result == {"person": {"name": "Bob", "age": 30}}

    def test_list_of_wrapped(self, citation_response_with_list: dict[str, Any]):
        result = strip_citations(citation_response_with_list)
        assert result == {"ids": [101, 205]}

    def test_non_citation_dict_unchanged(self):
        response = {"name": "Alice", "age": 30}
        result = strip_citations(response)
        assert result == {"name": "Alice", "age": 30}

    def test_deeply_nested(self):
        response = {
            "level1": {
                "level2": {
                    "field": {"value": "deep", "citations": [{"page": 0}]},
                },
            },
        }
        result = strip_citations(response)
        assert result == {"level1": {"level2": {"field": "deep"}}}

    def test_dict_with_extra_keys_not_stripped(self):
        response = {"name": {"value": "Alice", "citations": [], "extra": True}}
        result = strip_citations(response)
        assert result == response

    def test_empty_dict(self):
        assert strip_citations({}) == {}

    def test_scalar_values_in_list(self):
        response = {"tags": ["a", "b", "c"]}
        result = strip_citations(response)
        assert result == {"tags": ["a", "b", "c"]}

    def test_none_value_preserved(self):
        response = {"field": {"value": None, "citations": []}}
        result = strip_citations(response)
        assert result == {"field": None}

    def test_nested_value_preserved(self):
        response = {
            "field": {
                "value": {"nested_key": 42},
                "citations": [{"page": 0}],
            },
        }
        result = strip_citations(response)
        assert result == {"field": {"nested_key": 42}}
