"""Tests for schemas.core module."""

from enum import Enum

import pytest
from pydantic import ValidationError

from doc_intelligence.schemas.core import (
    BaseCitation,
    BoundingBox,
    Document,
    ExtractionConfig,
)


class _DummyMode(Enum):
    A = "a"


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------
class TestBoundingBox:
    def test_construction(self):
        bbox = BoundingBox(x0=0.1, top=0.2, x1=0.9, bottom=0.8)
        assert bbox.x0 == 0.1
        assert bbox.top == 0.2
        assert bbox.x1 == 0.9
        assert bbox.bottom == 0.8

    def test_model_dump(self):
        bbox = BoundingBox(x0=1.0, top=2.0, x1=3.0, bottom=4.0)
        assert bbox.model_dump() == {
            "x0": 1.0,
            "top": 2.0,
            "x1": 3.0,
            "bottom": 4.0,
        }

    def test_equality(self):
        a = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=1.0)
        b = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=1.0)
        assert a == b

    def test_inequality(self):
        a = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=1.0)
        b = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=0.5)
        assert a != b

    def test_int_coerced_to_float(self):
        bbox = BoundingBox(x0=1, top=2, x1=3, bottom=4)
        assert isinstance(bbox.x0, float)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0="abc", top=0, x1=0, bottom=0)

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=0.0, top=0.0, x1=0.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# BaseCitation
# ---------------------------------------------------------------------------
class TestBaseCitation:
    def test_model_config_title(self):
        assert BaseCitation.model_config.get("title") == "Citation"

    def test_can_instantiate(self):
        citation = BaseCitation()
        assert isinstance(citation, BaseCitation)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class TestDocument:
    def test_minimal_construction(self):
        doc = Document(uri="test.pdf", extraction_mode=_DummyMode.A)
        assert doc.uri == "test.pdf"
        assert doc.extraction_mode is _DummyMode.A

    def test_defaults(self):
        doc = Document(uri="test.pdf", extraction_mode=_DummyMode.A)
        assert doc.content is None
        assert doc.include_citations is True
        assert doc.response is None
        assert doc.response_metadata is None

    def test_include_citations_override(self):
        doc = Document(
            uri="test.pdf",
            extraction_mode=_DummyMode.A,
            include_citations=False,
        )
        assert doc.include_citations is False

    def test_missing_uri_raises(self):
        with pytest.raises(ValidationError):
            Document(extraction_mode=_DummyMode.A)  # type: ignore[call-arg]

    def test_missing_extraction_mode_raises(self):
        with pytest.raises(ValidationError):
            Document(uri="test.pdf")  # type: ignore[call-arg]

    def test_response_metadata_accepts_dict(self):
        doc = Document(
            uri="test.pdf",
            extraction_mode=_DummyMode.A,
            response_metadata={"key": "value"},
        )
        assert doc.response_metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------
class TestExtractionConfig:
    def test_construction(self):
        cfg = ExtractionConfig(include_citations=True)
        assert cfg.include_citations is True

    def test_false_value(self):
        cfg = ExtractionConfig(include_citations=False)
        assert cfg.include_citations is False

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            ExtractionConfig()  # type: ignore[call-arg]
