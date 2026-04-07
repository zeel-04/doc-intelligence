"""Tests for schemas.core module."""

import pytest
from pydantic import BaseModel, ValidationError

from doc_intelligence.schemas.core import (
    BaseCitation,
    BoundingBox,
    Document,
    ExtractionRequest,
    ExtractionResult,
)


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
        doc = Document(uri="test.pdf")
        assert doc.uri == "test.pdf"

    def test_defaults(self):
        doc = Document(uri="test.pdf")
        assert doc.content is None

    def test_missing_uri_raises(self):
        with pytest.raises(ValidationError):
            Document()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------
class TestExtractionResult:
    def test_construction_with_data_only(self):
        result = ExtractionResult(data={"name": "Alice"})
        assert result.data == {"name": "Alice"}
        assert result.metadata is None

    def test_construction_with_data_and_metadata(self):
        result = ExtractionResult(
            data={"name": "Alice"},
            metadata={"name": {"citations": [{"page": 0}]}},
        )
        assert result.data == {"name": "Alice"}
        assert result.metadata is not None
        assert "name" in result.metadata

    def test_data_accepts_pydantic_model(self):
        class Sample(BaseModel):
            name: str

        model = Sample(name="Bob")
        result = ExtractionResult(data=model)
        assert result.data.name == "Bob"

    def test_data_accepts_none(self):
        result = ExtractionResult(data=None)
        assert result.data is None

    def test_metadata_defaults_to_none(self):
        result = ExtractionResult(data="anything")
        assert result.metadata is None

    def test_model_dump(self):
        result = ExtractionResult(data={"key": "val"}, metadata={"field": {"page": 0}})
        dumped = result.model_dump()
        assert dumped["data"] == {"key": "val"}
        assert dumped["metadata"] == {"field": {"page": 0}}


# ---------------------------------------------------------------------------
# ExtractionRequest
# ---------------------------------------------------------------------------
class TestExtractionRequest:
    def test_construction(self):
        req = ExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            include_citations=True,
        )
        assert req.uri == "test.pdf"
        assert req.response_format is BaseModel
        assert req.include_citations is True

    def test_defaults(self):
        req = ExtractionRequest(uri="test.pdf", response_format=BaseModel)
        assert req.include_citations is True
        assert req.llm_config is None

    def test_include_citations_false(self):
        req = ExtractionRequest(
            uri="test.pdf", response_format=BaseModel, include_citations=False
        )
        assert req.include_citations is False

    def test_llm_config(self):
        req = ExtractionRequest(
            uri="test.pdf",
            response_format=BaseModel,
            llm_config={"temperature": 0.5},
        )
        assert req.llm_config == {"temperature": 0.5}

    def test_missing_uri_raises(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(response_format=BaseModel)  # type: ignore[call-arg]

    def test_missing_response_format_raises(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(uri="test.pdf")  # type: ignore[call-arg]
