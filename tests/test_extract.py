"""Tests for the top-level extract() convenience function."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.extract import extract
from doc_intelligence.schemas.core import ExtractionResult


class SampleModel(BaseModel):
    name: str = Field(..., description="person name")
    age: int = Field(..., description="person age")


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------
class TestExtract:
    def test_returns_extraction_result(self):
        expected = ExtractionResult(
            data=SampleModel(name="Alice", age=30), metadata=None
        )
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = expected
            result = extract("test.pdf", SampleModel, provider="openai")
        assert result.data.name == "Alice"
        assert result.data.age == 30
        assert result.metadata is None

    def test_creates_processor_with_provider_and_model(self):
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = ExtractionResult(data=None)
            extract(
                "test.pdf",
                SampleModel,
                provider="anthropic",
                model="claude-sonnet-4-20250514",
            )
        MockProc.assert_called_once_with(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            document_type="digital",
        )

    def test_forwards_extraction_options(self):
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = ExtractionResult(data=None)
            extract(
                "test.pdf",
                SampleModel,
                provider="openai",
                include_citations=False,
                extraction_mode="multi_pass",
                page_numbers=[0, 2],
                llm_config={"temperature": 0.5},
            )
        instance.extract.assert_called_once_with(
            uri="test.pdf",
            response_format=SampleModel,
            include_citations=False,
            extraction_mode="multi_pass",
            page_numbers=[0, 2],
            llm_config={"temperature": 0.5},
        )

    def test_defaults_to_openai_provider(self):
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = ExtractionResult(data=None)
            extract("test.pdf", SampleModel)
        MockProc.assert_called_once_with(
            provider="openai", model=None, document_type="digital"
        )

    def test_forwards_llm_kwargs(self):
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = ExtractionResult(data=None)
            extract(
                "test.pdf",
                SampleModel,
                provider="ollama",
                host="http://localhost:11434",
            )
        MockProc.assert_called_once_with(
            provider="ollama",
            model=None,
            document_type="digital",
            host="http://localhost:11434",
        )

    def test_document_type_scanned_forwarded_to_processor(self):
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = ExtractionResult(data=None)
            extract("test.pdf", SampleModel, document_type="scanned")
        MockProc.assert_called_once_with(
            provider="openai", model=None, document_type="scanned"
        )

    def test_document_type_defaults_to_digital(self):
        with patch("doc_intelligence.extract.PDFProcessor") as MockProc:
            instance = MockProc.return_value
            instance.extract.return_value = ExtractionResult(data=None)
            extract("test.pdf", SampleModel)
        _, kwargs = MockProc.call_args
        assert kwargs["document_type"] == "digital"
