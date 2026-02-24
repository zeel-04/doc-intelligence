"""Tests for extractor module."""

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.extractor import DigitalPDFExtractor
from doc_intelligence.schemas.pdf import PDF, PDFDocument
from doc_intelligence.types.pdf import PDFExtractionMode
from tests.conftest import FakeFormatter, FakeLLM


class SampleResponse(BaseModel):
    name: str = Field(..., description="person name")
    age: int = Field(..., description="person age")


@pytest.fixture
def extractor_with_llm():
    """Return a (DigitalPDFExtractor, FakeLLM) pair with a canned JSON response."""
    llm_response = json.dumps({"name": "Alice", "age": 30})
    llm = FakeLLM(text_response=llm_response)
    return DigitalPDFExtractor(llm=llm), llm


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestDigitalPDFExtractorInit:
    def test_stores_prompts(self):
        llm = FakeLLM()
        extractor = DigitalPDFExtractor(llm=llm)
        assert "expert" in extractor.system_prompt.lower()
        assert "{content_text}" in extractor.user_prompt
        assert "{schema}" in extractor.user_prompt


# ---------------------------------------------------------------------------
# extract — single-pass, no citations
# ---------------------------------------------------------------------------
class TestExtractSinglePassNoCitations:
    def test_returns_extracted_data_and_metadata(
        self, extractor_with_llm, sample_pdf: PDF
    ):
        extractor, _ = extractor_with_llm
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=False,
        )
        result = extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )
        assert isinstance(result["extracted_data"], SampleResponse)
        assert result["extracted_data"].name == "Alice"
        assert result["extracted_data"].age == 30
        assert result["metadata"] is None

    def test_llm_receives_system_and_user_prompt(
        self, extractor_with_llm, sample_pdf: PDF
    ):
        extractor, llm = extractor_with_llm
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=False,
        )
        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(output="the document text"),
            response_format=SampleResponse,
        )
        assert "expert" in llm.last_call_kwargs["system_prompt"].lower()
        assert "the document text" in llm.last_call_kwargs["user_prompt"]

    def test_llm_config_forwarded(self, extractor_with_llm, sample_pdf: PDF):
        extractor, llm = extractor_with_llm
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=False,
        )
        extractor.extract(
            document=doc,
            llm_config={"model": "gpt-4o", "temperature": 0.2},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )
        assert llm.last_call_kwargs["model"] == "gpt-4o"
        assert llm.last_call_kwargs["temperature"] == 0.2

    def test_schema_included_in_prompt(self, extractor_with_llm, sample_pdf: PDF):
        extractor, llm = extractor_with_llm
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=False,
        )
        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )
        prompt = llm.last_call_kwargs["user_prompt"]
        assert "OUTPUT SCHEMA:" in prompt
        assert "name" in prompt


# ---------------------------------------------------------------------------
# extract — single-pass, with citations
# ---------------------------------------------------------------------------
class TestExtractSinglePassWithCitations:
    def test_enriches_and_strips_citations(self, sample_pdf: PDF):
        citation_response = json.dumps(
            {
                "name": {
                    "value": "Alice",
                    "citations": [{"page": 0, "lines": [0]}],
                },
                "age": {
                    "value": 30,
                    "citations": [{"page": 0, "lines": [1]}],
                },
            }
        )
        llm = FakeLLM(text_response=citation_response)
        extractor = DigitalPDFExtractor(llm=llm)

        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=True,
        )
        result = extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert result["extracted_data"].name == "Alice"
        assert result["extracted_data"].age == 30
        assert result["metadata"] is not None
        assert "bboxes" in str(result["metadata"])

    def test_metadata_has_bboxes_not_lines(self, sample_pdf: PDF):
        citation_response = json.dumps(
            {
                "name": {
                    "value": "Bob",
                    "citations": [{"page": 0, "lines": [0]}],
                },
                "age": {
                    "value": 25,
                    "citations": [{"page": 0, "lines": [1]}],
                },
            }
        )
        llm = FakeLLM(text_response=citation_response)
        extractor = DigitalPDFExtractor(llm=llm)

        doc = PDFDocument(uri="test.pdf", content=sample_pdf, include_citations=True)
        result = extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        metadata = result["metadata"]
        name_cit = metadata["name"]["citations"][0]
        assert "bboxes" in name_cit
        assert "lines" not in name_cit


# ---------------------------------------------------------------------------
# extract — multi-pass
# ---------------------------------------------------------------------------
class TestExtractMultiPass:
    def test_multi_pass_raises(self, sample_pdf: PDF):
        llm = FakeLLM()
        extractor = DigitalPDFExtractor(llm=llm)
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        with pytest.raises(NotImplementedError, match="Multi-pass"):
            extractor.extract(
                document=doc,
                llm_config={},
                extraction_config={},
                formatter=FakeFormatter(),
                response_format=SampleResponse,
            )
