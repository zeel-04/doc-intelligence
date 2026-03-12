"""Tests for extractor module."""

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.pdf.extractor import DigitalPDFExtractor
from doc_intelligence.pdf.schemas import PDF, PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode
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

        assert result is not None
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

        assert result is not None
        metadata = result["metadata"]
        name_cit = metadata["name"]["citations"][0]
        assert "bboxes" in name_cit
        assert "lines" not in name_cit


# ---------------------------------------------------------------------------
# extract — multi-pass
# ---------------------------------------------------------------------------
class TestExtractMultiPass:
    """Three-pass extraction: raw → page grounding → line grounding."""

    def _make_doc(self, sample_pdf: PDF, citations: bool = True) -> PDFDocument:
        return PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=citations,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )

    def test_multi_pass_no_citations_returns_pass1_result(self, sample_pdf: PDF):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        llm = FakeLLM(responses=[pass1_json])
        extractor = DigitalPDFExtractor(llm=llm)

        result = extractor.extract(
            document=self._make_doc(sample_pdf, citations=False),
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert isinstance(result["extracted_data"], SampleResponse)
        assert result["extracted_data"].name == "Alice"
        assert result["extracted_data"].age == 30
        assert result["metadata"] is None
        assert llm._call_index == 1  # only Pass 1 called

    def test_multi_pass_no_citations_stores_pass1_on_document(self, sample_pdf: PDF):
        pass1_json = json.dumps({"name": "Bob", "age": 25})
        llm = FakeLLM(responses=[pass1_json])
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_doc(sample_pdf, citations=False)

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert isinstance(doc.pass1_result, SampleResponse)
        assert doc.pass1_result.name == "Bob"

    def test_multi_pass_with_citations_calls_llm_three_times(self, sample_pdf: PDF):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [1]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)

        extractor.extract(
            document=self._make_doc(sample_pdf, citations=True),
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert llm._call_index == 3

    def test_multi_pass_with_citations_returns_correct_extracted_data(
        self, sample_pdf: PDF
    ):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [1]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)

        result = extractor.extract(
            document=self._make_doc(sample_pdf, citations=True),
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert isinstance(result["extracted_data"], SampleResponse)
        assert result["extracted_data"].name == "Alice"
        assert result["extracted_data"].age == 30

    def test_multi_pass_with_citations_returns_metadata_with_bboxes(
        self, sample_pdf: PDF
    ):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [1]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)

        result = extractor.extract(
            document=self._make_doc(sample_pdf, citations=True),
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert result["metadata"] is not None
        assert "bboxes" in str(result["metadata"])

    def test_multi_pass_stores_pass2_page_map_on_document(self, sample_pdf: PDF):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [1]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 1, "lines": [0]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_doc(sample_pdf, citations=True)

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        assert doc.pass2_page_map == {"name": [0], "age": [1]}

    def test_multi_pass_pass1_prompt_has_no_citation_schema(self, sample_pdf: PDF):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [1]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)

        extractor.extract(
            document=self._make_doc(sample_pdf, citations=True),
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        # Pass 1 prompt must NOT contain citation wrapper keys
        pass1_prompt = llm.all_calls[0]["user_prompt"]
        assert "citations" not in pass1_prompt.lower()

    def test_multi_pass_pass3_prompt_contains_pass1_answer(self, sample_pdf: PDF):
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [1]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)

        extractor.extract(
            document=self._make_doc(sample_pdf, citations=True),
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

        pass3_prompt = llm.all_calls[2]["user_prompt"]
        assert "Alice" in pass3_prompt

    def test_multi_pass_invalid_mode_raises(self, sample_pdf: PDF):
        """Unsupported extraction_mode still raises ValueError."""
        from unittest.mock import MagicMock

        llm = FakeLLM()
        extractor = DigitalPDFExtractor(llm=llm)
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        doc.extraction_mode = MagicMock()  # not a real PDFExtractionMode

        with pytest.raises(ValueError, match="Unsupported extraction mode"):
            extractor.extract(
                document=doc,
                llm_config={},
                extraction_config={},
                formatter=FakeFormatter(),
                response_format=SampleResponse,
            )


# ---------------------------------------------------------------------------
# single-pass vs multi-pass alignment
# ---------------------------------------------------------------------------
class TestSinglePassVsMultiPassAlignment:
    """Both modes must produce identical extracted_data and equivalent metadata.

    The invariant: given the same underlying LLM answers (same field values,
    same citation pages/lines), single-pass and multi-pass must agree on
    every field value and every resolved bbox.
    """

    # Shared citation response used as the LLM answer in both modes.
    _CITATION_JSON = json.dumps(
        {
            "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
            "age": {"value": 30, "citations": [{"page": 0, "lines": [1]}]},
        }
    )
    _PLAIN_JSON = json.dumps({"name": "Alice", "age": 30})
    _PAGE_MAP_JSON = json.dumps({"name": [0], "age": [0]})

    def _run_single_pass(self, sample_pdf: PDF) -> dict[str, Any]:
        llm = FakeLLM(text_response=self._CITATION_JSON)
        extractor = DigitalPDFExtractor(llm=llm)
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=True,
            extraction_mode=PDFExtractionMode.SINGLE_PASS,
        )
        return extractor.extract(  # type: ignore[return-value]
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

    def _run_multi_pass(self, sample_pdf: PDF) -> dict[str, Any]:
        llm = FakeLLM(
            responses=[self._PLAIN_JSON, self._PAGE_MAP_JSON, self._CITATION_JSON]
        )
        extractor = DigitalPDFExtractor(llm=llm)
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=True,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        return extractor.extract(  # type: ignore[return-value]
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

    def test_extracted_data_matches(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp["extracted_data"] == mp["extracted_data"]

    def test_extracted_data_name_matches(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp["extracted_data"].name == mp["extracted_data"].name == "Alice"

    def test_extracted_data_age_matches(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp["extracted_data"].age == mp["extracted_data"].age == 30

    def test_metadata_fields_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert set(sp["metadata"].keys()) == set(mp["metadata"].keys())

    def test_metadata_citation_values_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        for field in ("name", "age"):
            assert sp["metadata"][field]["value"] == mp["metadata"][field]["value"], (
                f"metadata value mismatch for field '{field}'"
            )

    def test_metadata_bboxes_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        for field in ("name", "age"):
            sp_bboxes = sp["metadata"][field]["citations"][0]["bboxes"]
            mp_bboxes = mp["metadata"][field]["citations"][0]["bboxes"]
            assert sp_bboxes == mp_bboxes, (
                f"bbox mismatch for field '{field}': {sp_bboxes} vs {mp_bboxes}"
            )

    def test_metadata_citation_pages_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        for field in ("name", "age"):
            sp_page = sp["metadata"][field]["citations"][0]["page"]
            mp_page = mp["metadata"][field]["citations"][0]["page"]
            assert sp_page == mp_page, (
                f"citation page mismatch for field '{field}': {sp_page} vs {mp_page}"
            )
