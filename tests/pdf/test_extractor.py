"""Tests for extractor module."""

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.pdf.extractor import DigitalPDFExtractor
from doc_intelligence.pdf.formatter import DigitalPDFFormatter
from doc_intelligence.pdf.schemas import PDF, Line, Page, PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import BoundingBox, ExtractionResult
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
        assert isinstance(result.data, SampleResponse)
        assert result.data.name == "Alice"
        assert result.data.age == 30
        assert result.metadata is None

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
        assert result.data.name == "Alice"
        assert result.data.age == 30
        assert result.metadata is not None
        assert "bboxes" in str(result.metadata)

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
        assert result.metadata is not None
        metadata = result.metadata
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

        assert isinstance(result.data, SampleResponse)
        assert result.data.name == "Alice"
        assert result.data.age == 30
        assert result.metadata is None
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

        assert isinstance(result.data, SampleResponse)
        assert result.data.name == "Alice"
        assert result.data.age == 30

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

        assert result.metadata is not None
        assert "bboxes" in str(result.metadata)

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
# page_numbers filtering
# ---------------------------------------------------------------------------
class TestExtractPageNumbers:
    """page_numbers on PDFDocument is forwarded to the formatter in all modes."""

    _BBOX = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=0.1)

    def _make_multipage_doc(
        self,
        page_numbers: list[int] | None = None,
        include_citations: bool = False,
        mode: PDFExtractionMode = PDFExtractionMode.SINGLE_PASS,
    ) -> PDFDocument:
        """Return a 3-page PDFDocument with distinct per-page content."""
        pages = [
            Page(
                lines=[Line(text=f"page{i} content", bounding_box=self._BBOX)],
                width=100,
                height=100,
            )
            for i in range(3)
        ]
        return PDFDocument(
            uri="test.pdf",
            content=PDF(pages=pages),
            include_citations=include_citations,
            extraction_mode=mode,
            page_numbers=page_numbers,
        )

    # -- single-pass --

    def test_single_pass_only_sends_requested_pages_to_llm(self):
        """page_numbers=[1] → only page 1 content in the LLM prompt."""
        llm = FakeLLM(text_response=json.dumps({"name": "Alice", "age": 30}))
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_multipage_doc(page_numbers=[1])

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        prompt = llm.last_call_kwargs["user_prompt"]
        assert "page1 content" in prompt
        assert "page0 content" not in prompt
        assert "page2 content" not in prompt

    def test_single_pass_none_page_numbers_sends_all_pages(self):
        """page_numbers=None (default) → all pages reach the LLM."""
        llm = FakeLLM(text_response=json.dumps({"name": "Alice", "age": 30}))
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_multipage_doc(page_numbers=None)

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        prompt = llm.last_call_kwargs["user_prompt"]
        assert "page0 content" in prompt
        assert "page1 content" in prompt
        assert "page2 content" in prompt

    def test_single_pass_multiple_page_numbers_filters_correctly(self):
        """page_numbers=[0, 2] → pages 0 and 2 present, page 1 absent."""
        llm = FakeLLM(text_response=json.dumps({"name": "Alice", "age": 30}))
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_multipage_doc(page_numbers=[0, 2])

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        prompt = llm.last_call_kwargs["user_prompt"]
        assert "page0 content" in prompt
        assert "page2 content" in prompt
        assert "page1 content" not in prompt

    # -- multi-pass pass 1 --

    def test_multi_pass_pass1_respects_page_numbers(self):
        """Pass 1 only sends requested pages to the LLM."""
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [0]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_multipage_doc(
            page_numbers=[0],
            include_citations=True,
            mode=PDFExtractionMode.MULTI_PASS,
        )

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        pass1_prompt = llm.all_calls[0]["user_prompt"]
        assert "page0 content" in pass1_prompt
        assert "page1 content" not in pass1_prompt
        assert "page2 content" not in pass1_prompt

    # -- multi-pass pass 2 --

    def test_multi_pass_pass2_respects_page_numbers(self):
        """Pass 2 only scans requested pages when grounding."""
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        pass2_json = json.dumps({"name": [0], "age": [0]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [0]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_multipage_doc(
            page_numbers=[0],
            include_citations=True,
            mode=PDFExtractionMode.MULTI_PASS,
        )

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        pass2_prompt = llm.all_calls[1]["user_prompt"]
        assert "page0 content" in pass2_prompt
        assert "page1 content" not in pass2_prompt
        assert "page2 content" not in pass2_prompt

    # -- multi-pass pass 3 intersection --

    def test_multi_pass_pass3_intersects_user_filter_with_pass2_result(self):
        """Pass 3 uses the intersection of user page_numbers and Pass 2's page map."""
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        # Pass 2 says data is on pages 0 and 1
        pass2_json = json.dumps({"name": [0], "age": [1]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 0, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 0, "lines": [0]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)
        # User only allows page 0 — page 1 from Pass 2 should be filtered out
        doc = self._make_multipage_doc(
            page_numbers=[0],
            include_citations=True,
            mode=PDFExtractionMode.MULTI_PASS,
        )

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        pass3_prompt = llm.all_calls[2]["user_prompt"]
        assert "page0 content" in pass3_prompt
        assert "page1 content" not in pass3_prompt

    def test_multi_pass_pass3_falls_back_to_pass2_when_intersection_empty(self):
        """If intersection of user filter and Pass 2 is empty, use Pass 2 pages."""
        pass1_json = json.dumps({"name": "Alice", "age": 30})
        # Pass 2 says data is on page 2 only — disjoint from user's [0]
        pass2_json = json.dumps({"name": [2], "age": [2]})
        pass3_json = json.dumps(
            {
                "name": {"value": "Alice", "citations": [{"page": 2, "lines": [0]}]},
                "age": {"value": 30, "citations": [{"page": 2, "lines": [0]}]},
            }
        )
        llm = FakeLLM(responses=[pass1_json, pass2_json, pass3_json])
        extractor = DigitalPDFExtractor(llm=llm)
        doc = self._make_multipage_doc(
            page_numbers=[0],
            include_citations=True,
            mode=PDFExtractionMode.MULTI_PASS,
        )

        extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=DigitalPDFFormatter(),
            response_format=SampleResponse,
        )

        # Falls back to Pass 2's page 2 since intersection was empty
        pass3_prompt = llm.all_calls[2]["user_prompt"]
        assert "page2 content" in pass3_prompt


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

    def _run_single_pass(self, sample_pdf: PDF) -> ExtractionResult:
        llm = FakeLLM(text_response=self._CITATION_JSON)
        extractor = DigitalPDFExtractor(llm=llm)
        doc = PDFDocument(
            uri="test.pdf",
            content=sample_pdf,
            include_citations=True,
            extraction_mode=PDFExtractionMode.SINGLE_PASS,
        )
        return extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

    def _run_multi_pass(self, sample_pdf: PDF) -> ExtractionResult:
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
        return extractor.extract(
            document=doc,
            llm_config={},
            extraction_config={},
            formatter=FakeFormatter(),
            response_format=SampleResponse,
        )

    def test_extracted_data_matches(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.data == mp.data

    def test_extracted_data_name_matches(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.data.name == mp.data.name == "Alice"

    def test_extracted_data_age_matches(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.data.age == mp.data.age == 30

    def test_metadata_fields_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.metadata is not None
        assert mp.metadata is not None
        assert set(sp.metadata.keys()) == set(mp.metadata.keys())

    def test_metadata_citation_values_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.metadata is not None
        assert mp.metadata is not None
        for field in ("name", "age"):
            assert sp.metadata[field]["value"] == mp.metadata[field]["value"], (
                f"metadata value mismatch for field '{field}'"
            )

    def test_metadata_bboxes_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.metadata is not None
        assert mp.metadata is not None
        for field in ("name", "age"):
            sp_bboxes = sp.metadata[field]["citations"][0]["bboxes"]
            mp_bboxes = mp.metadata[field]["citations"][0]["bboxes"]
            assert sp_bboxes == mp_bboxes, (
                f"bbox mismatch for field '{field}': {sp_bboxes} vs {mp_bboxes}"
            )

    def test_metadata_citation_pages_match(self, sample_pdf: PDF):
        sp = self._run_single_pass(sample_pdf)
        mp = self._run_multi_pass(sample_pdf)
        assert sp.metadata is not None
        assert mp.metadata is not None
        for field in ("name", "age"):
            sp_page = sp.metadata[field]["citations"][0]["page"]
            mp_page = mp.metadata[field]["citations"][0]["page"]
            assert sp_page == mp_page, (
                f"citation page mismatch for field '{field}': {sp_page} vs {mp_page}"
            )
