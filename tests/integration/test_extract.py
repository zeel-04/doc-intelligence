"""Data-driven integration tests for the full extraction pipeline."""

from collections.abc import Callable

import pytest

from doc_intelligence.pdf.processor import PDFProcessor
from tests.conftest import FakeLLM
from tests.integration.conftest import resolve_schema
from tests.integration.test_cases import EXTRACT_CASES, LIVE_EXTRACT_CASES


# ---------------------------------------------------------------------------
# Mocked LLM extraction tests
# ---------------------------------------------------------------------------
class TestExtractMocked:
    """Full pipeline extraction tests using FakeLLM."""

    @pytest.mark.parametrize("case", EXTRACT_CASES, ids=lambda c: c["id"])
    def test_extract(self, case: dict, pdf_path: Callable[[str], str]) -> None:
        schema = resolve_schema(case["schema"])
        llm = FakeLLM(text_response=case["mock_llm_response"])
        processor = PDFProcessor(llm=llm, **case["config"])

        result = processor.extract(pdf_path(case["pdf"]), schema)

        assert result.data is not None
        assert result.data.model_dump() == case["expected_data"]


# ---------------------------------------------------------------------------
# Live LLM extraction tests (requires --run-live)
# ---------------------------------------------------------------------------
class TestExtractLive:
    """Full pipeline extraction tests using real LLM APIs."""

    @pytest.mark.live
    @pytest.mark.parametrize("case", LIVE_EXTRACT_CASES, ids=lambda c: c["id"])
    def test_extract_live(self, case: dict, pdf_path: Callable[[str], str]) -> None:
        schema = resolve_schema(case["schema"])
        processor = PDFProcessor(provider=case["provider"], **case["config"])

        result = processor.extract(pdf_path(case["pdf"]), schema)

        assert result.data is not None
        assert result.data.model_dump() == case["expected_data"]
