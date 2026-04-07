"""Data-driven integration tests for PDFFormatter."""

from collections.abc import Callable

import pytest

from doc_intelligence.pdf.formatter import PDFFormatter
from doc_intelligence.pdf.parser import PDFParser
from tests.integration.test_cases import FORMAT_CASES


# ---------------------------------------------------------------------------
# Format tests
# ---------------------------------------------------------------------------
class TestFormat:
    """Data-driven format tests using real parsed PDFs."""

    @pytest.mark.parametrize("case", FORMAT_CASES, ids=lambda c: c["id"])
    def test_format(self, case: dict, pdf_path: Callable[[str], str]) -> None:
        parser = PDFParser()
        doc = parser.parse(pdf_path(case["pdf"]))

        formatter = PDFFormatter()
        output = formatter.format_document_for_llm(
            doc, include_citations=case["include_citations"]
        )

        for substring in case["expected"]["contains"]:
            assert substring in output, f"Expected {substring!r} in formatted output"

        for substring in case["expected"].get("not_contains", []):
            assert substring not in output, (
                f"Did not expect {substring!r} in formatted output"
            )
