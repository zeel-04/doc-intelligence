"""Data-driven integration tests for DigitalPDFParser."""

from collections.abc import Callable

import pytest

from doc_intelligence.pdf.parser import DigitalPDFParser
from doc_intelligence.pdf.schemas import PDFDocument, blocks_to_lines
from tests.integration.test_cases import PARSE_CASES


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------
class TestParse:
    """Data-driven parse tests using real PDFs."""

    @pytest.mark.parametrize("case", PARSE_CASES, ids=lambda c: c["id"])
    def test_parse(self, case: dict, pdf_path: Callable[[str], str]) -> None:
        parser = DigitalPDFParser()
        doc = PDFDocument(uri=pdf_path(case["pdf"]))
        result = parser.parse(doc)

        expected = case["expected"]
        assert result.content is not None
        assert len(result.content.pages) == expected["num_pages"]

        for page_spec in expected["pages"]:
            page = result.content.pages[page_spec["page_index"]]
            lines = blocks_to_lines(page.blocks)
            assert len(lines) == page_spec["num_lines"]

            for i, line_spec in enumerate(page_spec["lines"]):
                assert line_spec["contains"] in lines[i].text, (
                    f"Line {i}: expected {line_spec['contains']!r} in {lines[i].text!r}"
                )
