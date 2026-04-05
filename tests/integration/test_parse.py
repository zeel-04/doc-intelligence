"""Data-driven integration tests for DigitalPDFParser."""

from collections.abc import Callable

import pytest

from doc_intelligence.pdf.parser import DigitalPDFParser
from doc_intelligence.pdf.schemas import PDFDocument
from doc_intelligence.schemas.core import TextBlock
from tests.integration.test_cases import PARSE_CASES


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------
class TestParse:
    """Data-driven parse tests using real PDFs.

    With block-level architecture, each digital PDF line is its own
    ``TextBlock``.  ``num_lines`` in test cases maps to the block count,
    and each ``lines[i]`` assertion checks the i-th block's first line.
    """

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
            blocks = page.blocks
            assert len(blocks) == page_spec["num_lines"]

            for i, line_spec in enumerate(page_spec["lines"]):
                block = blocks[i]
                assert isinstance(block, TextBlock)
                assert line_spec["contains"] in block.lines[0].text, (
                    f"Block {i}: expected {line_spec['contains']!r} "
                    f"in {block.lines[0].text!r}"
                )
