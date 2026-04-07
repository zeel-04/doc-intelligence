"""Data-driven integration tests for PDFParser."""

from collections.abc import Callable

import pytest

from doc_intelligence.pdf.parser import PDFParser
from doc_intelligence.pdf.types import ParseStrategy, ScannedPipelineType
from doc_intelligence.schemas.core import TextBlock
from tests.conftest import FakeLLM
from tests.integration.test_cases import PARSE_CASES, VLM_PARSE_CASES


# ---------------------------------------------------------------------------
# Parse tests (digital)
# ---------------------------------------------------------------------------
class TestParse:
    """Data-driven parse tests using real PDFs.

    With block-level architecture, each digital PDF line is its own
    ``TextBlock``.  ``num_lines`` in test cases maps to the block count,
    and each ``lines[i]`` assertion checks the i-th block's first line.
    """

    @pytest.mark.parametrize("case", PARSE_CASES, ids=lambda c: c["id"])
    def test_parse(self, case: dict, pdf_path: Callable[[str], str]) -> None:
        parser = PDFParser()
        result = parser.parse(pdf_path(case["pdf"]))

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


# ---------------------------------------------------------------------------
# VLM parse tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestVLMParse:
    """Data-driven VLM parse tests using mocked LLM responses.

    Uses real PDFs for page rendering but fakes the VLM response.
    """

    @pytest.mark.parametrize("case", VLM_PARSE_CASES, ids=lambda c: c["id"])
    def test_vlm_parse(self, case: dict, pdf_path: Callable[[str], str]) -> None:
        fake_llm = FakeLLM(text_response=case["mock_vlm_response"])
        parser = PDFParser(
            strategy=ParseStrategy.SCANNED,
            scanned_pipeline=ScannedPipelineType.VLM,
            llm=fake_llm,
        )
        result = parser.parse(pdf_path(case["pdf"]))

        expected = case["expected"]
        assert result.content is not None
        assert len(result.content.pages) == expected["num_pages"]

        for page_spec in expected["pages"]:
            page = result.content.pages[page_spec["page_index"]]
            blocks = [b for b in page.blocks if isinstance(b, TextBlock)]
            assert len(blocks) == page_spec["num_blocks"]

            for i, block_spec in enumerate(page_spec["blocks"]):
                block = blocks[i]
                block_text = " ".join(line.text for line in block.lines)
                assert block_spec["contains"] in block_text, (
                    f"Block {i}: expected {block_spec['contains']!r} in {block_text!r}"
                )
