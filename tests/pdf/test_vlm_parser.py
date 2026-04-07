"""Tests for the VLM (Type 2) scanned parsing sub-pipeline."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from doc_intelligence.pdf.parser import (
    PDFParser,
    _encode_image_to_data_url,
    _parse_vlm_response,
)
from doc_intelligence.pdf.schemas import PDF, PDFDocument
from doc_intelligence.pdf.types import ParseStrategy, ScannedPipelineType
from doc_intelligence.schemas.core import (
    ChartBlock,
    ImageBlock,
    TableBlock,
    TextBlock,
)
from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_image(h: int = 100, w: int = 200, channels: int = 3) -> np.ndarray:
    return np.zeros((h, w, channels), dtype=np.uint8)


def _vlm_response_json(pages: list[dict]) -> str:
    return json.dumps({"pages": pages})


def _simple_vlm_response(text: str = "Hello world", block_type: str = "text") -> str:
    return _vlm_response_json(
        [
            {
                "page_index": 0,
                "blocks": [
                    {
                        "block_type": block_type,
                        "bbox": {"x0": 10, "y0": 20, "x1": 190, "y1": 80},
                        "text": text,
                    }
                ],
            }
        ]
    )


def _make_vlm_parser(
    text_response: str = "",
    responses: list[str] | None = None,
    vlm_batch_size: int = 1,
) -> tuple[PDFParser, FakeLLM]:
    fake_llm = FakeLLM(text_response=text_response, responses=responses)
    parser = PDFParser(
        strategy=ParseStrategy.SCANNED,
        scanned_pipeline=ScannedPipelineType.VLM,
        llm=fake_llm,
        vlm_batch_size=vlm_batch_size,
    )
    return parser, fake_llm


# ---------------------------------------------------------------------------
# PDFParser VLM parse
# ---------------------------------------------------------------------------
class TestPDFParserVLMParse:
    def test_returns_pdf_document(self):
        parser, _ = _make_vlm_parser(text_response=_simple_vlm_response())
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image()]
            result = parser.parse("scanned.pdf")
        assert isinstance(result, PDFDocument)

    def test_preserves_uri(self):
        parser, _ = _make_vlm_parser(text_response=_simple_vlm_response())
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image()]
            result = parser.parse("my/scan.pdf")
        assert result.uri == "my/scan.pdf"

    def test_page_count_matches_response(self):
        # With batch_size=2, both pages are sent in one call
        response = _vlm_response_json(
            [
                {"page_index": 0, "blocks": []},
                {"page_index": 1, "blocks": []},
            ]
        )
        parser, _ = _make_vlm_parser(text_response=response, vlm_batch_size=2)
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image(), _make_image()]
            result = parser.parse("scan.pdf")
        assert result.content is not None
        assert len(result.content.pages) == 2

    def test_empty_pdf_returns_no_pages(self):
        parser, _ = _make_vlm_parser(text_response="")
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = []
            result = parser.parse("empty.pdf")
        assert result.content is not None
        assert len(result.content.pages) == 0

    def test_page_dimensions_from_image(self):
        response = _vlm_response_json([{"page_index": 0, "blocks": []}])
        parser, _ = _make_vlm_parser(text_response=response)
        image = _make_image(h=800, w=600)
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [image]
            result = parser.parse("scan.pdf")
        page = result.content.pages[0]
        assert page.width == 600
        assert page.height == 800

    def test_llm_receives_data_urls(self):
        parser, fake_llm = _make_vlm_parser(text_response=_simple_vlm_response())
        images_received = []

        def capture_images(system_prompt, user_prompt, images=None, **kw):
            if images:
                images_received.extend(images)
            return fake_llm.text_response

        fake_llm.generate = capture_images
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image()]
            parser.parse("scan.pdf")
        assert len(images_received) == 1
        assert images_received[0].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# VLM block type mapping
# ---------------------------------------------------------------------------
class TestPDFParserVLMBlocks:
    def test_text_block(self):
        response = _simple_vlm_response("Hello world", "text")
        pages = _parse_vlm_response(response, [(200, 100)])
        block = pages[0].blocks[0]
        assert isinstance(block, TextBlock)
        assert block.lines[0].text == "Hello world"

    def test_header_becomes_text_block(self):
        response = _simple_vlm_response("Document Title", "header")
        pages = _parse_vlm_response(response, [(200, 100)])
        assert isinstance(pages[0].blocks[0], TextBlock)

    def test_footer_becomes_text_block(self):
        response = _simple_vlm_response("Page 1 of 3", "footer")
        pages = _parse_vlm_response(response, [(200, 100)])
        assert isinstance(pages[0].blocks[0], TextBlock)

    def test_table_block(self):
        table_text = "| Name | Age |\n|---|---|\n| Alice | 30 |"
        response = _simple_vlm_response(table_text, "table")
        pages = _parse_vlm_response(response, [(200, 100)])
        block = pages[0].blocks[0]
        assert isinstance(block, TableBlock)
        # Header row + data row (separator is skipped)
        assert len(block.rows) == 2
        assert block.rows[0][0].text == "Name"
        assert block.rows[1][1].text == "30"

    def test_figure_becomes_image_block(self):
        response = _simple_vlm_response("A bar chart", "figure")
        pages = _parse_vlm_response(response, [(200, 100)])
        block = pages[0].blocks[0]
        assert isinstance(block, ImageBlock)
        assert block.description == "A bar chart"

    def test_bbox_mapped_correctly(self):
        response = _simple_vlm_response("text", "text")
        pages = _parse_vlm_response(response, [(200, 100)])
        bbox = pages[0].blocks[0].bounding_box
        # Raw pixel coords (10,20,190,80) normalized by page dims (200,100)
        assert bbox.x0 == pytest.approx(0.05)
        assert bbox.top == pytest.approx(0.2)
        assert bbox.x1 == pytest.approx(0.95)
        assert bbox.bottom == pytest.approx(0.8)

    def test_multiline_text_split_into_lines(self):
        response = _simple_vlm_response("Line one\nLine two\nLine three", "text")
        pages = _parse_vlm_response(response, [(200, 100)])
        block = pages[0].blocks[0]
        assert isinstance(block, TextBlock)
        assert len(block.lines) == 3
        assert block.lines[1].text == "Line two"

    def test_blank_lines_skipped(self):
        response = _simple_vlm_response("Line one\n\n\nLine two", "text")
        pages = _parse_vlm_response(response, [(200, 100)])
        block = pages[0].blocks[0]
        assert len(block.lines) == 2

    def test_empty_text_skipped(self):
        response = _simple_vlm_response("", "text")
        pages = _parse_vlm_response(response, [(200, 100)])
        assert len(pages[0].blocks) == 0

    def test_multiple_blocks_on_one_page(self):
        response = _vlm_response_json(
            [
                {
                    "page_index": 0,
                    "blocks": [
                        {
                            "block_type": "header",
                            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 30},
                            "text": "Title",
                        },
                        {
                            "block_type": "text",
                            "bbox": {"x0": 0, "y0": 40, "x1": 100, "y1": 80},
                            "text": "Body text",
                        },
                        {
                            "block_type": "table",
                            "bbox": {"x0": 0, "y0": 90, "x1": 100, "y1": 120},
                            "text": "| a | b |",
                        },
                    ],
                }
            ]
        )
        pages = _parse_vlm_response(response, [(200, 150)])
        assert len(pages[0].blocks) == 3
        assert isinstance(pages[0].blocks[0], TextBlock)
        assert isinstance(pages[0].blocks[1], TextBlock)
        assert isinstance(pages[0].blocks[2], TableBlock)


# ---------------------------------------------------------------------------
# VLM batching
# ---------------------------------------------------------------------------
class TestPDFParserVLMBatching:
    def test_batch_size_1_makes_n_calls(self):
        response = _vlm_response_json([{"page_index": 0, "blocks": []}])
        call_count = 0

        def counting_llm(system_prompt, user_prompt, images=None, **kw):
            nonlocal call_count
            call_count += 1
            return response

        parser, fake_llm = _make_vlm_parser(vlm_batch_size=1)
        fake_llm.generate = counting_llm
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image(), _make_image(), _make_image()]
            parser.parse("scan.pdf")
        assert call_count == 3

    def test_batch_size_n_makes_1_call(self):
        response = _vlm_response_json(
            [
                {"page_index": 0, "blocks": []},
                {"page_index": 1, "blocks": []},
                {"page_index": 2, "blocks": []},
            ]
        )
        call_count = 0

        def counting_llm(system_prompt, user_prompt, images=None, **kw):
            nonlocal call_count
            call_count += 1
            return response

        parser, fake_llm = _make_vlm_parser(vlm_batch_size=10)
        fake_llm.generate = counting_llm
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image(), _make_image(), _make_image()]
            parser.parse("scan.pdf")
        assert call_count == 1

    def test_batch_receives_correct_number_of_images(self):
        response = _vlm_response_json(
            [{"page_index": 0, "blocks": []}, {"page_index": 1, "blocks": []}]
        )
        received_image_counts = []

        def tracking_llm(system_prompt, user_prompt, images=None, **kw):
            received_image_counts.append(len(images) if images else 0)
            return response

        parser, fake_llm = _make_vlm_parser(vlm_batch_size=2)
        fake_llm.generate = tracking_llm
        with patch("doc_intelligence.pdf.parser._render_pdf_to_images") as mock_render:
            mock_render.return_value = [_make_image(), _make_image(), _make_image()]
            parser.parse("scan.pdf")
        # 3 pages, batch_size=2 → first batch has 2, second has 1
        assert received_image_counts == [2, 1]


# ---------------------------------------------------------------------------
# VLM response parsing edge cases
# ---------------------------------------------------------------------------
class TestVLMResponseParsing:
    def test_markdown_fences_stripped(self):
        raw = '```json\n{"pages": [{"page_index": 0, "blocks": []}]}\n```'
        pages = _parse_vlm_response(raw, [(100, 100)])
        assert len(pages) == 1

    def test_bare_list_response(self):
        raw = json.dumps([{"page_index": 0, "blocks": []}])
        pages = _parse_vlm_response(raw, [(100, 100)])
        assert len(pages) == 1

    def test_single_page_blocks_only(self):
        raw = json.dumps(
            {
                "blocks": [
                    {
                        "block_type": "text",
                        "bbox": {"x0": 0, "y0": 0, "x1": 50, "y1": 50},
                        "text": "Solo page",
                    }
                ]
            }
        )
        pages = _parse_vlm_response(raw, [(100, 100)])
        assert len(pages) == 1
        assert isinstance(pages[0].blocks[0], TextBlock)

    def test_missing_bbox_defaults_to_zero(self):
        raw = _vlm_response_json(
            [
                {
                    "page_index": 0,
                    "blocks": [
                        {"block_type": "text", "bbox": {}, "text": "No bbox"},
                    ],
                }
            ]
        )
        pages = _parse_vlm_response(raw, [(100, 100)])
        bbox = pages[0].blocks[0].bounding_box
        assert bbox.x0 == 0
        assert bbox.top == 0

    def test_empty_pages_list(self):
        raw = _vlm_response_json([])
        pages = _parse_vlm_response(raw, [])
        assert len(pages) == 0

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_vlm_response("not json at all", [(100, 100)])

    def test_unknown_response_shape(self):
        raw = json.dumps({"something_else": 42})
        pages = _parse_vlm_response(raw, [(100, 100)])
        assert len(pages) == 0


# ---------------------------------------------------------------------------
# _encode_image_to_data_url
# ---------------------------------------------------------------------------
class TestEncodeImageToDataUrl:
    def test_returns_data_url_prefix(self):
        image = _make_image(10, 10)
        result = _encode_image_to_data_url(image)
        assert result.startswith("data:image/png;base64,")

    def test_produces_valid_base64(self):
        import base64

        image = _make_image(10, 10)
        result = _encode_image_to_data_url(image)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        # PNG magic bytes
        assert decoded[:4] == b"\x89PNG"

    def test_roundtrip_preserves_shape(self):
        import base64
        from io import BytesIO

        from PIL import Image

        original = _make_image(50, 80, 3)
        data_url = _encode_image_to_data_url(original)
        b64_part = data_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        img = Image.open(BytesIO(decoded))
        assert img.size == (80, 50)  # PIL is (width, height)
