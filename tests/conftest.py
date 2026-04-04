"""Shared fixtures for the doc_intelligence test suite."""

from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, Field

from doc_intelligence.base import BaseExtractor, BaseFormatter, BaseLLM, BaseParser
from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.pdf.schemas import PDF, Page, PDFDocument, TextBlock
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import (
    BoundingBox,
    Document,
    ExtractionResult,
    Line,
    PydanticModel,
)


# ---------------------------------------------------------------------------
# Fake ABC implementations
# ---------------------------------------------------------------------------
class FakeLayoutDetector(BaseLayoutDetector):
    """A fake layout detector that returns a pre-set list of regions."""

    def __init__(self, regions: list[LayoutRegion] | None = None):
        self.regions = regions or []
        self.call_count = 0
        self.last_image: np.ndarray | None = None

    def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
        self.call_count += 1
        self.last_image = page_image
        return self.regions


class FakeOCREngine(BaseOCREngine):
    """A fake OCR engine that returns a pre-set list of lines."""

    def __init__(self, lines: list[Line] | None = None):
        self.lines = lines or []
        self.call_count = 0
        self.last_image: np.ndarray | None = None

    def ocr(self, region_image: np.ndarray) -> list[Line]:
        self.call_count += 1
        self.last_image = region_image
        return self.lines


class FakeLLM(BaseLLM):
    """A fake LLM that returns canned text responses without making API calls.

    Pass ``responses`` to cycle through multiple replies in call order.
    Falls back to ``text_response`` once the list is exhausted.
    """

    def __init__(
        self,
        text_response: str = '{"name": "test"}',
        responses: list[str] | None = None,
        model: str = "fake-model",
    ):
        super().__init__(model=model)
        self.text_response = text_response
        self.responses = responses
        self._call_index: int = 0
        self.last_call_kwargs: dict[str, Any] = {}
        self.all_calls: list[dict[str, Any]] = []

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        call_kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            **kwargs,
        }
        self.last_call_kwargs = call_kwargs
        self.all_calls.append(call_kwargs)
        if self.responses is not None and self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            self._call_index += 1
            return response
        return self.text_response


class FakeParser(BaseParser[PDFDocument]):
    """A fake parser that returns a pre-set PDFDocument."""

    def __init__(self, result: PDFDocument | None = None):
        self.result = result
        self.call_count = 0

    def parse(self, document: PDFDocument) -> PDFDocument:
        self.call_count += 1
        if self.result is not None:
            return self.result
        return document


class FakeFormatter(BaseFormatter):
    """A fake formatter that returns a fixed string."""

    def __init__(self, output: str = "formatted content"):
        self.output = output

    def format_document_for_llm(self, document: Document, **kwargs) -> str:
        return self.output


class FakeExtractor(BaseExtractor):
    """A fake extractor that returns a pre-set ExtractionResult."""

    def __init__(self, llm: BaseLLM, result: ExtractionResult | None = None):
        super().__init__(llm)
        self.result = result

    def extract(
        self,
        document: Document,
        llm_config: dict[str, Any],
        extraction_config: dict[str, Any],
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
    ) -> ExtractionResult:
        return self.result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sample Pydantic response models (for extractor / processor tests)
# ---------------------------------------------------------------------------
class SimpleExtraction(BaseModel):
    """A minimal extraction model for testing."""

    name: str = Field(..., description="person name")
    age: int = Field(..., description="person age")


class NestedExtraction(BaseModel):
    """An extraction model with a nested sub-model."""

    class AddressInfo(BaseModel):
        street: str
        city: str

    name: str
    address: AddressInfo


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_bbox() -> BoundingBox:
    """A normalized bounding box (values in 0–1 range)."""
    return BoundingBox(x0=0.1, top=0.2, x1=0.5, bottom=0.25)


@pytest.fixture
def sample_bbox_raw() -> BoundingBox:
    """An un-normalized bounding box (pixel coordinates)."""
    return BoundingBox(x0=50.0, top=100.0, x1=250.0, bottom=125.0)


# ---------------------------------------------------------------------------
# PDF structure: Line -> Page -> PDF
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_lines(sample_bbox: BoundingBox) -> list[Line]:
    """Three lines with distinct text and identical bounding boxes."""
    return [
        Line(text="First line of text", bounding_box=sample_bbox),
        Line(text="Second line of text", bounding_box=sample_bbox),
        Line(text="Third line of text", bounding_box=sample_bbox),
    ]


@pytest.fixture
def sample_page(sample_lines: list[Line]) -> Page:
    """A single page with 3 lines in one TextBlock, 500x800 dimensions."""
    return Page(blocks=[TextBlock(lines=sample_lines)], width=500, height=800)


@pytest.fixture
def sample_page_empty() -> Page:
    """A page with no blocks."""
    return Page(blocks=[], width=500, height=800)


@pytest.fixture
def sample_pdf(sample_page: Page) -> PDF:
    """A 2-page PDF (both pages identical)."""
    return PDF(pages=[sample_page, sample_page])


@pytest.fixture
def sample_pdf_single_page(sample_page: Page) -> PDF:
    """A single-page PDF."""
    return PDF(pages=[sample_page])


@pytest.fixture
def sample_pdf_empty() -> PDF:
    """A PDF with no pages."""
    return PDF(pages=[])


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_pdf_document(sample_pdf: PDF) -> PDFDocument:
    """A parsed PDFDocument with citations enabled, single-pass mode."""
    return PDFDocument(
        uri="tests/fixtures/sample.pdf",
        content=sample_pdf,
        include_citations=True,
        extraction_mode=PDFExtractionMode.SINGLE_PASS,
    )


@pytest.fixture
def sample_pdf_document_no_citations(sample_pdf: PDF) -> PDFDocument:
    """A parsed PDFDocument with citations disabled."""
    return PDFDocument(
        uri="tests/fixtures/sample.pdf",
        content=sample_pdf,
        include_citations=False,
        extraction_mode=PDFExtractionMode.SINGLE_PASS,
    )


@pytest.fixture
def sample_pdf_document_unparsed() -> PDFDocument:
    """A PDFDocument that has not been parsed yet (content=None)."""
    return PDFDocument(uri="tests/fixtures/sample.pdf")


# ---------------------------------------------------------------------------
# Fake ABC instances
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_parser(sample_pdf_document: PDFDocument) -> FakeParser:
    return FakeParser(result=sample_pdf_document)


@pytest.fixture
def fake_formatter() -> FakeFormatter:
    return FakeFormatter()


@pytest.fixture
def fake_extractor(fake_llm: FakeLLM) -> FakeExtractor:
    return FakeExtractor(
        llm=fake_llm,
        result=ExtractionResult(data=None, metadata=None),
    )


# ---------------------------------------------------------------------------
# Sample response models
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_extraction_model() -> type[SimpleExtraction]:
    return SimpleExtraction


@pytest.fixture
def nested_extraction_model() -> type[NestedExtraction]:
    return NestedExtraction


# ---------------------------------------------------------------------------
# OCR fakes
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_layout_region(sample_bbox: BoundingBox) -> LayoutRegion:
    """A single text layout region with 90% confidence."""
    return LayoutRegion(bounding_box=sample_bbox, region_type="text", confidence=0.9)


@pytest.fixture
def fake_layout_detector(sample_layout_region: LayoutRegion) -> FakeLayoutDetector:
    return FakeLayoutDetector(regions=[sample_layout_region])


@pytest.fixture
def fake_ocr_engine(sample_lines: list[Line]) -> FakeOCREngine:
    return FakeOCREngine(lines=sample_lines)


# ---------------------------------------------------------------------------
# Citation response dicts (for utils tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def citation_response_simple() -> dict[str, Any]:
    """A response with a single citation-wrapped field."""
    return {
        "name": {
            "value": "Alice",
            "citations": [{"page": 0, "lines": [0, 1]}],
        },
    }


@pytest.fixture
def citation_response_nested() -> dict[str, Any]:
    """A response with nested objects containing citations."""
    return {
        "person": {
            "name": {
                "value": "Bob",
                "citations": [{"page": 0, "lines": [2]}],
            },
            "age": {
                "value": 30,
                "citations": [{"page": 1, "lines": [0]}],
            },
        },
    }


@pytest.fixture
def citation_response_with_list() -> dict[str, Any]:
    """A response with a list of citation-wrapped items."""
    return {
        "ids": [
            {"value": 101, "citations": [{"page": 0, "lines": [0]}]},
            {"value": 205, "citations": [{"page": 0, "lines": [1]}]},
        ],
    }
