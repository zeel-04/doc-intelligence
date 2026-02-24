"""Shared fixtures for the doc_intelligence test suite."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.base import BaseExtractor, BaseFormatter, BaseLLM, BaseParser
from doc_intelligence.schemas.core import BoundingBox, Document, PydanticModel
from doc_intelligence.schemas.pdf import PDF, Line, Page, PDFDocument
from doc_intelligence.types.pdf import PDFExtractionMode


# ---------------------------------------------------------------------------
# Fake ABC implementations
# ---------------------------------------------------------------------------
class FakeLLM(BaseLLM):
    """A fake LLM that returns canned text responses without making API calls."""

    def __init__(self, text_response: str = '{"name": "test"}'):
        self.text_response = text_response
        self.last_call_kwargs: dict[str, Any] = {}

    def generate_structured_output(
        self,
        model: str,
        messages: list[dict[str, str]],
        reasoning: Any,
        output_format: type[PydanticModel],
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel | None:
        return None

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        self.last_call_kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            **kwargs,
        }
        return self.text_response


class FakeParser(BaseParser):
    """A fake parser that returns a pre-set PDFDocument."""

    def __init__(self, result: Document | None = None):
        self.result = result
        self.call_count = 0

    def parse(self, document: Document) -> PDFDocument:
        self.call_count += 1
        if self.result is not None:
            return self.result  # type: ignore[return-value]
        return document  # type: ignore[return-value]


class FakeFormatter(BaseFormatter):
    """A fake formatter that returns a fixed string."""

    def __init__(self, output: str = "formatted content"):
        self.output = output

    def format_document_for_llm(self, document: Document, **kwargs) -> str:
        return self.output


class FakeExtractor(BaseExtractor):
    """A fake extractor that returns a pre-set result."""

    def __init__(self, llm: BaseLLM, result: dict[str, Any] | None = None):
        super().__init__(llm)
        self.result = result

    def extract(
        self,
        document: Document,
        llm_config: dict[str, Any],
        extraction_config: dict[str, Any],
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
    ) -> dict[str, Any] | None:
        return self.result


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
    """A single page with 3 lines, 500x800 dimensions."""
    return Page(lines=sample_lines, width=500, height=800)


@pytest.fixture
def sample_page_empty() -> Page:
    """A page with no lines."""
    return Page(lines=[], width=500, height=800)


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
        result={"extracted_data": None, "metadata": None},
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
