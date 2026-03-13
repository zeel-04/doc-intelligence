"""Tests for processor module."""

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.pdf.extractor import DigitalPDFExtractor
from doc_intelligence.pdf.formatter import DigitalPDFFormatter
from doc_intelligence.pdf.parser import DigitalPDFParser
from doc_intelligence.pdf.processor import DocumentProcessor, PDFProcessor
from doc_intelligence.pdf.schemas import PDF, PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import ExtractionResult
from tests.conftest import (
    FakeExtractor,
    FakeFormatter,
    FakeLLM,
    FakeParser,
    SimpleExtraction,
)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestDocumentProcessorInit:
    def test_stores_components(
        self,
        fake_parser: FakeParser,
        fake_formatter: FakeFormatter,
        fake_extractor: FakeExtractor,
    ):
        proc = DocumentProcessor(
            parser=fake_parser,
            formatter=fake_formatter,
            extractor=fake_extractor,
        )
        assert proc.parser is fake_parser
        assert proc.formatter is fake_formatter
        assert proc.extractor is fake_extractor

    def test_no_document_attribute(
        self,
        fake_parser: FakeParser,
        fake_formatter: FakeFormatter,
        fake_extractor: FakeExtractor,
    ):
        proc = DocumentProcessor(
            parser=fake_parser,
            formatter=fake_formatter,
            extractor=fake_extractor,
        )
        assert not hasattr(proc, "document")


# ---------------------------------------------------------------------------
# from_digital_pdf
# ---------------------------------------------------------------------------
class TestFromDigitalPDF:
    def test_creates_correct_types(self, fake_llm: FakeLLM):
        proc = DocumentProcessor.from_digital_pdf(llm=fake_llm)
        assert isinstance(proc.parser, DigitalPDFParser)
        assert isinstance(proc.formatter, DigitalPDFFormatter)
        assert isinstance(proc.extractor, DigitalPDFExtractor)

    def test_no_document_on_processor(self, fake_llm: FakeLLM):
        proc = DocumentProcessor.from_digital_pdf(llm=fake_llm)
        assert not hasattr(proc, "document")


# ---------------------------------------------------------------------------
# extract — argument validation
# ---------------------------------------------------------------------------
class TestExtractValidation:
    def _make_processor(self, fake_extractor, fake_formatter):
        return DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=fake_extractor,
        )

    def test_non_basemodel_response_format_raises(
        self,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        proc = self._make_processor(fake_extractor, fake_formatter)
        with pytest.raises(ValueError, match="Pydantic model"):
            proc.extract(uri="test.pdf", response_format=str)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract — delegation and auto-parse
# ---------------------------------------------------------------------------
class TestExtractDelegation:
    def test_delegates_to_extractor(
        self,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
        sample_pdf: PDF,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="Test", age=1),
                metadata=None,
            ),
        )
        proc = DocumentProcessor(
            parser=FakeParser(result=parsed_doc),
            formatter=fake_formatter,
            extractor=extractor,
        )
        result = proc.extract(uri="test.pdf", response_format=SimpleExtraction)
        assert result.data.name == "Test"

    def test_auto_parses_document(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        parser = FakeParser(result=parsed_doc)
        extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="Auto", age=2),
                metadata=None,
            ),
        )
        proc = DocumentProcessor(
            parser=parser,
            formatter=fake_formatter,
            extractor=extractor,
        )
        proc.extract(uri="test.pdf", response_format=SimpleExtraction)
        assert parser.call_count == 1

    def test_reusable_across_multiple_uris(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        """Processor can extract from different URIs without re-creation."""
        parsed_doc = PDFDocument(uri="any.pdf", content=sample_pdf)
        parser = FakeParser(result=parsed_doc)
        extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="Multi", age=3),
                metadata=None,
            ),
        )
        proc = DocumentProcessor(
            parser=parser,
            formatter=fake_formatter,
            extractor=extractor,
        )
        result1 = proc.extract(uri="first.pdf", response_format=SimpleExtraction)
        result2 = proc.extract(uri="second.pdf", response_format=SimpleExtraction)
        assert result1.data.name == "Multi"
        assert result2.data.name == "Multi"
        assert parser.call_count == 2


# ---------------------------------------------------------------------------
# extract — extraction options
# ---------------------------------------------------------------------------
class TestExtractOptions:
    def _make_processor(self, sample_pdf, fake_llm):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        return DocumentProcessor(
            parser=FakeParser(result=parsed_doc),
            formatter=FakeFormatter(),
            extractor=FakeExtractor(
                llm=fake_llm,
                result=ExtractionResult(
                    data=SimpleExtraction(name="a", age=1),
                    metadata=None,
                ),
            ),
        )

    def test_default_options(self, sample_pdf: PDF, fake_llm: FakeLLM):
        proc = self._make_processor(sample_pdf, fake_llm)
        # Should not raise with defaults
        result = proc.extract(uri="test.pdf", response_format=SimpleExtraction)
        assert result.data.name == "a"

    def test_include_citations_false(self, sample_pdf: PDF, fake_llm: FakeLLM):
        proc = self._make_processor(sample_pdf, fake_llm)
        result = proc.extract(
            uri="test.pdf",
            response_format=SimpleExtraction,
            include_citations=False,
        )
        assert result.data is not None

    def test_extraction_mode_multi_pass(self, sample_pdf: PDF, fake_llm: FakeLLM):
        proc = self._make_processor(sample_pdf, fake_llm)
        result = proc.extract(
            uri="test.pdf",
            response_format=SimpleExtraction,
            extraction_mode="multi_pass",
        )
        assert result.data is not None

    def test_llm_config_forwarded(self, sample_pdf: PDF, fake_llm: FakeLLM):
        proc = self._make_processor(sample_pdf, fake_llm)
        result = proc.extract(
            uri="test.pdf",
            response_format=SimpleExtraction,
            llm_config={"model": "gpt-4o", "temperature": 0.5},
        )
        assert result.data is not None

    def test_invalid_extraction_mode_raises(self, sample_pdf: PDF, fake_llm: FakeLLM):
        proc = self._make_processor(sample_pdf, fake_llm)
        with pytest.raises(ValueError):
            proc.extract(
                uri="test.pdf",
                response_format=SimpleExtraction,
                extraction_mode="invalid_mode",
            )


# ---------------------------------------------------------------------------
# extract — restriction checks
# ---------------------------------------------------------------------------
class TestExtractRestrictions:
    """DocumentProcessor.extract() enforces size/page/depth limits."""

    def _make_processor(self, fake_extractor, fake_formatter):
        return DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=fake_extractor,
        )

    def test_oversized_pdf_raises(
        self,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        proc = self._make_processor(fake_extractor, fake_formatter)
        with patch("doc_intelligence.restrictions.os.path.isfile", return_value=True):
            with patch(
                "doc_intelligence.restrictions.os.path.getsize",
                return_value=20 * 1024 * 1024,  # 20 MB
            ):
                with patch("doc_intelligence.pdf.processor.settings") as mock_settings:
                    mock_settings.max_pdf_size_mb = 10.0
                    mock_settings.max_schema_depth = 10
                    mock_settings.max_pdf_pages = 100
                    with pytest.raises(ValueError, match="exceeds max size"):
                        proc.extract(uri="test.pdf", response_format=SimpleExtraction)

    def test_too_many_pages_raises(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="a", age=1),
                metadata=None,
            ),
        )
        proc = DocumentProcessor(
            parser=FakeParser(result=parsed_doc),
            formatter=fake_formatter,
            extractor=extractor,
        )
        with patch("doc_intelligence.pdf.processor.settings") as mock_settings:
            mock_settings.max_pdf_size_mb = 1000.0  # no size limit
            mock_settings.max_schema_depth = 10
            mock_settings.max_pdf_pages = 1  # only 1 page allowed; sample has 2
            with pytest.raises(ValueError, match="pages, limit is"):
                proc.extract(uri="test.pdf", response_format=SimpleExtraction)

    def test_schema_too_deep_raises(
        self,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        class Inner(BaseModel):
            val: int

        class Deep(BaseModel):
            inner: Inner

        proc = self._make_processor(fake_extractor, fake_formatter)
        with patch("doc_intelligence.pdf.processor.settings") as mock_settings:
            mock_settings.max_pdf_size_mb = 1000.0
            mock_settings.max_schema_depth = 0  # depth 1 model → raises
            mock_settings.max_pdf_pages = 100
            with pytest.raises(ValueError, match="depth"):
                proc.extract(uri="test.pdf", response_format=Deep)


# ---------------------------------------------------------------------------
# PDFProcessor
# ---------------------------------------------------------------------------
class TestPDFProcessor:
    def test_creation_with_llm(self, fake_llm: FakeLLM):
        proc = PDFProcessor(llm=fake_llm)
        assert proc._llm is fake_llm

    def test_creation_with_provider(self):
        with patch("doc_intelligence.pdf.processor.create_llm") as mock_factory:
            mock_factory.return_value = FakeLLM()
            proc = PDFProcessor(provider="openai", model="gpt-4o")
            mock_factory.assert_called_once_with("openai", "gpt-4o")
            assert proc._llm is mock_factory.return_value

    def test_creation_without_llm_or_provider_raises(self):
        with pytest.raises(ValueError, match="Either `llm` or `provider`"):
            PDFProcessor()

    def test_extract_delegates_to_processor(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        proc = PDFProcessor(llm=fake_llm)
        # Monkey-patch the inner processor's parser so it returns content
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="PDF", age=42),
                metadata=None,
            ),
        )
        result = proc.extract("test.pdf", SimpleExtraction)
        assert result.data.name == "PDF"
        assert result.data.age == 42

    def test_model_override_in_extract(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        proc = PDFProcessor(llm=fake_llm)
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="Override", age=1),
                metadata=None,
            ),
        )
        result = proc.extract("test.pdf", SimpleExtraction, model="gpt-4o-mini")
        assert result.data.name == "Override"

    def test_reusable_across_uris(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="any.pdf", content=sample_pdf)
        parser = FakeParser(result=parsed_doc)
        proc = PDFProcessor(llm=fake_llm)
        proc._processor.parser = parser
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="Reuse", age=99),
                metadata=None,
            ),
        )
        r1 = proc.extract("a.pdf", SimpleExtraction)
        r2 = proc.extract("b.pdf", SimpleExtraction)
        assert r1.data.name == r2.data.name == "Reuse"
        assert parser.call_count == 2

    def test_llm_config_merged_with_model(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
    ):
        """When both model and llm_config are given, model takes precedence."""
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        proc = PDFProcessor(llm=fake_llm)
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="Merge", age=1),
                metadata=None,
            ),
        )
        # model="override" should replace the model key in llm_config
        result = proc.extract(
            "test.pdf",
            SimpleExtraction,
            model="override",
            llm_config={"model": "original", "temperature": 0.5},
        )
        assert result.data is not None
