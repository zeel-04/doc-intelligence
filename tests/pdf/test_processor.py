"""Tests for processer module."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from doc_intelligence.pdf.extractor import DigitalPDFExtractor
from doc_intelligence.pdf.formatter import DigitalPDFFormatter
from doc_intelligence.pdf.parser import DigitalPDFParser
from doc_intelligence.pdf.processor import DocumentProcessor
from doc_intelligence.pdf.schemas import PDF, PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode
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
        sample_pdf_document: PDFDocument,
    ):
        proc = DocumentProcessor(
            parser=fake_parser,
            formatter=fake_formatter,
            extractor=fake_extractor,
            document=sample_pdf_document,
        )
        assert proc.parser is fake_parser
        assert proc.formatter is fake_formatter
        assert proc.extractor is fake_extractor
        assert proc.document is sample_pdf_document


# ---------------------------------------------------------------------------
# from_digital_pdf
# ---------------------------------------------------------------------------
class TestFromDigitalPDF:
    def test_creates_correct_types(self, fake_llm: FakeLLM):
        proc = DocumentProcessor.from_digital_pdf(uri="test.pdf", llm=fake_llm)
        assert isinstance(proc.parser, DigitalPDFParser)
        assert isinstance(proc.formatter, DigitalPDFFormatter)
        assert isinstance(proc.extractor, DigitalPDFExtractor)
        assert isinstance(proc.document, PDFDocument)

    def test_sets_uri(self, fake_llm: FakeLLM):
        proc = DocumentProcessor.from_digital_pdf(uri="/path/to/doc.pdf", llm=fake_llm)
        assert proc.document.uri == "/path/to/doc.pdf"

    def test_document_content_initially_none(self, fake_llm: FakeLLM):
        proc = DocumentProcessor.from_digital_pdf(uri="test.pdf", llm=fake_llm)
        assert proc.document.content is None


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
class TestParse:
    def test_delegates_to_parser(
        self,
        fake_parser: FakeParser,
        fake_formatter: FakeFormatter,
        fake_extractor: FakeExtractor,
        sample_pdf_document_unparsed: PDFDocument,
    ):
        proc = DocumentProcessor(
            parser=fake_parser,
            formatter=fake_formatter,
            extractor=fake_extractor,
            document=sample_pdf_document_unparsed,
        )
        proc.parse()
        assert fake_parser.call_count == 1

    def test_sets_document_content(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_extractor: FakeExtractor,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        parser = FakeParser(result=parsed_doc)
        unparsed = PDFDocument(uri="test.pdf")

        proc = DocumentProcessor(
            parser=parser,
            formatter=fake_formatter,
            extractor=fake_extractor,
            document=unparsed,
        )
        result = proc.parse()
        assert result.content is not None
        assert result.content is sample_pdf

    def test_returns_document(
        self,
        fake_parser: FakeParser,
        fake_formatter: FakeFormatter,
        fake_extractor: FakeExtractor,
        sample_pdf_document_unparsed: PDFDocument,
    ):
        proc = DocumentProcessor(
            parser=fake_parser,
            formatter=fake_formatter,
            extractor=fake_extractor,
            document=sample_pdf_document_unparsed,
        )
        result = proc.parse()
        assert result is proc.document


# ---------------------------------------------------------------------------
# extract — config validation
# ---------------------------------------------------------------------------
class TestExtractConfigValidation:
    def _make_processor(self, sample_pdf_document, fake_extractor, fake_formatter):
        return DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=fake_extractor,
            document=sample_pdf_document,
        )

    def test_invalid_key_raises(
        self,
        sample_pdf_document: PDFDocument,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        proc = self._make_processor(sample_pdf_document, fake_extractor, fake_formatter)
        with pytest.raises(ValueError, match="Invalid key"):
            proc.extract(config={"bad_key": 123})

    def test_missing_response_format_raises(
        self,
        sample_pdf_document: PDFDocument,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        proc = self._make_processor(sample_pdf_document, fake_extractor, fake_formatter)
        with pytest.raises(ValueError, match="Pydantic model"):
            proc.extract(config={})

    def test_non_basemodel_response_format_raises(
        self,
        sample_pdf_document: PDFDocument,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        proc = self._make_processor(sample_pdf_document, fake_extractor, fake_formatter)
        with pytest.raises(ValueError, match="Pydantic model"):
            proc.extract(config={"response_format": str})


# ---------------------------------------------------------------------------
# extract — delegation and auto-parse
# ---------------------------------------------------------------------------
class TestExtractDelegation:
    def test_delegates_to_extractor(
        self,
        sample_pdf_document: PDFDocument,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="Test", age=1),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=extractor,
            document=sample_pdf_document,
        )
        result = proc.extract(config={"response_format": SimpleExtraction})
        assert result["extracted_data"].name == "Test"

    def test_auto_parses_if_no_content(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        parser = FakeParser(result=parsed_doc)
        unparsed = PDFDocument(uri="test.pdf")
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="Auto", age=2),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=parser,
            formatter=fake_formatter,
            extractor=extractor,
            document=unparsed,
        )
        proc.extract(config={"response_format": SimpleExtraction})
        assert parser.call_count == 1

    def test_skips_parse_if_content_exists(
        self,
        sample_pdf_document: PDFDocument,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        parser = FakeParser(result=sample_pdf_document)
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="Skip", age=3),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=parser,
            formatter=fake_formatter,
            extractor=extractor,
            document=sample_pdf_document,
        )
        proc.extract(config={"response_format": SimpleExtraction})
        assert parser.call_count == 0


# ---------------------------------------------------------------------------
# extract — extraction_config application
# ---------------------------------------------------------------------------
class TestExtractConfigApplication:
    def test_applies_include_citations(
        self,
        sample_pdf_document: PDFDocument,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="a", age=1),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=extractor,
            document=sample_pdf_document,
        )
        proc.extract(
            config={
                "response_format": SimpleExtraction,
                "extraction_config": {
                    "include_citations": False,
                    "extraction_mode": "single_pass",
                },
            }
        )
        assert proc.document.include_citations is False

    def test_applies_extraction_mode(
        self,
        sample_pdf_document: PDFDocument,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="a", age=1),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=extractor,
            document=sample_pdf_document,
        )
        proc.extract(
            config={
                "response_format": SimpleExtraction,
                "extraction_config": {
                    "include_citations": True,
                    "extraction_mode": "single_pass",
                },
            }
        )
        assert proc.document.extraction_mode == PDFExtractionMode.SINGLE_PASS

    def test_default_extraction_config(
        self,
        sample_pdf_document: PDFDocument,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="a", age=1),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=extractor,
            document=sample_pdf_document,
        )
        proc.extract(config={"response_format": SimpleExtraction})
        assert proc.document.include_citations is True
        assert proc.document.extraction_mode == PDFExtractionMode.SINGLE_PASS


# ---------------------------------------------------------------------------
# extract — restriction checks
# ---------------------------------------------------------------------------
class TestExtractRestrictions:
    """DocumentProcessor.extract() enforces size/page/depth limits."""

    def _make_processor(self, sample_pdf_document, fake_extractor, fake_formatter):
        return DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=fake_extractor,
            document=sample_pdf_document,
        )

    def test_oversized_pdf_raises(
        self,
        sample_pdf_document: PDFDocument,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        from unittest.mock import patch

        proc = self._make_processor(sample_pdf_document, fake_extractor, fake_formatter)
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
                        proc.extract(config={"response_format": SimpleExtraction})

    def test_too_many_pages_raises(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        from unittest.mock import patch

        # Build a document that's already parsed (has content)
        doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        extractor = FakeExtractor(
            llm=fake_llm,
            result={
                "extracted_data": SimpleExtraction(name="a", age=1),
                "metadata": None,
            },
        )
        proc = DocumentProcessor(
            parser=FakeParser(),
            formatter=fake_formatter,
            extractor=extractor,
            document=doc,
        )
        with patch("doc_intelligence.pdf.processor.settings") as mock_settings:
            mock_settings.max_pdf_size_mb = 1000.0  # no size limit
            mock_settings.max_schema_depth = 10
            mock_settings.max_pdf_pages = 1  # only 1 page allowed; sample has 2
            with pytest.raises(ValueError, match="pages, limit is"):
                proc.extract(config={"response_format": SimpleExtraction})

    def test_schema_too_deep_raises(
        self,
        sample_pdf_document: PDFDocument,
        fake_extractor: FakeExtractor,
        fake_formatter: FakeFormatter,
    ):
        from unittest.mock import patch

        from pydantic import BaseModel as BM

        class Inner(BM):
            val: int

        class Deep(BM):
            inner: Inner

        proc = self._make_processor(sample_pdf_document, fake_extractor, fake_formatter)
        with patch("doc_intelligence.pdf.processor.settings") as mock_settings:
            mock_settings.max_pdf_size_mb = 1000.0
            mock_settings.max_schema_depth = 0  # depth 1 model → raises
            mock_settings.max_pdf_pages = 100
            with pytest.raises(ValueError, match="depth"):
                proc.extract(config={"response_format": Deep})
