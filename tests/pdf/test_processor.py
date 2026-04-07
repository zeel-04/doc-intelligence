"""Tests for processor module."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from doc_intelligence.pdf.processor import DocumentProcessor, PDFProcessor
from doc_intelligence.pdf.schemas import PDF, PDFDocument, PDFExtractionRequest
from doc_intelligence.pdf.types import (
    ParseStrategy,
    PDFExtractionMode,
    ScannedPipelineType,
)
from doc_intelligence.schemas.core import (
    ExtractionRequest,
    ExtractionResult,
)
from tests.conftest import (
    FakeExtractor,
    FakeFormatter,
    FakeLLM,
    FakeParser,
    SimpleExtraction,
)


# ---------------------------------------------------------------------------
# DocumentProcessor.__init__
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
# DocumentProcessor.extract
# ---------------------------------------------------------------------------
class TestDocumentProcessorExtract:
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
        request = PDFExtractionRequest(uri="test.pdf", response_format=SimpleExtraction)
        result = proc.extract(request)
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
        request = PDFExtractionRequest(uri="test.pdf", response_format=SimpleExtraction)
        proc.extract(request)
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
        request1 = PDFExtractionRequest(
            uri="first.pdf", response_format=SimpleExtraction
        )
        request2 = PDFExtractionRequest(
            uri="second.pdf", response_format=SimpleExtraction
        )
        result1 = proc.extract(request1)
        result2 = proc.extract(request2)
        assert result1.data.name == "Multi"
        assert result2.data.name == "Multi"
        assert parser.call_count == 2

    def test_parser_receives_uri(
        self,
        sample_pdf: PDF,
        fake_formatter: FakeFormatter,
        fake_llm: FakeLLM,
    ):
        parsed_doc = PDFDocument(uri="target.pdf", content=sample_pdf)
        parser = FakeParser(result=parsed_doc)
        extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="URI", age=1),
                metadata=None,
            ),
        )
        proc = DocumentProcessor(
            parser=parser,
            formatter=fake_formatter,
            extractor=extractor,
        )
        request = PDFExtractionRequest(
            uri="target.pdf", response_format=SimpleExtraction
        )
        proc.extract(request)
        assert parser.last_uri == "target.pdf"


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

    def test_non_basemodel_response_format_raises(self, fake_llm: FakeLLM):
        proc = PDFProcessor(llm=fake_llm)
        with pytest.raises(ValueError, match="Pydantic model"):
            proc.extract("test.pdf", str)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PDFProcessor — constructor defaults
# ---------------------------------------------------------------------------
class TestPDFProcessorDefaults:
    """Constructor config is used for every extract() call."""

    def _capturing_proc(
        self,
        fake_llm: FakeLLM,
        sample_pdf: PDF,
        captured: list[PDFExtractionRequest],
        **constructor_kwargs,
    ) -> PDFProcessor:
        """Create a PDFProcessor that captures the request it builds."""
        from doc_intelligence.base import BaseFormatter
        from doc_intelligence.schemas.core import Document as BaseDoc

        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        outer_result = ExtractionResult(
            data=SimpleExtraction(name="Cap", age=1), metadata=None
        )

        class CapturingExtractor(FakeExtractor):
            def extract(self, document: BaseDoc, request, formatter: BaseFormatter):
                captured.append(request)  # type: ignore[arg-type]
                return super().extract(document, request, formatter)

        proc = PDFProcessor(llm=fake_llm, **constructor_kwargs)
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = CapturingExtractor(
            llm=fake_llm, result=outer_result
        )
        return proc

    def test_include_citations_false(self, fake_llm: FakeLLM, sample_pdf: PDF):
        captured: list[PDFExtractionRequest] = []
        proc = self._capturing_proc(
            fake_llm, sample_pdf, captured, include_citations=False
        )
        proc.extract("test.pdf", SimpleExtraction)
        assert captured[0].include_citations is False

    def test_include_citations_defaults_true(self, fake_llm: FakeLLM, sample_pdf: PDF):
        captured: list[PDFExtractionRequest] = []
        proc = self._capturing_proc(fake_llm, sample_pdf, captured)
        proc.extract("test.pdf", SimpleExtraction)
        assert captured[0].include_citations is True

    def test_extraction_mode_multi_pass(self, fake_llm: FakeLLM, sample_pdf: PDF):
        captured: list[PDFExtractionRequest] = []
        proc = self._capturing_proc(
            fake_llm,
            sample_pdf,
            captured,
            extraction_mode=PDFExtractionMode.MULTI_PASS,
        )
        proc.extract("test.pdf", SimpleExtraction)
        assert captured[0].extraction_mode == PDFExtractionMode.MULTI_PASS

    def test_extraction_mode_defaults_single_pass(
        self, fake_llm: FakeLLM, sample_pdf: PDF
    ):
        captured: list[PDFExtractionRequest] = []
        proc = self._capturing_proc(fake_llm, sample_pdf, captured)
        proc.extract("test.pdf", SimpleExtraction)
        assert captured[0].extraction_mode == PDFExtractionMode.SINGLE_PASS

    def test_llm_config_forwarded(self, fake_llm: FakeLLM, sample_pdf: PDF):
        captured: list[PDFExtractionRequest] = []
        proc = self._capturing_proc(
            fake_llm,
            sample_pdf,
            captured,
            llm_config={"temperature": 0.2, "max_tokens": 1000},
        )
        proc.extract("test.pdf", SimpleExtraction)
        assert captured[0].llm_config == {"temperature": 0.2, "max_tokens": 1000}

    def test_llm_config_defaults_none(self, fake_llm: FakeLLM, sample_pdf: PDF):
        captured: list[PDFExtractionRequest] = []
        proc = self._capturing_proc(fake_llm, sample_pdf, captured)
        proc.extract("test.pdf", SimpleExtraction)
        assert captured[0].llm_config is None


# ---------------------------------------------------------------------------
# PDFProcessor — restriction checks
# ---------------------------------------------------------------------------
class TestPDFProcessorRestrictions:
    """PDFProcessor.extract() enforces size/page/depth limits."""

    def test_oversized_pdf_raises(self, fake_llm: FakeLLM):
        proc = PDFProcessor(llm=fake_llm)
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
                        proc.extract("test.pdf", SimpleExtraction)

    def test_too_many_pages_raises(self, fake_llm: FakeLLM):
        proc = PDFProcessor(llm=fake_llm)
        with (
            patch("doc_intelligence.restrictions.os.path.isfile", return_value=True),
            patch("doc_intelligence.restrictions.os.path.getsize", return_value=1024),
            patch("doc_intelligence.restrictions.pdfplumber.open") as mock_pdf_open,
            patch("doc_intelligence.pdf.processor.settings") as mock_settings,
        ):
            mock_pdf = mock_pdf_open.return_value.__enter__.return_value
            mock_pdf.pages = [None] * 20  # 20 pages
            mock_settings.max_pdf_size_mb = 1000.0
            mock_settings.max_schema_depth = 10
            mock_settings.max_pdf_pages = 5  # only 5 allowed
            with pytest.raises(ValueError, match="pages, limit is"):
                proc.extract("test.pdf", SimpleExtraction)

    def test_schema_too_deep_raises(self, fake_llm: FakeLLM):
        class Inner(BaseModel):
            val: int

        class Deep(BaseModel):
            inner: Inner

        proc = PDFProcessor(llm=fake_llm)
        with patch("doc_intelligence.pdf.processor.settings") as mock_settings:
            mock_settings.max_pdf_size_mb = 1000.0
            mock_settings.max_schema_depth = 0  # depth 1 model -> raises
            mock_settings.max_pdf_pages = 100
            with pytest.raises(ValueError, match="depth"):
                proc.extract("test.pdf", Deep)


# ---------------------------------------------------------------------------
# PDFProcessor — strategy selection
# ---------------------------------------------------------------------------
class TestPDFProcessorStrategy:
    def test_default_strategy_is_digital(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(llm=fake_llm)
        assert proc._processor.parser._strategy == ParseStrategy.DIGITAL

    def test_digital_strategy(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(llm=fake_llm, strategy=ParseStrategy.DIGITAL)
        assert proc._processor.parser._strategy == ParseStrategy.DIGITAL

    def test_scanned_strategy_default_vlm(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(
            llm=fake_llm,
            strategy=ParseStrategy.SCANNED,
        )
        assert proc._processor.parser._strategy == ParseStrategy.SCANNED
        assert proc._processor.parser._scanned_pipeline == ScannedPipelineType.VLM

    def test_scanned_two_stage_raises_not_implemented(self, fake_llm: FakeLLM) -> None:
        with pytest.raises(NotImplementedError, match="TWO_STAGE.*not yet implemented"):
            PDFProcessor(
                llm=fake_llm,
                strategy=ParseStrategy.SCANNED,
                scanned_pipeline=ScannedPipelineType.TWO_STAGE,
            )

    def test_scanned_vlm_strategy(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(
            llm=fake_llm,
            strategy=ParseStrategy.SCANNED,
            scanned_pipeline=ScannedPipelineType.VLM,
        )
        assert proc._processor.parser._strategy == ParseStrategy.SCANNED
        assert proc._processor.parser._scanned_pipeline == ScannedPipelineType.VLM
        assert proc._processor.parser._llm is fake_llm

    def test_scanned_vlm_batch_size(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(
            llm=fake_llm,
            strategy=ParseStrategy.SCANNED,
            scanned_pipeline=ScannedPipelineType.VLM,
            vlm_batch_size=5,
        )
        assert proc._processor.parser._vlm_batch_size == 5

    def test_scanned_vlm_without_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Either `llm` or `provider`"):
            PDFProcessor(
                strategy=ParseStrategy.SCANNED,
                scanned_pipeline=ScannedPipelineType.VLM,
            )


# ---------------------------------------------------------------------------
# PDFProcessor — extraction mode
# ---------------------------------------------------------------------------
class TestPDFProcessorExtractionMode:
    def test_extraction_mode_single_pass(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
    ) -> None:
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        proc = PDFProcessor(llm=fake_llm, extraction_mode=PDFExtractionMode.SINGLE_PASS)
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="SP", age=1),
                metadata=None,
            ),
        )
        result = proc.extract("test.pdf", SimpleExtraction)
        assert result.data is not None

    def test_extraction_mode_multi_pass(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
    ) -> None:
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        proc = PDFProcessor(llm=fake_llm, extraction_mode=PDFExtractionMode.MULTI_PASS)
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="MP", age=2),
                metadata=None,
            ),
        )
        result = proc.extract("test.pdf", SimpleExtraction)
        assert result.data is not None

    def test_invalid_extraction_mode_raises(
        self, fake_llm: FakeLLM, sample_pdf: PDF
    ) -> None:
        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        proc = PDFProcessor(llm=fake_llm)
        proc._processor.parser = FakeParser(result=parsed_doc)
        proc._processor.extractor = FakeExtractor(
            llm=fake_llm,
            result=ExtractionResult(
                data=SimpleExtraction(name="X", age=1), metadata=None
            ),
        )
        # Invalid mode stored on processor — validated when building the request
        proc._extraction_mode = "invalid_mode"  # type: ignore[assignment]
        with pytest.raises(ValueError):
            proc.extract("test.pdf", SimpleExtraction)
