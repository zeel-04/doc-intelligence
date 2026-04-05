"""Tests for processor module."""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import BaseModel, Field

from doc_intelligence.pdf.extractor import PDFExtractor
from doc_intelligence.pdf.formatter import PDFFormatter
from doc_intelligence.pdf.parser import DigitalPDFParser, ScannedPDFParser
from doc_intelligence.pdf.processor import DocumentProcessor, PDFProcessor
from doc_intelligence.pdf.schemas import PDF, PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import BoundingBox, ExtractionResult
from tests.conftest import (
    FakeExtractor,
    FakeFormatter,
    FakeLayoutDetector,
    FakeLLM,
    FakeOCREngine,
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
        assert isinstance(proc.formatter, PDFFormatter)
        assert isinstance(proc.extractor, PDFExtractor)

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


# ---------------------------------------------------------------------------
# extract — page_numbers propagation
# ---------------------------------------------------------------------------
class TestExtractPageNumbersPropagation:
    """page_numbers passed to extract() is stored on the PDFDocument."""

    def _make_capturing_processor(
        self,
        sample_pdf: PDF,
        fake_llm: FakeLLM,
        captured: list[PDFDocument],
    ) -> DocumentProcessor:
        """Return a processor whose extractor records the document it receives."""
        from doc_intelligence.base import BaseExtractor, BaseFormatter
        from doc_intelligence.schemas.core import Document as BaseDoc

        class CapturingExtractor(FakeExtractor):
            def extract(
                self,
                document: BaseDoc,
                llm_config: dict,
                extraction_config: dict,
                formatter: BaseFormatter,
                response_format: type,
            ) -> ExtractionResult:
                captured.append(document)  # type: ignore[arg-type]
                return super().extract(
                    document, llm_config, extraction_config, formatter, response_format
                )

        parsed_doc = PDFDocument(uri="test.pdf", content=sample_pdf)
        return DocumentProcessor(
            parser=FakeParser(result=parsed_doc),
            formatter=FakeFormatter(),
            extractor=CapturingExtractor(
                llm=fake_llm,
                result=ExtractionResult(
                    data=SimpleExtraction(name="a", age=1),
                    metadata=None,
                ),
            ),
        )

    def test_page_numbers_propagated_to_document(
        self, sample_pdf: PDF, fake_llm: FakeLLM
    ):
        captured: list[PDFDocument] = []
        proc = self._make_capturing_processor(sample_pdf, fake_llm, captured)
        proc.extract(
            uri="test.pdf",
            response_format=SimpleExtraction,
            page_numbers=[0, 1],
        )
        assert captured[0].page_numbers == [0, 1]

    def test_page_numbers_none_by_default(self, sample_pdf: PDF, fake_llm: FakeLLM):
        captured: list[PDFDocument] = []
        proc = self._make_capturing_processor(sample_pdf, fake_llm, captured)
        proc.extract(uri="test.pdf", response_format=SimpleExtraction)
        assert captured[0].page_numbers is None

    def test_page_numbers_single_entry(self, sample_pdf: PDF, fake_llm: FakeLLM):
        captured: list[PDFDocument] = []
        proc = self._make_capturing_processor(sample_pdf, fake_llm, captured)
        proc.extract(
            uri="test.pdf",
            response_format=SimpleExtraction,
            page_numbers=[2],
        )
        assert captured[0].page_numbers == [2]


# ---------------------------------------------------------------------------
# from_scanned_pdf — wiring
# ---------------------------------------------------------------------------
class TestFromScannedPDF:
    def test_creates_scanned_parser(self, fake_llm: FakeLLM) -> None:
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert isinstance(proc.parser, ScannedPDFParser)

    def test_uses_pdf_formatter(self, fake_llm: FakeLLM) -> None:
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert isinstance(proc.formatter, PDFFormatter)

    def test_uses_pdf_extractor(self, fake_llm: FakeLLM) -> None:
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert isinstance(proc.extractor, PDFExtractor)

    def test_custom_layout_detector_injected(self, fake_llm: FakeLLM) -> None:
        detector = FakeLayoutDetector()
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=detector,
            ocr_engine=FakeOCREngine(),
        )
        assert proc.parser._layout_detector is detector  # type: ignore[attr-defined]

    def test_custom_ocr_engine_injected(self, fake_llm: FakeLLM) -> None:
        engine = FakeOCREngine()
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=FakeLayoutDetector(),
            ocr_engine=engine,
        )
        assert proc.parser._ocr_engine is engine  # type: ignore[attr-defined]

    def test_custom_dpi_forwarded(self, fake_llm: FakeLLM) -> None:
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
            dpi=300,
        )
        assert proc.parser._dpi == 300  # type: ignore[attr-defined]

    def test_default_dpi_is_150(self, fake_llm: FakeLLM) -> None:
        proc = DocumentProcessor.from_scanned_pdf(
            llm=fake_llm,
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert proc.parser._dpi == 150  # type: ignore[attr-defined]

    def test_missing_layout_detector_raises(self, fake_llm: FakeLLM) -> None:
        """from_scanned_pdf() raises ValueError when layout_detector is None."""
        with pytest.raises(ValueError, match="layout_detector is required"):
            DocumentProcessor.from_scanned_pdf(
                llm=fake_llm,
                ocr_engine=FakeOCREngine(),
            )

    def test_missing_ocr_engine_raises(self, fake_llm: FakeLLM) -> None:
        """from_scanned_pdf() raises ValueError when ocr_engine is None."""
        with pytest.raises(ValueError, match="ocr_engine is required"):
            DocumentProcessor.from_scanned_pdf(
                llm=fake_llm,
                layout_detector=FakeLayoutDetector(),
            )


# ---------------------------------------------------------------------------
# from_scanned_pdf — end-to-end pipeline
# ---------------------------------------------------------------------------
class TestScannedPipelineEndToEnd:
    """Full extract() call through the scanned pipeline with fake components."""

    def _make_scanned_processor(
        self, fake_llm: FakeLLM, result: ExtractionResult
    ) -> DocumentProcessor:
        from doc_intelligence.ocr.base import LayoutRegion
        from doc_intelligence.schemas.core import BoundingBox, Line

        bbox = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=0.1)
        regions = [LayoutRegion(bounding_box=bbox, region_type="text", confidence=0.9)]
        lines = [Line(text="sample text", bounding_box=bbox)]

        return DocumentProcessor(
            parser=ScannedPDFParser(
                layout_detector=FakeLayoutDetector(regions=regions),
                ocr_engine=FakeOCREngine(lines=lines),
            ),
            formatter=PDFFormatter(),
            extractor=FakeExtractor(llm=fake_llm, result=result),
        )

    def test_extract_returns_extraction_result(self, fake_llm: FakeLLM) -> None:
        expected = ExtractionResult(
            data=SimpleExtraction(name="OCR", age=1), metadata=None
        )
        proc = self._make_scanned_processor(fake_llm, expected)

        with patch(
            "doc_intelligence.pdf.parser._render_pdf_to_images",
            return_value=[np.zeros((100, 80, 3), dtype=np.uint8)],
        ):
            result = proc.extract(uri="scan.pdf", response_format=SimpleExtraction)

        assert isinstance(result, ExtractionResult)
        assert result.data.name == "OCR"

    def test_extract_page_has_text_block(self, fake_llm: FakeLLM) -> None:
        """Parsed PDFDocument must contain a TextBlock from OCR output."""
        from doc_intelligence.base import BaseExtractor, BaseFormatter
        from doc_intelligence.schemas.core import Document as BaseDoc
        from doc_intelligence.schemas.core import TextBlock

        captured: list[PDFDocument] = []

        class CapturingExtractor(FakeExtractor):
            def extract(
                self,
                document: BaseDoc,
                llm_config: dict,
                extraction_config: dict,
                formatter: BaseFormatter,
                response_format: type,
            ) -> ExtractionResult:
                captured.append(document)  # type: ignore[arg-type]
                return super().extract(
                    document, llm_config, extraction_config, formatter, response_format
                )

        from doc_intelligence.ocr.base import LayoutRegion

        bbox = BoundingBox(x0=0.0, top=0.0, x1=1.0, bottom=0.1)
        from doc_intelligence.schemas.core import Line

        proc = DocumentProcessor(
            parser=ScannedPDFParser(
                layout_detector=FakeLayoutDetector(
                    regions=[
                        LayoutRegion(
                            bounding_box=bbox, region_type="text", confidence=0.9
                        )
                    ]
                ),
                ocr_engine=FakeOCREngine(
                    lines=[Line(text="ocr line", bounding_box=bbox)]
                ),
            ),
            formatter=PDFFormatter(),
            extractor=CapturingExtractor(
                llm=fake_llm,
                result=ExtractionResult(
                    data=SimpleExtraction(name="a", age=1), metadata=None
                ),
            ),
        )

        with patch(
            "doc_intelligence.pdf.parser._render_pdf_to_images",
            return_value=[np.zeros((100, 80, 3), dtype=np.uint8)],
        ):
            proc.extract(uri="scan.pdf", response_format=SimpleExtraction)

        assert len(captured) == 1
        doc = captured[0]
        assert doc.content is not None
        assert len(doc.content.pages) == 1
        assert isinstance(doc.content.pages[0].blocks[0], TextBlock)

    def test_existing_from_digital_pdf_unchanged(self, fake_llm: FakeLLM) -> None:
        """from_digital_pdf() still wires DigitalPDFParser correctly."""
        proc = DocumentProcessor.from_digital_pdf(llm=fake_llm)
        assert isinstance(proc.parser, DigitalPDFParser)
        assert isinstance(proc.formatter, PDFFormatter)
        assert isinstance(proc.extractor, PDFExtractor)


# ---------------------------------------------------------------------------
# PDFProcessor — document_type parameter
# ---------------------------------------------------------------------------
class TestPDFProcessorDocumentType:
    def test_default_is_digital(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(llm=fake_llm)
        assert isinstance(proc._processor.parser, DigitalPDFParser)

    def test_digital_type_uses_digital_parser(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(llm=fake_llm, document_type="digital")
        assert isinstance(proc._processor.parser, DigitalPDFParser)

    def test_scanned_type_uses_scanned_parser(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(
            llm=fake_llm,
            document_type="scanned",
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert isinstance(proc._processor.parser, ScannedPDFParser)

    def test_scanned_type_uses_pdf_formatter(self, fake_llm: FakeLLM) -> None:
        proc = PDFProcessor(
            llm=fake_llm,
            document_type="scanned",
            layout_detector=FakeLayoutDetector(),
            ocr_engine=FakeOCREngine(),
        )
        assert isinstance(proc._processor.formatter, PDFFormatter)

    def test_scanned_type_missing_detector_raises(self, fake_llm: FakeLLM) -> None:
        with pytest.raises(ValueError, match="layout_detector is required"):
            PDFProcessor(llm=fake_llm, document_type="scanned")

    def test_scanned_type_missing_engine_raises(self, fake_llm: FakeLLM) -> None:
        with pytest.raises(ValueError, match="ocr_engine is required"):
            PDFProcessor(
                llm=fake_llm,
                document_type="scanned",
                layout_detector=FakeLayoutDetector(),
            )
