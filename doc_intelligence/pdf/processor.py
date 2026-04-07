"""Document processing pipeline and PDF convenience wrapper."""

from typing import Any

from loguru import logger
from pydantic import BaseModel

from doc_intelligence.base import (
    BaseExtractor,
    BaseFormatter,
    BaseParser,
)
from doc_intelligence.config import settings
from doc_intelligence.llm import BaseLLM, create_llm
from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine
from doc_intelligence.pdf.extractor import PDFExtractor
from doc_intelligence.pdf.formatter import PDFFormatter
from doc_intelligence.pdf.parser import PDFParser
from doc_intelligence.pdf.schemas import PDFExtractionRequest
from doc_intelligence.pdf.types import (
    ParseStrategy,
    PDFExtractionMode,
    ScannedPipelineType,
)
from doc_intelligence.restrictions import (
    check_page_count,
    check_pdf_size,
    check_schema_depth,
)
from doc_intelligence.schemas.core import (
    ExtractionRequest,
    ExtractionResult,
    PydanticModel,
)


class DocumentProcessor:
    """Generic document processing pipeline.

    Orchestrates parse → extract for any document type.  Knows nothing
    about PDFs — component wiring happens at construction time.

    Args:
        parser: Parses raw documents into structured content.
        formatter: Formats parsed content for LLM consumption.
        extractor: Extracts structured data using an LLM.
    """

    def __init__(
        self,
        parser: BaseParser,
        formatter: BaseFormatter,
        extractor: BaseExtractor,
    ):
        self.parser = parser
        self.formatter = formatter
        self.extractor = extractor

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract structured data from a document.

        Args:
            request: An :class:`ExtractionRequest` (or subclass) carrying
                the document URI, response schema, and extraction options.

        Returns:
            An :class:`ExtractionResult` with ``.data`` and
            ``.metadata`` attributes.
        """
        document = self.parser.parse(request.uri)
        logger.info("Document parsed successfully")
        return self.extractor.extract(document, request, self.formatter)


class PDFProcessor:
    """High-level convenience class for PDF extraction.

    Wraps :class:`DocumentProcessor` with pre-configured PDF components.
    Accepts either an existing ``llm`` instance (useful for testing or
    sharing a client) or ``provider`` + ``model`` strings (for quick
    setup via the factory).

    All pipeline-level settings are fixed at construction time.
    :meth:`extract` accepts only the document URI, the extraction
    schema, and an optional page filter — nothing else.  If you need
    different pipeline settings, create a separate processor.

    Args:
        llm: A pre-built :class:`BaseLLM` instance.
        provider: LLM provider name (e.g. ``"openai"``, ``"anthropic"``).
            Used with :func:`create_llm` when ``llm`` is not provided.
        model: Model name for the provider. If *None*, the provider's
            default is used.
        strategy: ``DIGITAL`` (default) or ``SCANNED``. Selects the
            parsing path inside :class:`PDFParser`.
        scanned_pipeline: ``VLM`` (default) or ``TWO_STAGE`` (not yet
            implemented). Selects the scanned sub-pipeline. Only used
            when ``strategy`` is ``SCANNED``.
        include_citations: Whether to include citation metadata in
            extraction results (default ``True``).
        extraction_mode: ``SINGLE_PASS`` (default) or ``MULTI_PASS``.
        llm_config: Generation parameters forwarded to the LLM on
            every call (e.g. ``{"temperature": 0.2}``).
        layout_detector: A :class:`BaseLayoutDetector` implementation.
            Required when ``scanned_pipeline`` is ``TWO_STAGE``.
        ocr_engine: A :class:`BaseOCREngine` implementation.
            Required when ``scanned_pipeline`` is ``TWO_STAGE``.
        dpi: Page rendering resolution for scanned PDFs (default 150).
        vlm_batch_size: Pages per VLM call (default 1). Only used when
            ``scanned_pipeline`` is ``VLM``.

    Raises:
        ValueError: If neither ``llm`` nor ``provider`` is specified.

    Examples:
        >>> processor = PDFProcessor(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     extraction_mode=PDFExtractionMode.MULTI_PASS,
        ... )
        >>> r1 = processor.extract("jan.pdf", Invoice)
        >>> r2 = processor.extract("feb.pdf", Invoice)
        >>> r3 = processor.extract("receipt.pdf", Receipt)
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        strategy: ParseStrategy = ParseStrategy.DIGITAL,
        scanned_pipeline: ScannedPipelineType = ScannedPipelineType.VLM,
        include_citations: bool = True,
        extraction_mode: PDFExtractionMode = PDFExtractionMode.SINGLE_PASS,
        llm_config: dict[str, Any] | None = None,
        layout_detector: BaseLayoutDetector | None = None,
        ocr_engine: BaseOCREngine | None = None,
        dpi: int = 150,
        vlm_batch_size: int = 1,
    ):
        if llm is not None:
            self._llm = llm
        elif provider is not None:
            self._llm = create_llm(provider, model)
        else:
            raise ValueError("Either `llm` or `provider` must be specified")

        self._include_citations = include_citations
        self._extraction_mode = extraction_mode
        self._llm_config = llm_config

        self._processor = DocumentProcessor(
            parser=PDFParser(
                strategy=strategy,
                scanned_pipeline=scanned_pipeline,
                layout_detector=layout_detector,
                ocr_engine=ocr_engine,
                llm=self._llm if scanned_pipeline == ScannedPipelineType.VLM else None,
                dpi=dpi,
                vlm_batch_size=vlm_batch_size,
            ),
            formatter=PDFFormatter(),
            extractor=PDFExtractor(self._llm),
        )

    def extract(
        self,
        uri: str,
        response_format: type[PydanticModel],
        *,
        page_numbers: list[int] | None = None,
    ) -> ExtractionResult:
        """Extract structured data from a PDF.

        Args:
            uri: Path or URL of the PDF to process.
            response_format: Pydantic model class describing the expected
                extraction schema.
            page_numbers: Optional page restriction (0-indexed). Defaults
                to all pages.

        Returns:
            An :class:`ExtractionResult` with ``.data`` and
            ``.metadata`` attributes.

        Raises:
            ValueError: If the response format is not a Pydantic model,
                or if size/page/depth limits are exceeded.
        """
        if not issubclass(response_format, BaseModel):
            raise ValueError("response_format must be a Pydantic model")

        # Restriction checks — before any expensive work
        check_pdf_size(uri, settings.max_pdf_size_mb)
        check_page_count(uri, settings.max_pdf_pages)
        check_schema_depth(response_format, settings.max_schema_depth)

        request = PDFExtractionRequest(
            uri=uri,
            response_format=response_format,
            include_citations=self._include_citations,
            extraction_mode=self._extraction_mode,
            page_numbers=page_numbers,
            llm_config=self._llm_config,
        )

        return self._processor.extract(request)
