from typing import Any, Literal

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
from doc_intelligence.pdf.extractor import DigitalPDFExtractor
from doc_intelligence.pdf.formatter import DigitalPDFFormatter
from doc_intelligence.pdf.parser import DigitalPDFParser, ScannedPDFParser
from doc_intelligence.pdf.schemas import PDFDocument, PDFExtractionConfig
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.restrictions import (
    check_page_count,
    check_pdf_size,
    check_schema_depth,
)
from doc_intelligence.schemas.core import ExtractionResult, PydanticModel


class DocumentProcessor:
    """Reusable document processing pipeline.

    The processor is decoupled from any specific document — components
    (parser, formatter, extractor) are set at creation time, while the
    document URI and extraction options are provided per-call via
    :meth:`extract`.

    Args:
        parser: Parses raw documents into structured content.
        formatter: Formats parsed content for LLM consumption.
        extractor: Extracts structured data using an LLM.
    """

    def __init__(
        self,
        parser: BaseParser[PDFDocument],
        formatter: BaseFormatter,
        extractor: BaseExtractor,
    ):
        self.parser = parser
        self.formatter = formatter
        self.extractor = extractor

    @classmethod
    def from_digital_pdf(cls, llm: BaseLLM, **kwargs: Any) -> "DocumentProcessor":
        """Create a processor pre-configured for digital PDF extraction.

        Args:
            llm: The LLM backend to use for extraction.
            **kwargs: Additional arguments forwarded to the constructor.

        Returns:
            A configured :class:`DocumentProcessor` instance.
        """
        return cls(
            parser=DigitalPDFParser(),
            formatter=DigitalPDFFormatter(),
            extractor=DigitalPDFExtractor(llm),
            **kwargs,
        )

    @classmethod
    def from_scanned_pdf(
        cls,
        llm: BaseLLM,
        layout_detector: BaseLayoutDetector | None = None,
        ocr_engine: BaseOCREngine | None = None,
        dpi: int = 150,
        **kwargs: Any,
    ) -> "DocumentProcessor":
        """Create a processor pre-configured for scanned PDF extraction.

        Uses :class:`~doc_intelligence.pdf.parser.ScannedPDFParser` for
        parsing, reusing :class:`DigitalPDFFormatter` and
        :class:`DigitalPDFExtractor` for the rest of the pipeline — no new
        formatter or extractor is needed.

        Args:
            llm: The LLM backend to use for extraction.
            layout_detector: A :class:`BaseLayoutDetector` implementation.
                Required — pass your own detector instance.
            ocr_engine: A :class:`BaseOCREngine` implementation.
                Required — pass your own OCR engine instance.
            dpi: Page rendering resolution in dots per inch (default 150).
            **kwargs: Additional arguments forwarded to the constructor.

        Returns:
            A configured :class:`DocumentProcessor` instance.

        Raises:
            ValueError: If ``layout_detector`` or ``ocr_engine`` is not
                provided.
        """
        if layout_detector is None:
            raise ValueError(
                "layout_detector is required — supply your own "
                "BaseLayoutDetector implementation."
            )
        if ocr_engine is None:
            raise ValueError(
                "ocr_engine is required — supply your own BaseOCREngine implementation."
            )
        return cls(
            parser=ScannedPDFParser(
                layout_detector=layout_detector,
                ocr_engine=ocr_engine,
                dpi=dpi,
            ),
            formatter=DigitalPDFFormatter(),
            extractor=DigitalPDFExtractor(llm),
            **kwargs,
        )

    def _parse(self, document: PDFDocument) -> PDFDocument:
        """Parse the document if it has no content yet.

        Args:
            document: The document to parse.

        Returns:
            The same document with ``content`` populated.
        """
        document.content = self.parser.parse(document).content
        logger.info("Document parsed successfully")
        return document

    def extract(
        self,
        uri: str,
        response_format: type[PydanticModel],
        *,
        include_citations: bool = True,
        extraction_mode: str = "single_pass",
        page_numbers: list[int] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract structured data from a document.

        A fresh :class:`PDFDocument` is created for each call, so the
        processor can be reused across multiple files.

        Args:
            uri: Path or URL of the PDF to process.
            response_format: Pydantic model class describing the
                expected extraction schema.
            include_citations: Whether to include citation metadata
                in the result.
            extraction_mode: ``"single_pass"`` or ``"multi_pass"``.
            page_numbers: Optional list of 0-indexed page numbers to
                restrict extraction to.
            llm_config: Additional LLM configuration (e.g.
                ``{"model": "gpt-4o", "temperature": 0.2}``).

        Returns:
            An :class:`ExtractionResult` with ``.data`` and
            ``.metadata`` attributes.

        Raises:
            ValueError: If the response format is not a Pydantic model,
                or if size/page/depth limits are exceeded.
        """
        if not issubclass(response_format, BaseModel):
            raise ValueError("Response format must be a Pydantic model")

        # Build a fresh document for this call
        mode = PDFExtractionMode(extraction_mode)
        document = PDFDocument(
            uri=uri,
            include_citations=include_citations,
            extraction_mode=mode,
            page_numbers=page_numbers,
        )

        # Restriction checks — before any expensive work
        check_pdf_size(document.uri, settings.max_pdf_size_mb)
        check_schema_depth(response_format, settings.max_schema_depth)

        # Build extraction config
        extraction_config = PDFExtractionConfig(
            include_citations=include_citations,
            extraction_mode=mode,
            page_numbers=page_numbers,
        ).model_dump()

        # Auto-parse if not done
        if not document.content:
            self._parse(document)

        # Page count check — requires parsed content
        if document.content and hasattr(document.content, "pages"):
            check_page_count(len(document.content.pages), settings.max_pdf_pages)

        return self.extractor.extract(
            document=document,
            llm_config=llm_config or {},
            extraction_config=extraction_config,
            formatter=self.formatter,
            response_format=response_format,
        )


class PDFProcessor:
    """High-level convenience class for PDF extraction.

    Wraps :class:`DocumentProcessor` with pre-configured PDF components.
    Accepts either an existing ``llm`` instance (useful for testing or
    sharing a client) or ``provider`` + ``model`` strings (for quick
    setup via the factory).

    Args:
        llm: A pre-built :class:`BaseLLM` instance.
        provider: LLM provider name (e.g. ``"openai"``, ``"anthropic"``).
            Used with :func:`create_llm` when ``llm`` is not provided.
        model: Model name for the provider. If *None*, the provider's
            default is used.
        document_type: ``"digital"`` (default) or ``"scanned"``. Selects
            the underlying parser — :class:`DigitalPDFParser` or
            :class:`~doc_intelligence.pdf.parser.ScannedPDFParser`.
        layout_detector: A :class:`BaseLayoutDetector` implementation.
            Required when ``document_type="scanned"``.
        ocr_engine: A :class:`BaseOCREngine` implementation.
            Required when ``document_type="scanned"``.
        **llm_kwargs: Additional keyword arguments forwarded to
            :func:`create_llm` (e.g. ``api_key``, ``host``).

    Raises:
        ValueError: If neither ``llm`` nor ``provider`` is specified.

    Examples:
        >>> processor = PDFProcessor(provider="openai", model="gpt-4o")
        >>> result = processor.extract("invoice.pdf", InvoiceSchema)

        >>> llm = OpenAILLM(model="gpt-4o")
        >>> processor = PDFProcessor(llm=llm)
        >>> result = processor.extract("invoice.pdf", InvoiceSchema)
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        document_type: Literal["digital", "scanned"] = "digital",
        layout_detector: BaseLayoutDetector | None = None,
        ocr_engine: BaseOCREngine | None = None,
        **llm_kwargs: Any,
    ):
        if llm is not None:
            self._llm = llm
        elif provider is not None:
            self._llm = create_llm(provider, model, **llm_kwargs)
        else:
            raise ValueError("Either `llm` or `provider` must be specified")
        if document_type == "scanned":
            self._processor = DocumentProcessor.from_scanned_pdf(
                llm=self._llm,
                layout_detector=layout_detector,
                ocr_engine=ocr_engine,
            )
        else:
            self._processor = DocumentProcessor.from_digital_pdf(llm=self._llm)

    def extract(
        self,
        uri: str,
        response_format: type[PydanticModel],
        *,
        model: str | None = None,
        include_citations: bool = True,
        extraction_mode: str = "single_pass",
        page_numbers: list[int] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract structured data from a PDF.

        Args:
            uri: Path or URL of the PDF to process.
            response_format: Pydantic model class describing the
                expected extraction schema.
            model: Optional per-call model override (transient — does
                not mutate the processor's default model).
            include_citations: Whether to include citation metadata.
            extraction_mode: ``"single_pass"`` or ``"multi_pass"``.
            page_numbers: Optional page restriction (0-indexed).
            llm_config: Additional LLM config dict. If ``model`` is
                also provided, it takes precedence.

        Returns:
            An :class:`ExtractionResult` with ``.data`` and
            ``.metadata`` attributes.
        """
        effective_llm_config = dict(llm_config or {})
        if model is not None:
            effective_llm_config["model"] = model
        return self._processor.extract(
            uri=uri,
            response_format=response_format,
            include_citations=include_citations,
            extraction_mode=extraction_mode,
            page_numbers=page_numbers,
            llm_config=effective_llm_config if effective_llm_config else None,
        )
