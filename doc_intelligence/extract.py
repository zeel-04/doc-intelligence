"""Top-level convenience function for one-liner extraction."""

from typing import Any

from .pdf.processor import PDFProcessor
from .schemas.core import ExtractionResult, PydanticModel


def extract(
    uri: str,
    response_format: type[PydanticModel],
    *,
    provider: str = "openai",
    model: str | None = None,
    include_citations: bool = True,
    extraction_mode: str = "single_pass",
    page_numbers: list[int] | None = None,
    llm_config: dict[str, Any] | None = None,
    **llm_kwargs: Any,
) -> ExtractionResult:
    """Extract structured data from a PDF in a single call.

    Creates a :class:`PDFProcessor` internally and delegates to it.
    Ideal for scripts and notebooks where minimal setup is preferred.

    Args:
        uri: Path or URL of the PDF to process.
        response_format: Pydantic model class describing the expected
            extraction schema.
        provider: LLM provider name (default ``"openai"``).
        model: Model name for the provider. If *None*, the provider's
            default is used.
        include_citations: Whether to include citation metadata.
        extraction_mode: ``"single_pass"`` or ``"multi_pass"``.
        page_numbers: Optional page restriction (0-indexed).
        llm_config: Additional LLM config dict forwarded per-call.
        **llm_kwargs: Additional provider-specific arguments forwarded
            to :func:`create_llm` (e.g. ``api_key``, ``host``).

    Returns:
        An :class:`ExtractionResult` with ``.data`` and ``.metadata``
        attributes.

    Examples:
        >>> from doc_intelligence import extract
        >>> result = extract("invoice.pdf", InvoiceSchema, provider="openai")
        >>> print(result.data)
    """
    processor = PDFProcessor(provider=provider, model=model, **llm_kwargs)
    return processor.extract(
        uri=uri,
        response_format=response_format,
        include_citations=include_citations,
        extraction_mode=extraction_mode,
        page_numbers=page_numbers,
        llm_config=llm_config,
    )
