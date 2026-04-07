"""PDF extraction — single-pass and multi-pass modes."""

from typing import Any

from loguru import logger

from doc_intelligence.base import BaseExtractor, BaseFormatter
from doc_intelligence.llm import BaseLLM
from doc_intelligence.pdf.schemas import PDFDocument, PDFExtractionRequest
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.pdf.utils import enrich_citations_with_bboxes
from doc_intelligence.pydantic_to_json_instance_schema import (
    pydantic_to_json_instance_schema,
    stringify_schema,
)
from doc_intelligence.schemas.core import (
    Document,
    ExtractionRequest,
    ExtractionResult,
    PydanticModel,
)
from doc_intelligence.utils import strip_citations

_PASS2_SYSTEM_PROMPT = "Act as an expert in document analysis and page localisation."
_PASS2_USER_PROMPT = """\
Given the document below and the previously extracted answer, identify which \
page(s) contain the supporting text for each field.

DOCUMENT:
{content_text}

EXTRACTED ANSWER:
{pass1_json}

Respond as JSON mapping each field path to a list of 0-indexed page numbers:
{{"field_path": [page_numbers]}}
"""

_PASS3_USER_PROMPT = """\
Your job is to locate the exact block indices for each field in the pages below.
Use the citation-aware schema to mark where each value appears.

PREVIOUSLY EXTRACTED ANSWER:
{pass1_json}

DOCUMENT (relevant pages only):
{content_text}

OUTPUT SCHEMA:
{schema}

Generate output in JSON format.
"""


class PDFExtractor(BaseExtractor):
    def __init__(self, llm: BaseLLM):
        super().__init__(llm)

        self.system_prompt = """Act as an expert in the field of document extraction and information extraction from documents."""
        self.user_prompt = """Your job is to extract structured mentioned in schema data from a document given below.

DOCUMENT:
{content_text}

OUTPUT SCHEMA:
{schema}

Generate output in JSON format.
"""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        document: Document,
        request: ExtractionRequest,
        formatter: BaseFormatter,
    ) -> ExtractionResult:
        pdf_request = _as_pdf_request(request)
        llm_config = pdf_request.llm_config or {}

        if pdf_request.extraction_mode == PDFExtractionMode.SINGLE_PASS:
            return self._run_single_pass(document, pdf_request, formatter, llm_config)
        elif pdf_request.extraction_mode == PDFExtractionMode.MULTI_PASS:
            return self._run_multi_pass(document, pdf_request, formatter, llm_config)
        else:
            raise ValueError(
                f"Unsupported extraction mode: {pdf_request.extraction_mode}"
            )

    # ------------------------------------------------------------------
    # Single-pass (existing behaviour, extracted to a private method)
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        document: Document,
        request: PDFExtractionRequest,
        formatter: BaseFormatter,
        llm_config: dict[str, Any],
    ) -> ExtractionResult:
        json_instance_schema = stringify_schema(
            pydantic_to_json_instance_schema(
                request.response_format,
                citation=request.include_citations,
                citation_level="block",
            )
        )
        logger.debug(
            f"PDFExtractor: extract: json_instance_schema: {json_instance_schema}"
        )
        content_text = formatter.format_document_for_llm(
            document,
            page_numbers=request.page_numbers,
            include_citations=request.include_citations,
        )
        logger.debug(f"PDFExtractor: extract: content_text: {content_text}")
        user_prompt = self.user_prompt.format(
            content_text=content_text, schema=json_instance_schema
        )

        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            **llm_config,
        )

        response_dict = self.json_parser.parse(response)

        if request.include_citations:
            response_metadata = enrich_citations_with_bboxes(response_dict, document)
            response_dict = strip_citations(response_metadata)
        else:
            response_metadata = None

        return ExtractionResult(
            data=request.response_format(**response_dict),
            metadata=response_metadata,
        )

    # ------------------------------------------------------------------
    # Multi-pass orchestration
    # ------------------------------------------------------------------

    def _run_multi_pass(
        self,
        document: Document,
        request: PDFExtractionRequest,
        formatter: BaseFormatter,
        llm_config: dict[str, Any],
    ) -> ExtractionResult:
        # Pass 1 — raw extraction (no citations)
        pass1_result = self._extract_pass1(document, request, formatter, llm_config)
        logger.debug(f"PDFExtractor: multi-pass: pass1 complete: {pass1_result}")

        if not request.include_citations:
            return ExtractionResult(data=pass1_result, metadata=None)

        # Pass 2 — page grounding
        page_map = self._extract_pass2(
            document, request, formatter, pass1_result, llm_config
        )
        logger.debug(f"PDFExtractor: multi-pass: pass2 page_map: {page_map}")

        # Pass 3 — block grounding on relevant pages only
        metadata = self._extract_pass3(
            document,
            request,
            formatter,
            pass1_result,
            page_map,
            llm_config,
        )
        logger.debug("PDFExtractor: multi-pass: pass3 complete")

        return ExtractionResult(data=pass1_result, metadata=metadata)

    # ------------------------------------------------------------------
    # Pass 1 — raw extraction without citations
    # ------------------------------------------------------------------

    def _extract_pass1(
        self,
        document: Document,
        request: PDFExtractionRequest,
        formatter: BaseFormatter,
        llm_config: dict[str, Any],
    ) -> PydanticModel:
        """Raw extraction — schema has no citation wrappers."""
        json_instance_schema = stringify_schema(
            pydantic_to_json_instance_schema(request.response_format, citation=False)
        )
        content_text = formatter.format_document_for_llm(
            document,
            page_numbers=request.page_numbers,
            include_citations=False,
        )

        user_prompt = self.user_prompt.format(
            content_text=content_text, schema=json_instance_schema
        )
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            **llm_config,
        )
        response_dict = self.json_parser.parse(response)
        return request.response_format(**response_dict)

    # ------------------------------------------------------------------
    # Pass 2 — page grounding
    # ------------------------------------------------------------------

    def _extract_pass2(
        self,
        document: Document,
        request: PDFExtractionRequest,
        formatter: BaseFormatter,
        pass1_result: PydanticModel,
        llm_config: dict[str, Any],
    ) -> dict[str, list[int]]:
        """Ask the LLM which pages each field appears on."""
        content_text = formatter.format_document_for_llm(
            document,
            page_numbers=request.page_numbers,
            include_citations=request.include_citations,
        )
        user_prompt = _PASS2_USER_PROMPT.format(
            content_text=content_text,
            pass1_json=pass1_result.model_dump_json(),
        )
        response = self.llm.generate(
            system_prompt=_PASS2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **llm_config,
        )
        page_map: dict[str, list[int]] = self.json_parser.parse(response)
        return page_map

    # ------------------------------------------------------------------
    # Pass 3 — block grounding on relevant pages
    # ------------------------------------------------------------------

    def _extract_pass3(
        self,
        document: Document,
        request: PDFExtractionRequest,
        formatter: BaseFormatter,
        pass1_result: PydanticModel,
        page_mapping: dict[str, list[int]],
        llm_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Block-level grounding restricted to the pages identified in Pass 2."""
        all_pages = sorted({p for pages in page_mapping.values() for p in pages})
        if request.page_numbers:
            intersected = [p for p in all_pages if p in request.page_numbers]
            if intersected:
                all_pages = intersected

        json_instance_schema = stringify_schema(
            pydantic_to_json_instance_schema(
                request.response_format, citation=True, citation_level="block"
            )
        )
        content_text = formatter.format_document_for_llm(
            document,
            page_numbers=all_pages,
            include_citations=request.include_citations,
        )
        user_prompt = _PASS3_USER_PROMPT.format(
            pass1_json=pass1_result.model_dump_json(),
            content_text=content_text,
            schema=json_instance_schema,
        )
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            **llm_config,
        )
        response_dict = self.json_parser.parse(response)
        return enrich_citations_with_bboxes(response_dict, document)


def _as_pdf_request(request: ExtractionRequest) -> PDFExtractionRequest:
    """Narrow an ExtractionRequest to PDFExtractionRequest.

    If the request is already a PDFExtractionRequest, return it directly.
    Otherwise wrap it with PDF defaults.
    """
    if isinstance(request, PDFExtractionRequest):
        return request
    return PDFExtractionRequest(
        uri=request.uri,
        response_format=request.response_format,
        include_citations=request.include_citations,
        llm_config=request.llm_config,
    )
