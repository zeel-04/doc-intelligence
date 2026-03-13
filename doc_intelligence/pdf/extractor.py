"""Digital PDF extraction — single-pass and multi-pass modes."""

from typing import Any

from loguru import logger

from doc_intelligence.base import BaseExtractor, BaseFormatter
from doc_intelligence.llm import BaseLLM
from doc_intelligence.pdf.schemas import PDFDocument
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.pdf.utils import enrich_citations_with_bboxes
from doc_intelligence.pydantic_to_json_instance_schema import (
    pydantic_to_json_instance_schema,
    stringify_schema,
)
from doc_intelligence.schemas.core import Document, ExtractionResult, PydanticModel
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
Your job is to locate the exact line numbers for each field in the pages below.
Use the citation-aware schema to mark where each value appears.

PREVIOUSLY EXTRACTED ANSWER:
{pass1_json}

DOCUMENT (relevant pages only):
{content_text}

OUTPUT SCHEMA:
{schema}

Generate output in JSON format.
"""


class DigitalPDFExtractor(BaseExtractor):
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
        llm_config: dict[str, Any],
        extraction_config: dict[str, Any],
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
    ) -> ExtractionResult:
        if document.extraction_mode == PDFExtractionMode.SINGLE_PASS:
            return self._run_single_pass(
                document, formatter, response_format, llm_config
            )
        elif document.extraction_mode == PDFExtractionMode.MULTI_PASS:
            assert isinstance(document, PDFDocument)
            return self._run_multi_pass(
                document, formatter, response_format, llm_config
            )
        else:
            raise ValueError(f"Unsupported extraction mode: {document.extraction_mode}")

    # ------------------------------------------------------------------
    # Single-pass (existing behaviour, extracted to a private method)
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        document: Document,
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
        llm_config: dict[str, Any],
    ) -> ExtractionResult:
        json_instance_schema = stringify_schema(
            pydantic_to_json_instance_schema(
                response_format,
                citation=document.include_citations,
                citation_level="line",
            )
        )
        logger.debug(
            f"DigitalPDFExtractor: extract: json_instance_schema: {json_instance_schema}"
        )
        content_text = formatter.format_document_for_llm(document)
        logger.debug(f"DigitalPDFExtractor: extract: content_text: {content_text}")
        user_prompt = self.user_prompt.format(
            content_text=content_text, schema=json_instance_schema
        )

        response = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            **llm_config,
        )

        response_dict = self.json_parser.parse(response)

        if document.include_citations:
            response_metadata = enrich_citations_with_bboxes(response_dict, document)
            response_dict = strip_citations(response_metadata)
        else:
            response_metadata = None

        return ExtractionResult(
            data=response_format(**response_dict),
            metadata=response_metadata,
        )

    # ------------------------------------------------------------------
    # Multi-pass orchestration
    # ------------------------------------------------------------------

    def _run_multi_pass(
        self,
        document: PDFDocument,
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
        llm_config: dict[str, Any],
    ) -> ExtractionResult:
        # Pass 1 — raw extraction (no citations)
        pass1_result = self._extract_pass1(
            document, formatter, response_format, llm_config
        )
        document.pass1_result = pass1_result
        logger.debug(f"DigitalPDFExtractor: multi-pass: pass1 complete: {pass1_result}")

        if not document.include_citations:
            return ExtractionResult(data=pass1_result, metadata=None)

        # Pass 2 — page grounding
        page_map = self._extract_pass2(document, formatter, pass1_result, llm_config)
        document.pass2_page_map = page_map
        logger.debug(f"DigitalPDFExtractor: multi-pass: pass2 page_map: {page_map}")

        # Pass 3 — line grounding on relevant pages only
        metadata = self._extract_pass3(
            document, formatter, pass1_result, page_map, response_format, llm_config
        )
        logger.debug("DigitalPDFExtractor: multi-pass: pass3 complete")

        return ExtractionResult(data=pass1_result, metadata=metadata)

    # ------------------------------------------------------------------
    # Pass 1 — raw extraction without citations
    # ------------------------------------------------------------------

    def _extract_pass1(
        self,
        document: PDFDocument,
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
        llm_config: dict[str, Any],
    ) -> PydanticModel:
        """Raw extraction — schema has no citation wrappers."""
        json_instance_schema = stringify_schema(
            pydantic_to_json_instance_schema(response_format, citation=False)
        )
        original_citations = document.include_citations
        document.include_citations = False
        content_text = formatter.format_document_for_llm(document)
        document.include_citations = original_citations

        user_prompt = self.user_prompt.format(
            content_text=content_text, schema=json_instance_schema
        )
        response = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            **llm_config,
        )
        response_dict = self.json_parser.parse(response)
        return response_format(**response_dict)

    # ------------------------------------------------------------------
    # Pass 2 — page grounding
    # ------------------------------------------------------------------

    def _extract_pass2(
        self,
        document: PDFDocument,
        formatter: BaseFormatter,
        pass1_result: PydanticModel,
        llm_config: dict[str, Any],
    ) -> dict[str, list[int]]:
        """Ask the LLM which pages each field appears on."""
        content_text = formatter.format_document_for_llm(document)
        user_prompt = _PASS2_USER_PROMPT.format(
            content_text=content_text,
            pass1_json=pass1_result.model_dump_json(),
        )
        response = self.llm.generate_text(
            system_prompt=_PASS2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            **llm_config,
        )
        page_map: dict[str, list[int]] = self.json_parser.parse(response)
        return page_map

    # ------------------------------------------------------------------
    # Pass 3 — line grounding on relevant pages
    # ------------------------------------------------------------------

    def _extract_pass3(
        self,
        document: PDFDocument,
        formatter: BaseFormatter,
        pass1_result: PydanticModel,
        page_mapping: dict[str, list[int]],
        response_format: type[PydanticModel],
        llm_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Line-level grounding restricted to the pages identified in Pass 2."""
        all_pages = sorted({p for pages in page_mapping.values() for p in pages})

        json_instance_schema = stringify_schema(
            pydantic_to_json_instance_schema(
                response_format, citation=True, citation_level="line"
            )
        )
        content_text = formatter.format_document_for_llm(
            document, page_numbers=all_pages
        )
        user_prompt = _PASS3_USER_PROMPT.format(
            pass1_json=pass1_result.model_dump_json(),
            content_text=content_text,
            schema=json_instance_schema,
        )
        response = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            **llm_config,
        )
        response_dict = self.json_parser.parse(response)
        return enrich_citations_with_bboxes(response_dict, document)
