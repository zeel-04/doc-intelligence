from typing import Any

from loguru import logger

from .base import BaseExtractor, BaseFormatter
from .llm import BaseLLM
from .pydantic_to_json_instance_schema import (
    pydantic_to_json_instance_schema,
    stringify_schema,
)
from .schemas.core import Document, PydanticModel
from .types.pdf import PDFExtractionMode
from .utils import enrich_citations_with_bboxes, strip_citations


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

    def extract(
        self,
        document: Document,
        llm_config: dict[str, Any],
        extraction_config: dict[str, Any],
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
    ) -> tuple[PydanticModel, dict[str, Any] | None]:
        if document.extraction_mode == PDFExtractionMode.SINGLE_PASS:
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
            content_text = formatter.format_document_for_llm(
                document, **extraction_config
            )
            logger.debug(f"DigitalPDFExtractor: extract: content_text: {content_text}")
            user_prompt = self.user_prompt.format(
                content_text=content_text, schema=json_instance_schema
            )

            response = self.llm.generate_text(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                **llm_config,
            )

        if document.extraction_mode == PDFExtractionMode.MULTI_PASS:
            raise NotImplementedError("Multi-pass extraction is not implemented yet")

        response_dict = self.json_parser.parse(response)

        if document.include_citations:
            response_metadata = enrich_citations_with_bboxes(response_dict, document)
            response_dict = strip_citations(response_metadata)
        else:
            response_metadata = None

        return response_format(**response_dict), response_metadata
