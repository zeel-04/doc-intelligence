from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

from .base import (
    BaseExtractor,
    BaseFormatter,
    BaseParser,
)
from .extractor import DigitalPDFExtractor
from .formatter import DigitalPDFFormatter
from .llm import BaseLLM
from .parser import DigitalPDFParser
from .schemas.core import Document
from .schemas.pdf import PDF, PDFDocument, PDFExtractionConfig
from .types.pdf import PDFExtractionMode


class DocumentProcessor:
    def __init__(
        self,
        parser: BaseParser,
        formatter: BaseFormatter,
        extractor: BaseExtractor,
        document: Document,
    ):
        self.parser = parser
        self.formatter = formatter
        self.extractor = extractor
        self.document = document

    @classmethod
    def from_digital_pdf(cls, uri: str, llm: BaseLLM, **kwargs) -> "DocumentProcessor":
        return cls(
            parser=DigitalPDFParser(),
            formatter=DigitalPDFFormatter(),
            extractor=DigitalPDFExtractor(
                llm,
            ),
            document=PDFDocument(uri=uri),
            **kwargs,
        )

    def parse(self) -> Document:
        self.document.content = self.parser.parse(self.document).content
        logger.info("Document parsed successfully")
        return self.document

    def extract(
        self,
        config: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Extract information from the document using the LLM.
        {
            "response_format": Pydantic Model,
            "llm_config": {
                **kwargs,
            },
            "extraction_config": {
                "include_citations": True,
                "extraction_mode": "single_pass",
                "page_numbers": [0, 1], # optional
            }
        }
        """
        # Config Vadidation
        for key in config:
            if key not in ["llm_config", "extraction_config", "response_format"]:
                raise ValueError(f"Invalid key: {key}")

        response_format = config.get("response_format", None)
        if response_format is None:
            raise ValueError("Please provide a Pydantic model for the response format")
        if not issubclass(response_format, BaseModel):
            raise ValueError("Response format must be a Pydantic model")

        llm_config = config.get("llm_config", {})

        extraction_config = config.get("extraction_config", {})
        if extraction_config:
            include_citations = extraction_config.get(
                "include_citations", self.document.include_citations
            )
            self.document.include_citations = include_citations
            extraction_mode = extraction_config.get(
                "extraction_mode", self.document.extraction_mode.value
            )
            self.document.extraction_mode = PDFExtractionMode(extraction_mode)
            page_numbers = extraction_config.get("page_numbers", None)
            extraction_mode = PDFExtractionMode(extraction_mode)
            extraction_config = PDFExtractionConfig(
                include_citations=include_citations,
                extraction_mode=extraction_mode,
                page_numbers=page_numbers,
            ).model_dump()

        # Auto-parse and format if not done
        if not self.document.content:
            self.parse()

        return self.extractor.extract(
            document=self.document,
            llm_config=llm_config,
            extraction_config=extraction_config,
            formatter=self.formatter,
            response_format=response_format,
        )
