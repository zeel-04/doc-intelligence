from typing import Any

from loguru import logger

from document_ai.base import (
    BaseExtractor,
    BaseFormatter,
    BaseParser,
)
from document_ai.extractor import DigitalPDFExtractor
from document_ai.formatter import DigitalPDFFormatter
from document_ai.llm import BaseLLM
from document_ai.parser import DigitalPDFParser
from document_ai.schemas import Document, PDFDocument, PydanticModel
from document_ai.types import CitationType


class DocumentProcessor:
    def __init__(
        self,
        parser: BaseParser,
        formatter: BaseFormatter,
        extractor: BaseExtractor,
        document: Document,
        include_line_numbers: bool = True,
    ):
        self.parser = parser
        self.formatter = formatter
        self.extractor = extractor
        self.document = document
        self.include_line_numbers = include_line_numbers
        self.citation_type = CitationType if include_line_numbers else Any

    @classmethod
    def from_digital_pdf(
        cls, uri: str, llm: BaseLLM, include_line_numbers: bool = True, **kwargs
    ) -> "DocumentProcessor":
        """Create processor for Digital PDF documents"""
        return cls(
            parser=DigitalPDFParser(),
            formatter=DigitalPDFFormatter(),
            extractor=DigitalPDFExtractor(llm),
            document=PDFDocument(uri=uri),
            include_line_numbers=include_line_numbers,
            **kwargs,
        )

    def parse(self) -> Document:
        self.document.content = self.parser.parse(self.document)
        logger.info("Document parsed successfully")
        return self.document

    def format_document_for_llm(self, page_numbers: list[int] | None = None) -> str:
        if not self.document.content:
            raise ValueError("Please parse the document first")
        self.document.llm_input = self.formatter.format_document_for_llm(
            self.document,
            self.include_line_numbers,
            page_numbers=page_numbers,
        )
        logger.info("Document formatted successfully")
        return self.document.llm_input

    def extract(
        self,
        model: str,
        reasoning: Any,
        response_format: type[PydanticModel],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        openai_text: dict[str, Any] | None = None,
        page_numbers: list[int] | None = None,
    ) -> Any:
        # Auto-parse and format if not done
        if not self.document.content:
            self.parse()
        if not self.document.llm_input:
            self.format_document_for_llm(page_numbers=page_numbers)

        return self.extractor.extract(
            document=self.document,
            model=model,
            reasoning=reasoning,
            response_format=response_format,
            include_line_numbers=self.include_line_numbers,
            llm_input=self.document.llm_input,  # type: ignore[reportUnknownReturnType]
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            openai_text=openai_text,
        )
