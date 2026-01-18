from typing import Any

from loguru import logger

from .base import (
    BaseExtractor,
    BaseFormatter,
    BaseParser,
)
from .extractor import DigitalPDFExtractor
from .formatter import DigitalPDFFormatter
from .llm import BaseLLM
from .parser import DigitalPDFParser
from .schemas.core import Document, PDFDocument, PydanticModel
from .schemas.pdf.digital import (
    PageCitation,
    PageLineBboxCitation,
    PageLineCitation,
)

# Citation type mapping based on include_line_numbers flag
CITATION_TYPES_DIGITAL_PDF_MAP = {
    True: (list[PageLineCitation], list[PageLineBboxCitation]),
    False: (list[PageCitation], Any),
}


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
        # Keep citation types for external access (e.g., defining Pydantic models)
        self.citation_type, self.citation_type_with_bboxes = (
            CITATION_TYPES_DIGITAL_PDF_MAP[include_line_numbers]
        )

    @classmethod
    def from_digital_pdf(
        cls, uri: str, llm: BaseLLM, include_line_numbers: bool = True, **kwargs
    ) -> "DocumentProcessor":
        citation_type, citation_type_with_bboxes = CITATION_TYPES_DIGITAL_PDF_MAP[
            include_line_numbers
        ]
        return cls(
            parser=DigitalPDFParser(),
            formatter=DigitalPDFFormatter(),
            extractor=DigitalPDFExtractor(
                llm,
                include_line_numbers=include_line_numbers,
                citation_type=citation_type,
                citation_type_with_bboxes=citation_type_with_bboxes,
            ),
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

        logger.debug(f"Document LLM input: {self.document.llm_input}")

        return self.extractor.extract(
            document=self.document,
            model=model,
            reasoning=reasoning,
            response_format=response_format,
            llm_input=self.document.llm_input,  # type: ignore[reportUnknownReturnType]
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            openai_text=openai_text,
        )
