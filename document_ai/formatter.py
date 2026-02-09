from loguru import logger

from .base import BaseFormatter
from .schemas.core import Document, PydanticModel
from .types.pdf import PDFExtractionMode


class DigitalPDFFormatter(BaseFormatter):
    def _format_with_line_numbers(self, content: type[PydanticModel]) -> list[str]:
        paginated = []
        if not content.pages:  # type: ignore
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):  # type: ignore
            lines_text = ""
            for line_number, line in enumerate(page.lines):
                line_text = f"{line_number}: {line.text}" + "\n"
                lines_text += line_text
            paginated.append(f"<page number={page_number}>\n{lines_text}</page>")
        return paginated

    def _format_without_line_numbers(self, content: type[PydanticModel]) -> list[str]:
        paginated = []
        if not content.pages:  # type: ignore
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):  # type: ignore
            lines_text = ""
            for _, line in enumerate(page.lines):
                line_text = f"{line.text}" + "\n"
                lines_text += line_text
            paginated.append(f"<page number={page_number}>\n{lines_text}</page>")
        return paginated

    def format_document_for_llm(
        self,
        document: Document,
        **kwargs,
    ) -> str:
        content = document.content
        if content is None:
            raise ValueError(
                "DigitalPDFFormatter: Document content is None. "
                "Make sure to parse the document before formatting."
            )
        page_numbers = kwargs.get("page_numbers", None)
        if page_numbers and content.pages:
            page_numbers.sort()
            page_numbers = list(set(page_numbers))
            content.pages = [
                page
                for page_number, page in enumerate(content.pages)
                if page_number in page_numbers
            ]
            logger.info(f"Formatting {len(content.pages)} pages")
        if document.extraction_mode == PDFExtractionMode.MULTI_PASS:
            raise NotImplementedError("Multi-pass extraction is not implemented yet")
        if (
            document.include_citations
            and document.extraction_mode == PDFExtractionMode.SINGLE_PASS
        ):
            return "\n\n".join(self._format_with_line_numbers(content))
        else:
            return "\n\n".join(self._format_without_line_numbers(content))
