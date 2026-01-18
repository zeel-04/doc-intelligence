from loguru import logger

from .base import BaseFormatter
from .schemas.core import Document, PydanticModel


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
        include_line_numbers: bool,
        page_numbers: list[int] | None = None,
    ) -> str:
        content = document.content
        if page_numbers and content.pages:  # type: ignore
            page_numbers.sort()
            page_numbers = list(set(page_numbers))
            content.pages = [  # type: ignore
                page
                for page_number, page in enumerate(content.pages)  # type: ignore
                if page_number in page_numbers
            ]
            logger.info(f"Formatting {len(content.pages)} pages")  # type: ignore
        if include_line_numbers:
            return "\n\n".join(self._format_with_line_numbers(content))  # type: ignore
        else:
            return "\n\n".join(self._format_without_line_numbers(content))  # type: ignore
