from .base import BaseFormatter
from .schemas import Document, PydanticModel


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
        self, document: Document, include_line_numbers: bool
    ) -> str:
        content = document.content
        if include_line_numbers:
            return "\n\n".join(self._format_with_line_numbers(content))  # type: ignore
        else:
            return "\n\n".join(self._format_without_line_numbers(content))  # type: ignore
