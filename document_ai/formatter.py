from pydantic import BaseModel

from .base import BaseFormatter


class PDFFormatter(BaseFormatter):
    def format_for_llm(self, content: BaseModel) -> list[str]:
        llm_formatted_pages = []
        if not content.pages:  # type: ignore
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):  # type: ignore
            lines_text = ""
            for line_number, line in enumerate(page.lines):
                line_text = f"{line_number}: {line.text}" + "\n"
                lines_text += line_text
            llm_formatted_pages.append(
                f"<page number={page_number}>\n{lines_text}</page>"
            )
        return llm_formatted_pages
