from loguru import logger

from doc_intelligence.base import BaseFormatter
from doc_intelligence.pdf.schemas import PDF, blocks_to_lines
from doc_intelligence.schemas.core import Document


class DigitalPDFFormatter(BaseFormatter):
    def _format_with_line_numbers(self, content: PDF) -> list[str]:
        paginated = []
        if not content.pages:
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):
            lines_text = ""
            for line_number, line in enumerate(blocks_to_lines(page.blocks)):
                line_text = f"{line_number}: {line.text}" + "\n"
                lines_text += line_text
            paginated.append(f"<page number={page_number}>\n{lines_text}</page>")
        return paginated

    def _format_without_line_numbers(self, content: PDF) -> list[str]:
        paginated = []
        if not content.pages:
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):
            lines_text = ""
            for line in blocks_to_lines(page.blocks):
                line_text = f"{line.text}" + "\n"
                lines_text += line_text
            paginated.append(f"<page number={page_number}>\n{lines_text}</page>")
        return paginated

    def format_document_for_llm(
        self,
        document: Document,
        **kwargs,
    ) -> str:
        raw_content = document.content
        if raw_content is None:
            raise ValueError(
                "DigitalPDFFormatter: Document content is None. "
                "Make sure to parse the document before formatting."
            )
        pdf_content: PDF = raw_content  # type: ignore[assignment]

        # Build a filtered page view without mutating the original document
        page_numbers = kwargs.get("page_numbers", None)
        if page_numbers and pdf_content.pages:
            unique_sorted = sorted(set(page_numbers))
            pages_to_format = [
                page for i, page in enumerate(pdf_content.pages) if i in unique_sorted
            ]
            logger.info(f"Formatting {len(pages_to_format)} pages")
        else:
            pages_to_format = pdf_content.pages

        view = PDF(pages=pages_to_format)

        if document.include_citations:
            return "\n\n".join(self._format_with_line_numbers(view))
        else:
            return "\n\n".join(self._format_without_line_numbers(view))
