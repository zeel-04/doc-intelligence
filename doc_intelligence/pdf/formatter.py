"""PDF formatter — converts parsed PDF content into LLM-consumable text.

Emits block-level markup so that LLM citations reference block indices
rather than flat line numbers.  ``ImageBlock`` and ``ChartBlock`` are
silently skipped (not yet supported for LLM consumption).
"""

from loguru import logger

from doc_intelligence.base import BaseFormatter
from doc_intelligence.pdf.schemas import PDF
from doc_intelligence.schemas.core import (
    ChartBlock,
    ContentBlock,
    Document,
    ImageBlock,
    TableBlock,
    TextBlock,
)


def _render_block_text(block: ContentBlock) -> str:
    """Render the textual body of a single block.

    Args:
        block: A ContentBlock instance.

    Returns:
        The block's text content as a string, or an empty string for
        block types that are not yet supported (image, chart).
    """
    if isinstance(block, TextBlock):
        return "\n".join(line.text for line in block.lines)
    if isinstance(block, TableBlock):
        rows_text: list[str] = []
        for row in block.rows:
            rows_text.append("| " + " | ".join(cell.text for cell in row) + " |")
        return "\n".join(rows_text)
    # ImageBlock / ChartBlock — skip
    return ""


class PDFFormatter(BaseFormatter):
    def _format_with_block_indices(self, content: PDF) -> list[str]:
        """Format pages with block-index tags for citation support."""
        paginated: list[str] = []
        if not content.pages:
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):
            block_index = 0
            parts: list[str] = []
            for block in page.blocks:
                if isinstance(block, (ImageBlock, ChartBlock)):
                    continue
                text = _render_block_text(block)
                parts.append(
                    f'<block index="{block_index}" type="{block.block_type}">'
                    f"\n{text}\n</block>"
                )
                block_index += 1
            paginated.append(
                f'<page number="{page_number}">\n' + "\n".join(parts) + "\n</page>"
            )
        return paginated

    def _format_without_block_indices(self, content: PDF) -> list[str]:
        """Format pages as plain text without citation markup."""
        paginated: list[str] = []
        if not content.pages:
            raise ValueError("PDFFormatter: format_for_llm: Document pages are not set")
        for page_number, page in enumerate(content.pages):
            lines_text = ""
            for block in page.blocks:
                if isinstance(block, (ImageBlock, ChartBlock)):
                    continue
                text = _render_block_text(block)
                if text:
                    lines_text += text + "\n"
            paginated.append(f'<page number="{page_number}">\n{lines_text}</page>')
        return paginated

    def format_document_for_llm(
        self,
        document: Document,
        **kwargs,
    ) -> str:
        raw_content = document.content
        if raw_content is None:
            raise ValueError(
                "PDFFormatter: Document content is None. "
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

        include_citations = kwargs.get("include_citations", True)
        if include_citations:
            return "\n\n".join(self._format_with_block_indices(view))
        else:
            return "\n\n".join(self._format_without_block_indices(view))
