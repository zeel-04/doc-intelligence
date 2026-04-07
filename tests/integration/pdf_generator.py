"""PDF generation utilities for integration test fixtures.

Uses reportlab's Platypus flowable API so fixtures can be extended to
include tables, multi-column layouts, images, and styled text without
rewriting the generation logic.

Usage::

    gen = PDFGenerator("path/to/output.pdf")
    gen.add_paragraph("Name: John")
    gen.add_paragraph("Age: 30")
    gen.add_table([["Field", "Value"], ["Name", "John"], ["Age", "30"]])
    gen.build()
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_styles = getSampleStyleSheet()


class PDFGenerator:
    """Builds a PDF from flowable elements using reportlab Platypus.

    Args:
        output_path: Destination file path for the generated PDF.
        page_size: Page dimensions tuple. Defaults to A4.
        margins: Left/right/top/bottom margins in points. Defaults to 20mm each.
    """

    def __init__(
        self,
        output_path: str | Path,
        page_size: tuple[float, float] = A4,
        margins: float = 20 * mm,
    ) -> None:
        self._output_path = str(output_path)
        self._page_size = page_size
        self._margins = margins
        self._story: list = []

    def add_paragraph(
        self,
        text: str,
        style: str = "Normal",
        space_after: float = 0,
    ) -> "PDFGenerator":
        """Add a text paragraph.

        Args:
            text: The paragraph text.
            style: A reportlab stylesheet key (e.g. ``"Normal"``, ``"Heading1"``).
            space_after: Vertical space to add after the paragraph (in points).

        Returns:
            Self, for method chaining.
        """
        self._story.append(Paragraph(text, _styles[style]))
        if space_after:
            self._story.append(Spacer(1, space_after))
        return self

    def add_table(
        self,
        data: list[list[str]],
        col_widths: list[float] | None = None,
        header_row: bool = True,
    ) -> "PDFGenerator":
        """Add a table.

        Args:
            data: 2D list of cell values (rows × cols).
            col_widths: Optional list of column widths in points. If ``None``,
                reportlab distributes columns evenly.
            header_row: If ``True``, style the first row as a header with a
                grey background.

        Returns:
            Self, for method chaining.
        """
        table = Table(data, colWidths=col_widths)
        ts = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]
        if header_row:
            ts += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        table.setStyle(TableStyle(ts))
        self._story.append(table)
        return self

    def add_spacer(self, height: float = 6 * mm) -> "PDFGenerator":
        """Add vertical whitespace.

        Args:
            height: Space height in points. Defaults to 6mm.

        Returns:
            Self, for method chaining.
        """
        self._story.append(Spacer(1, height))
        return self

    def build(self) -> None:
        """Write all queued flowables to the output PDF file."""
        doc = SimpleDocTemplate(
            self._output_path,
            pagesize=self._page_size,
            leftMargin=self._margins,
            rightMargin=self._margins,
            topMargin=self._margins,
            bottomMargin=self._margins,
        )
        doc.build(self._story)


# ---------------------------------------------------------------------------
# Test fixture generators
# ---------------------------------------------------------------------------


def generate_simple_one_page(output_path: str | Path) -> None:
    """Generate a simple one-page PDF with 3 text lines.

    Produces a PDF that pdfplumber parses into exactly 3 Line objects:
    - ``Name: John``
    - ``Age: 30``
    - ``City: Springfield``

    Args:
        output_path: Destination file path.
    """
    (
        PDFGenerator(output_path)
        .add_paragraph("Name: John")
        .add_paragraph("Age: 30")
        .add_paragraph("City: Springfield")
        .build()
    )


if __name__ == "__main__":
    import sys

    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("tests/integration/pdfs/simple_one_page.pdf")
    )
    generate_simple_one_page(out)
    print(f"Generated: {out}")
