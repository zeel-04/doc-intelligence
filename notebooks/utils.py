"""Shared notebook utilities for doc_intelligence demos."""

from io import BytesIO
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pypdfium2 as pdfium
import requests
from PIL import ImageDraw


def show_pdf_with_bboxes(
    pdf_path: str,
    data: dict,
    out_file: str | None = None,
    bbox_color: str = "red",
    bbox_width: int = 3,
    scale: float = 2.0,
    figsize: tuple = (12, 16),
) -> None:
    """Render a PDF page with highlighted bounding boxes.

    When *out_file* is provided the figure is saved to disk; otherwise it is
    displayed inline (suitable for Jupyter notebooks).

    Args:
        pdf_path: Local file path or HTTP(S) URL to a PDF.
        data: Dict with ``page`` (int) and ``bboxes`` (list of bbox dicts).
            Each bbox dict must contain ``x0``, ``top``, ``x1``, ``bottom``
            as normalised 0-1 coordinates.
        out_file: If given, save the figure to this path instead of showing it.
        bbox_color: Outline colour for bounding boxes.
        bbox_width: Line width for bounding boxes.
        scale: Rendering scale (higher = better quality, slower).
        figsize: Matplotlib figure size.
    """
    page_idx = data["page"]
    bboxes = data["bboxes"]

    parsed = urlparse(pdf_path)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(pdf_path)
        resp.raise_for_status()
        pdf = pdfium.PdfDocument(BytesIO(resp.content))
    else:
        pdf = pdfium.PdfDocument(pdf_path)

    page_obj = pdf[page_idx]
    pil_image = page_obj.render(scale=scale).to_pil()
    width, height = pil_image.size

    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        draw.rectangle(
            [
                (bbox["x0"] * width, bbox["top"] * height),
                (bbox["x1"] * width, bbox["bottom"] * height),
            ],
            outline=bbox_color,
            width=bbox_width,
        )

    plt.figure(figsize=figsize)
    plt.imshow(pil_image)
    plt.axis("off")
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> saved to {out_file}")
    else:
        plt.show()

    page_obj.close()
    pdf.close()
