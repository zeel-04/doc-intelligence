"""Shared notebook utilities for doc_intelligence demos."""

from io import BytesIO
from urllib.parse import urlparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
import requests
from PIL import ImageDraw


def _open_pdf(pdf_path: str) -> pdfium.PdfDocument:
    """Open a PDF from a local path or URL."""
    parsed = urlparse(pdf_path)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(pdf_path)
        resp.raise_for_status()
        return pdfium.PdfDocument(BytesIO(resp.content))
    return pdfium.PdfDocument(pdf_path)


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

    pdf = _open_pdf(pdf_path)
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


# Default color palette for multi-field visualization
FIELD_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
]


def show_all_fields(
    pdf_path: str,
    metadata: dict,
    out_dir: str | None = None,
    bbox_width: int = 3,
    scale: float = 2.0,
    figsize: tuple = (14, 18),
    fill_opacity: int = 40,
) -> None:
    """Render PDF pages with color-coded bounding boxes for every extracted field.

    Each field gets a distinct color. A legend is drawn on each page showing
    which color maps to which field. Pages without any citations are skipped.

    Args:
        pdf_path: Local file path or HTTP(S) URL to a PDF.
        metadata: The ``result.metadata`` dict from an extraction with
            ``include_citations=True``.  Structure::

                {
                    "field_name": {
                        "value": ...,
                        "citations": [{"page": int, "bboxes": [...]}]
                    }
                }
        out_dir: Directory to save per-page images.  Files are named
            ``page_<N>.png``.  If *None*, images are displayed inline.
        bbox_width: Outline width for bounding boxes.
        scale: Rendering scale (higher = better quality).
        figsize: Matplotlib figure size.
        fill_opacity: 0-255 opacity for the translucent bbox fill.
    """
    if metadata is None:
        print("No metadata (citations were not enabled).")
        return

    # Collect per-page drawing instructions: {page_idx: [(field, color, bbox), ...]}
    page_draws: dict[int, list[tuple[str, str, dict]]] = {}
    field_color_map: dict[str, str] = {}

    color_idx = 0
    for field_name, field_data in metadata.items():
        if not isinstance(field_data, dict) or "citations" not in field_data:
            continue
        color = FIELD_COLORS[color_idx % len(FIELD_COLORS)]
        field_color_map[field_name] = color
        color_idx += 1

        for citation in field_data["citations"]:
            page_idx = citation["page"]
            for bbox in citation.get("bboxes", []):
                page_draws.setdefault(page_idx, []).append(
                    (field_name, color, bbox)
                )

    if not page_draws:
        print("No bounding boxes found in metadata.")
        return

    pdf = _open_pdf(pdf_path)

    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)

    for page_idx in sorted(page_draws):
        page_obj = pdf[page_idx]
        pil_image = page_obj.render(scale=scale).to_pil().convert("RGBA")
        width, height = pil_image.size

        # Translucent overlay for filled rectangles
        overlay = pil_image.copy()
        draw = ImageDraw.Draw(overlay)

        fields_on_page = set()
        for field_name, color, bbox in page_draws[page_idx]:
            fields_on_page.add(field_name)
            coords = [
                (bbox["x0"] * width, bbox["top"] * height),
                (bbox["x1"] * width, bbox["bottom"] * height),
            ]
            # Translucent fill
            r, g, b = _hex_to_rgb(color)
            draw.rectangle(coords, fill=(r, g, b, fill_opacity))
            # Solid outline
            draw.rectangle(coords, outline=color, width=bbox_width)

        # Blend overlay
        pil_image = pil_image.copy()
        from PIL import Image

        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = pil_image.convert("RGB")

        # Build legend
        legend_patches = [
            mpatches.Patch(color=field_color_map[f], label=f)
            for f in fields_on_page
        ]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(pil_image)
        ax.set_title(f"Page {page_idx}", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=10,
            framealpha=0.9,
        )
        plt.tight_layout()

        if out_dir:
            out_file = os.path.join(out_dir, f"page_{page_idx}.png")
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> page {page_idx} saved to {out_file}")
        else:
            plt.show()

        page_obj.close()

    pdf.close()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
