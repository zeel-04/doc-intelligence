"""Layout detection + OCR via vision LLMs (OpenAI or Gemini).

Sends each PDF page as an image to a vision model and asks it to return
structured layout blocks (text, table, figure, header, footer, etc.) with
bounding boxes and OCR'd text in a single pass.

Usage:
    python notebooks/layout_ocr.py <pdf_path> --provider openai
    python notebooks/layout_ocr.py <pdf_path> --provider gemini --model gemini-2.5-flash
    python notebooks/layout_ocr.py <pdf_path> --pages 0 1 2 --out results.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

BLOCK_TYPES = [
    "text",
    "table",
    "figure",
    "header",
    "footer",
    "page_number",
    "caption",
    "list",
    "formula",
]


class BBox(BaseModel):
    """Bounding box coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int


class LayoutBlock(BaseModel):
    """A single detected region on the page."""

    block_type: str = Field(
        ..., description=f"One of: {', '.join(BLOCK_TYPES)}"
    )
    bbox: BBox
    text: str = Field(
        ..., description="OCR'd text content. For tables use markdown pipe syntax."
    )


class PageLayout(BaseModel):
    """Layout analysis result for a single page."""

    page_index: int
    width: int
    height: int
    blocks: list[LayoutBlock]


class DocumentLayout(BaseModel):
    """Full document layout + OCR result."""

    pages: list[PageLayout]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a document layout analysis and OCR engine. Given an image of a \
document page, detect every content region and return structured JSON.

For each region, provide:
- block_type: one of [text, table, figure, header, footer, page_number, caption, list, formula]
- bbox: bounding box as {x0, y0, x1, y1}
- text: the OCR'd text content of that region. For tables, use markdown pipe-delimited format. \
For figures, describe the figure briefly.

Rules:
- Preserve reading order (top-to-bottom, left-to-right).
- Every piece of visible text must belong to exactly one block.
- Do NOT merge unrelated regions. Headers, footers, and page numbers are separate blocks.
- For multi-column layouts, read left column fully before right column.
- Be thorough: do not skip any content on the page.\
"""

USER_PROMPT = """\
Analyse this document page. Detect all layout regions and OCR the text. \
Return JSON matching this schema exactly:

{schema}

Return ONLY the JSON object, no markdown fences or explanation.\
"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def render_page_to_base64(pdf_path: str | Path, page_index: int, scale: float = 2.0) -> str:
    """Render a PDF page to a base64-encoded PNG data URL."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index]
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    width, height = pil_image.size
    page.close()
    pdf.close()

    from io import BytesIO

    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", width, height


def get_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    count = len(pdf)
    pdf.close()
    return count


# ---------------------------------------------------------------------------
# Provider: OpenAI
# ---------------------------------------------------------------------------


def run_openai(
    image_data_url: str,
    page_width: int,
    page_height: int,
    page_index: int,
    model: str = "gpt-4o",
) -> PageLayout:
    """Send a page image to OpenAI and get layout + OCR."""
    from openai import OpenAI

    client = OpenAI()

    schema_text = json.dumps(PageLayout.model_json_schema(), indent=2)
    user_text = USER_PROMPT.format(schema=schema_text)

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
    )

    raw = json.loads(response.output_text)

    # The model may return the blocks list directly or nested under a key
    if "blocks" in raw:
        blocks_data = raw["blocks"]
    elif "pages" in raw and len(raw["pages"]) > 0:
        blocks_data = raw["pages"][0].get("blocks", [])
    else:
        blocks_data = raw if isinstance(raw, list) else []

    blocks = [LayoutBlock.model_validate(b) for b in blocks_data]
    return PageLayout(
        page_index=page_index,
        width=page_width,
        height=page_height,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# Provider: Gemini
# ---------------------------------------------------------------------------


def run_gemini(
    image_data_url: str,
    page_width: int,
    page_height: int,
    page_index: int,
    model: str = "gemini-2.5-flash",
) -> PageLayout:
    """Send a page image to Gemini and get layout + OCR."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client()

    schema_text = json.dumps(PageLayout.model_json_schema(), indent=2)
    user_text = USER_PROMPT.format(schema=schema_text)

    # Decode the data URL to raw bytes for Gemini's inline_data format
    header, b64_data = image_data_url.split(",", 1)
    mime_type = header.split(":")[1].split(";")[0]
    image_bytes = base64.b64decode(b64_data)

    response = client.models.generate_content(
        model=model,
        contents=[
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(text=user_text),
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            mime_type=mime_type,
                            data=image_bytes,
                        )
                    ),
                ],
            )
        ],
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )

    raw = json.loads(response.text)

    if "blocks" in raw:
        blocks_data = raw["blocks"]
    elif "pages" in raw and len(raw["pages"]) > 0:
        blocks_data = raw["pages"][0].get("blocks", [])
    else:
        blocks_data = raw if isinstance(raw, list) else []

    blocks = [LayoutBlock.model_validate(b) for b in blocks_data]
    return PageLayout(
        page_index=page_index,
        width=page_width,
        height=page_height,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": run_openai,
    "gemini": run_gemini,
}


def analyse_document(
    pdf_path: str | Path,
    provider: str = "openai",
    model: str | None = None,
    page_indices: list[int] | None = None,
    scale: float = 2.0,
) -> DocumentLayout:
    """Run layout detection + OCR on a PDF document.

    Args:
        pdf_path: Path to the PDF file.
        provider: "openai" or "gemini".
        model: Model name override. Defaults to gpt-4o / gemini-2.5-flash.
        page_indices: Specific pages to process (0-indexed). None = all pages.
        scale: Rendering scale for PDF-to-image conversion.

    Returns:
        DocumentLayout with per-page blocks.
    """
    run_fn = PROVIDERS.get(provider.lower())
    if run_fn is None:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from: {sorted(PROVIDERS)}")

    total_pages = get_page_count(pdf_path)
    if page_indices is None:
        page_indices = list(range(total_pages))

    kwargs = {}
    if model:
        kwargs["model"] = model

    pages: list[PageLayout] = []
    for idx in page_indices:
        if idx < 0 or idx >= total_pages:
            logger.warning(f"Skipping out-of-range page index: {idx}")
            continue

        logger.info(f"Processing page {idx + 1}/{total_pages} ...")
        data_url, w, h = render_page_to_base64(pdf_path, idx, scale=scale)
        page_layout = run_fn(
            image_data_url=data_url,
            page_width=w,
            page_height=h,
            page_index=idx,
            **kwargs,
        )
        logger.info(f"  -> {len(page_layout.blocks)} blocks detected")
        pages.append(page_layout)

    return DocumentLayout(pages=pages)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

BLOCK_COLORS = {
    "text": "#3cb44b",
    "table": "#4363d8",
    "figure": "#f58231",
    "header": "#e6194b",
    "footer": "#911eb4",
    "page_number": "#42d4f4",
    "caption": "#f032e6",
    "list": "#bfef45",
    "formula": "#469990",
}


def visualize(
    pdf_path: str | Path,
    result: DocumentLayout,
    out_dir: str | None = None,
    scale: float = 2.0,
    figsize: tuple = (14, 18),
) -> None:
    """Render pages with color-coded layout bounding boxes.

    Args:
        pdf_path: Path to the source PDF.
        result: The DocumentLayout from analyse_document().
        out_dir: Save images to this directory. None = display inline.
        scale: PDF rendering scale.
        figsize: Matplotlib figure size.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import ImageDraw

    pdf = pdfium.PdfDocument(str(pdf_path))

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    for page_layout in result.pages:
        page_obj = pdf[page_layout.page_index]
        pil_image = page_obj.render(scale=scale).to_pil()
        img_w, img_h = pil_image.size
        draw = ImageDraw.Draw(pil_image)

        types_seen = set()
        for block in page_layout.blocks:
            color = BLOCK_COLORS.get(block.block_type, "#999999")
            types_seen.add(block.block_type)

            # Convert 0-1000 normalised coords to pixel coords
            x0 = block.bbox.x0 / 1000 * img_w
            y0 = block.bbox.y0 / 1000 * img_h
            x1 = block.bbox.x1 / 1000 * img_w
            y1 = block.bbox.y1 / 1000 * img_h
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Legend
        legend_patches = [
            mpatches.Patch(
                color=BLOCK_COLORS.get(t, "#999999"), label=t
            )
            for t in sorted(types_seen)
        ]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(pil_image)
        ax.set_title(f"Page {page_layout.page_index}", fontsize=14, fontweight="bold")
        ax.axis("off")
        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.9)
        plt.tight_layout()

        if out_dir:
            out_file = str(Path(out_dir) / f"page_{page_layout.page_index}.png")
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved -> {out_file}")
        else:
            plt.show()

        page_obj.close()

    pdf.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layout detection + OCR via vision LLMs"
    )
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        default=None,
        help="Page indices to process (0-indexed). Default: all",
    )
    parser.add_argument("--scale", type=float, default=2.0, help="Render scale")
    parser.add_argument("--out", default=None, help="Save JSON results to this file")
    parser.add_argument(
        "--visualize",
        default=None,
        help="Save visualization images to this directory",
    )

    args = parser.parse_args()

    result = analyse_document(
        pdf_path=args.pdf,
        provider=args.provider,
        model=args.model,
        page_indices=args.pages,
        scale=args.scale,
    )

    # Print summary
    for page in result.pages:
        print(f"\n--- Page {page.page_index} ({page.width}x{page.height}) ---")
        for block in page.blocks:
            preview = block.text[:80].replace("\n", " ")
            print(
                f"  [{block.block_type:12s}] "
                f"({block.bbox.x0},{block.bbox.y0})-({block.bbox.x1},{block.bbox.y1}) "
                f"{preview}..."
            )

    # Save JSON
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(result.model_dump_json(indent=2))
        print(f"\nResults saved to {args.out}")

    # Visualize
    if args.visualize:
        visualize(args.pdf, result, out_dir=args.visualize, scale=args.scale)


if __name__ == "__main__":
    main()
