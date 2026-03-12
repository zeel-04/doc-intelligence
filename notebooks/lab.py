"""Lab script — single-pass and multi-pass extraction demo."""

import os
from io import BytesIO
from pprint import pprint
from typing import cast
from urllib.parse import urlparse

import matplotlib
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from doc_intelligence.llm import OpenAILLM
from doc_intelligence.pdf.processor import DocumentProcessor
from doc_intelligence.pdf.schemas import PDFDocument

matplotlib.use("Agg")  # non-interactive backend — saves to file instead of GUI

load_dotenv()

PDF_URL = "https://example-files.online-convert.com/document/pdf/example.pdf"
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_pdf_with_bboxes(
    pdf_path: str,
    data: dict,
    out_file: str,
    bbox_color: str = "red",
    bbox_width: int = 3,
    scale: float = 2.0,
    figsize: tuple = (12, 16),
) -> None:
    """Render a PDF page with highlighted bboxes and save to *out_file*."""
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

    from PIL import ImageDraw

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
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved to {out_file}")

    page_obj.close()
    pdf.close()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
llm = OpenAILLM()


class License(BaseModel):
    license_name: str


# ===========================================================================
# Single-pass extraction
# ===========================================================================
print("\n" + "=" * 60)
print("SINGLE-PASS EXTRACTION")
print("=" * 60)

processor = DocumentProcessor.from_digital_pdf(uri=PDF_URL, llm=llm)

config = {
    "response_format": License,
    "llm_config": {
        "model": "gpt-4o-mini",
    },
    "extraction_config": {
        "include_citations": True,
        "extraction_mode": "single_pass",
        "page_numbers": [0, 1],
    },
}

response = processor.extract(config)

print("\nResult:")
pprint(response)

print("\nVisualising bbox...")
show_pdf_with_bboxes(
    PDF_URL,
    response["metadata"]["license_name"]["citations"][0],
    out_file=os.path.join(OUT_DIR, "single_pass_bbox.png"),
)

# ===========================================================================
# Multi-pass extraction
# ===========================================================================
print("\n" + "=" * 60)
print("MULTI-PASS EXTRACTION  (Pass 1 → Pass 2 → Pass 3)")
print("=" * 60)

mp_processor = DocumentProcessor.from_digital_pdf(uri=PDF_URL, llm=llm)

mp_config = {
    "response_format": License,
    "llm_config": {
        "model": "gpt-4o-mini",
    },
    "extraction_config": {
        "include_citations": True,
        "extraction_mode": "multi_pass",
    },
}

mp_response = mp_processor.extract(mp_config)

mp_doc = cast(PDFDocument, mp_processor.document)
print("\nPass 1 result (plain Pydantic model, no citation wrappers):")
print(" ", mp_doc.pass1_result)

print("\nPass 2 page map (which pages each field was found on):")
print(" ", mp_doc.pass2_page_map)

print("\nFinal result (extracted_data from Pass 1, metadata/bboxes from Pass 3):")
pprint(mp_response)

print("\nVisualising bbox...")
show_pdf_with_bboxes(
    PDF_URL,
    mp_response["metadata"]["license_name"]["citations"][0],
    out_file=os.path.join(OUT_DIR, "multi_pass_bbox.png"),
)

print("\nDone.")
