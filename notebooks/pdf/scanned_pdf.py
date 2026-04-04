"""End-client script: extract structured data from a scanned PDF and visualize citations.

Uses Paddle for layout detection & OCR, and OpenAI gpt-5-mini for extraction.
Output images are saved to notebooks/pdf/output/scanned/ with color-coded
bounding boxes showing where each extracted field was found in the PDF.

Usage:
    python notebooks/pdf/scanned_pdf.py
"""

import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# -- Make notebooks/utils.py importable --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import show_all_fields

from doc_intelligence import DocumentProcessor, OpenAILLM
from doc_intelligence.ocr.paddle import PaddleLayoutDetector, PaddleOCREngine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_invoice.pdf")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "scanned")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Define extraction schema (matches the medical billing invoice)
# ---------------------------------------------------------------------------


class ServiceLine(BaseModel):
    date: str | None = Field(None, description="Date of service (e.g. 09/24/21)")
    code: str | None = Field(None, description="Procedure/billing code (e.g. U0003)")
    description: str = Field(..., description="Description of the service")
    charge: float = Field(..., description="Charge amount")
    adjustment: float = Field(..., description="Adjustment amount")
    insurance_payment: float = Field(..., description="Insurance payment amount")
    patient_responsibility: float = Field(..., description="Patient responsibility amount")
    patient_payment: float = Field(..., description="Patient payment amount")
    balance: float = Field(..., description="Remaining balance")


class MedicalInvoice(BaseModel):
    service_lines: list[ServiceLine] = Field(
        ..., description="Individual service line items from the billing table"
    )
    total_account_balance: float = Field(
        ..., description="Total account balance shown at the bottom"
    )
    insurance_balance: float = Field(
        ..., description="Insurance balance from the aging table"
    )
    patient_balance: float = Field(
        ..., description="Patient balance from the aging table"
    )


# ---------------------------------------------------------------------------
# 2. Build pipeline: Paddle (layout + OCR) → OpenAI gpt-5-mini (extraction)
# ---------------------------------------------------------------------------
llm = OpenAILLM(model="gpt-5-mini")

processor = DocumentProcessor.from_scanned_pdf(
    llm=llm,
    layout_detector=PaddleLayoutDetector(),
    ocr_engine=PaddleOCREngine(),
    dpi=200,
)

# ---------------------------------------------------------------------------
# 3. Extract
# ---------------------------------------------------------------------------
print(f"Extracting from: {PDF_PATH}")
print("Layout & OCR: PaddleOCR | Extraction LLM: gpt-5-mini")
print("-" * 60)

result = processor.extract(
    uri=PDF_PATH,
    response_format=MedicalInvoice,
    include_citations=True,
    extraction_mode="single_pass",
)

# ---------------------------------------------------------------------------
# 4. Print extracted data
# ---------------------------------------------------------------------------
print("\n=== Extracted Data ===")
print(result.data.model_dump_json(indent=2))

# ---------------------------------------------------------------------------
# 5. Visualize citations with colored bounding boxes
# ---------------------------------------------------------------------------
print("\n=== Generating bbox visualizations ===")
show_all_fields(
    pdf_path=PDF_PATH,
    metadata=result.metadata,
    out_dir=OUT_DIR,
)

print(f"\nDone! Check {OUT_DIR}/ for annotated page images.")
