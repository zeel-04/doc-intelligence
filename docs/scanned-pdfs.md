# Scanned PDF Extraction

Document AI supports scanned (image-only) PDFs through an OCR pipeline that produces
the same structured output as the digital pipeline. The existing formatter and extractor
work unchanged — only the parser differs.

## Installation

Install the `ocr` optional dependency group alongside the base package:

```bash
uv pip install "doc-intelligence[ocr]"
```

Or with pip:

```bash
pip install "doc-intelligence[ocr]"
```

This installs `paddleocr` and `paddlepaddle`, which provide the default layout detector
and OCR engine.

## Quick Start

### One-liner

```python
from pydantic import BaseModel
from doc_intelligence import extract

class Invoice(BaseModel):
    vendor: str
    total: float

result = extract(
    "scanned_invoice.pdf",
    Invoice,
    provider="openai",
    document_type="scanned",
)

print(result.data)
# Invoice(vendor='Acme Corp', total=1234.56)
```

The only change from the digital pipeline is `document_type="scanned"`.

### PDFProcessor

```python
from doc_intelligence import PDFProcessor

processor = PDFProcessor(provider="openai", document_type="scanned")
result = processor.extract("scanned_invoice.pdf", Invoice)
```

### DocumentProcessor (advanced)

For full control — custom DPI, custom detectors, shared processor across many files:

```python
from doc_intelligence import DocumentProcessor
from doc_intelligence.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o")
processor = DocumentProcessor.from_scanned_pdf(llm=llm, dpi=200)

result = processor.extract("scanned_invoice.pdf", Invoice)
```

## Options

### DPI

Higher DPI improves OCR accuracy on dense documents at the cost of more memory and
processing time. The default is `150`.

```python
processor = DocumentProcessor.from_scanned_pdf(llm=llm, dpi=300)
```

### Custom layout detector / OCR engine

See the [Custom OCR Components](custom-ocr.md) guide to swap in your own implementations.

## How It Works

The scanned pipeline has three stages:

1. **Render** — each PDF page is rasterised to a numpy array using `pypdfium2`.
2. **Layout detection** — `PaddleLayoutDetector` segments the page image into typed
   regions (text, table, figure, …).
3. **OCR** — `PaddleOCREngine` reads text from each region concurrently. Text regions
   become `TextBlock`s; table regions become `TableBlock`s.

The resulting `PDFDocument` is identical in structure to what `DigitalPDFParser`
produces, so citation tracking and multi-pass extraction work without any changes.
