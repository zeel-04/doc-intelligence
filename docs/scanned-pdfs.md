# Scanned PDF Extraction

Document AI supports scanned (image-only) PDFs through an OCR pipeline that produces
the same structured output as the digital pipeline. The existing formatter and extractor
work unchanged — only the parser differs.

You must supply your own `BaseLayoutDetector` and `BaseOCREngine` implementations.
See the [Custom OCR Components](custom-ocr.md) guide for the contracts and examples.

## Quick Start

### DocumentProcessor (recommended)

```python
from doc_intelligence import DocumentProcessor
from doc_intelligence.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o")
processor = DocumentProcessor.from_scanned_pdf(
    llm=llm,
    layout_detector=my_layout_detector,
    ocr_engine=my_ocr_engine,
    dpi=200,
)

result = processor.extract("scanned_invoice.pdf", Invoice)
```

### PDFProcessor

```python
from doc_intelligence import PDFProcessor

processor = PDFProcessor(
    provider="openai",
    document_type="scanned",
    layout_detector=my_layout_detector,
    ocr_engine=my_ocr_engine,
)
result = processor.extract("scanned_invoice.pdf", Invoice)
```

## Options

### DPI

Higher DPI improves OCR accuracy on dense documents at the cost of more memory and
processing time. The default is `150`.

```python
processor = DocumentProcessor.from_scanned_pdf(
    llm=llm,
    layout_detector=my_layout_detector,
    ocr_engine=my_ocr_engine,
    dpi=300,
)
```

### Custom layout detector / OCR engine

See the [Custom OCR Components](custom-ocr.md) guide to build your own implementations.

## How It Works

The scanned pipeline has three stages:

1. **Render** — each PDF page is rasterised to a numpy array using `pypdfium2`.
2. **Layout detection** — your `BaseLayoutDetector` segments the page image into typed
   regions (text, table, figure, …).
3. **OCR** — your `BaseOCREngine` reads text from each region concurrently. Text regions
   become `TextBlock`s; table regions become `TableBlock`s.

The resulting `PDFDocument` is identical in structure to what `DigitalPDFParser`
produces, so citation tracking and multi-pass extraction work without any changes.
