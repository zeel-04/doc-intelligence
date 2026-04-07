# Scanned PDF Extraction

Document AI supports scanned (image-only) PDFs through two pipeline types that
produce the same structured output as the digital pipeline. The existing formatter
and extractor work unchanged — only the parser differs.

## VLM Pipeline (recommended)

The VLM pipeline sends page images to a vision-capable LLM that performs layout
detection and OCR in a single call. No additional models or components needed —
it reuses the same LLM used for extraction.

### PDFProcessor

```python
from doc_intelligence import PDFProcessor, ParseStrategy, ScannedPipelineType

processor = PDFProcessor(
    provider="openai",
    strategy=ParseStrategy.SCANNED,
)

result = processor.extract("scanned_invoice.pdf", Invoice)
```

### DocumentProcessor (full control)

```python
from doc_intelligence import DocumentProcessor
from doc_intelligence.pdf.parser import PDFParser
from doc_intelligence.pdf.formatter import PDFFormatter
from doc_intelligence.pdf.extractor import PDFExtractor
from doc_intelligence.pdf.types import ParseStrategy, ScannedPipelineType
from doc_intelligence.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o")
processor = DocumentProcessor(
    parser=PDFParser(
        strategy=ParseStrategy.SCANNED,
        scanned_pipeline=ScannedPipelineType.VLM,
        llm=llm,
    ),
    formatter=PDFFormatter(),
    extractor=PDFExtractor(llm),
)

result = processor.extract("scanned_invoice.pdf", Invoice)
```

### Batch size

Control how many pages are sent per VLM call. Higher values reduce the number
of API calls but increase per-call token usage.

```python
processor = PDFProcessor(
    provider="openai",
    strategy=ParseStrategy.SCANNED,
    scanned_pipeline=ScannedPipelineType.VLM,
    vlm_batch_size=5,  # 5 pages per call
)
```

## Two-Stage Pipeline (not yet implemented)

!!! warning
    The two-stage pipeline (`ScannedPipelineType.TWO_STAGE`) is **not yet
    implemented**. Selecting it will raise `NotImplementedError`. Use the
    VLM pipeline above instead.

The two-stage pipeline will use separate layout detection and OCR models. You
will supply your own `BaseLayoutDetector` and `BaseOCREngine` implementations.
See the [Custom OCR Components](custom-ocr.md) guide for the planned contracts.

## Options

### DPI

Higher DPI improves OCR accuracy on dense documents at the cost of more memory and
processing time. The default is `150`.

```python
processor = PDFProcessor(
    provider="openai",
    strategy=ParseStrategy.SCANNED,
    scanned_pipeline=ScannedPipelineType.VLM,
    dpi=300,
)
```

## How It Works

### VLM Pipeline (Type 2)

1. **Render** — each PDF page is rasterised to a numpy array using `pypdfium2`.
2. **Encode** — page images are converted to base64 PNG data URLs.
3. **VLM call** — pages are batched and sent to a vision-capable LLM with a prompt
   requesting structured layout + OCR JSON output.
4. **Parse** — the JSON response is parsed into `Page` objects with typed
   `ContentBlock` items (TextBlock, TableBlock, etc.).

### Two-Stage Pipeline (Type 1) — planned

1. **Render** — each PDF page will be rasterised to a numpy array using `pypdfium2`.
2. **Layout detection** — your `BaseLayoutDetector` will segment the page image into
   typed regions (text, table, figure, …).
3. **OCR** — your `BaseOCREngine` will read text from each region concurrently. Text
   regions become `TextBlock`s; table regions become `TableBlock`s.

The resulting `PDFDocument` is identical in structure to what the digital strategy of
`PDFParser` produces, so citation tracking and multi-pass extraction work without any changes.
