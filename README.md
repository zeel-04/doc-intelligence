# Document AI

**Documentation:** [https://zeel-04.github.io/doc-intelligence/](https://zeel-04.github.io/doc-intelligence/)

A library for parsing, formatting, and processing documents that can be used to build AI-powered document processing pipelines with structured data extraction and citation support.

![Document AI](./docs/static/doc_intelligence.jpg)

## Features

- Extract structured data from PDF documents using LLMs
- Automatic citation tracking with page numbers, line numbers, and bounding boxes
- Support for digital PDFs and scanned (image-only) PDFs via OCR
- Type-safe data models using Pydantic
- Multi-provider LLM support: OpenAI, Anthropic, Gemini, Ollama
- Pluggable OCR pipeline — swap in any layout detector or OCR engine

## Installation

### Requirements

- Python >= 3.10
- An API key for your chosen LLM provider (OpenAI, Anthropic, or Gemini) — or a local Ollama server

### Install with uv

```bash
uv pip install doc-intelligence
```

Or with pip:

```bash
pip install doc-intelligence
```

## Quick Start

Set up your API key (example with OpenAI):

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Configure a `PDFProcessor` once, then pass the document and schema per call:

```python
from doc_intelligence import PDFExtractionMode, PDFProcessor
from pydantic import BaseModel

class License(BaseModel):
    license_name: str

processor = PDFProcessor(
    provider="openai",
    model="gpt-4o-mini",
    include_citations=True,
    extraction_mode=PDFExtractionMode.SINGLE_PASS,
)

result = processor.extract(
    "https://example-files.online-convert.com/document/pdf/example.pdf",
    License,
)
print(f"Extracted data: {result.data}")
print(f"Metadata: {result.metadata}")
```

### Sample Output

The `extract` method returns an `ExtractionResult` with `.data` and `.metadata` attributes:

```python
result.data
# License(license_name='Attribution-ShareAlike 3.0 Unported')

result.metadata
# {
#     'license_name': {
#         'value': 'Attribution-ShareAlike 3.0 Unported',
#         'citations': [{
#             'page': 0,
#             'bboxes': [{
#                 'x0': 0.201,
#                 'top': 0.859,
#                 'x1': 0.565,
#                 'bottom': 0.872
#             }]
#         }]
#     }
# }
```

## Scanned PDFs

For image-only PDFs, use `strategy=ParseStrategy.SCANNED` and supply your own layout detector and OCR engine:

```python
from doc_intelligence import PDFProcessor, ParseStrategy

processor = PDFProcessor(
    provider="openai",
    strategy=ParseStrategy.SCANNED,
    layout_detector=my_layout_detector,
    ocr_engine=my_ocr_engine,
)
result = processor.extract("scanned_invoice.pdf", Invoice)
```

See the [Scanned PDFs guide](https://zeel-04.github.io/doc-intelligence/scanned-pdfs/) and
[Custom OCR Components](https://zeel-04.github.io/doc-intelligence/custom-ocr/) docs for details.

## Documentation

For more detailed documentation, see the [docs](./docs) directory or visit the [documentation site](https://zeel-04.github.io/doc-intelligence/).

## Development Setup

Prerequisites:

- Python 3.10+
- uv

```bash
git clone https://github.com/zeel-04/doc-intelligence.git
cd doc_intelligence
uv venv
uv sync
```
