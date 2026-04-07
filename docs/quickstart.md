# Quickstart

This guide will help you get started with Document AI.

## Installation

## Requirements

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

## Environment Setup

Document AI supports multiple LLM providers. Set up your API key for your chosen provider:

```bash
# OpenAI
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Or Anthropic
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env

# Or Gemini
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

## Basic Usage

Here's a simple example to extract structured data from a PDF document.

Create a `PDFProcessor` once with your pipeline configuration. The document and schema are always provided per call:

```python
from dotenv import load_dotenv
from pydantic import BaseModel

from doc_intelligence import PDFExtractionMode, PDFProcessor

load_dotenv()

class License(BaseModel):
    license_name: str

processor = PDFProcessor(
    provider="openai",
    model="gpt-5",
    include_citations=True,
    extraction_mode=PDFExtractionMode.SINGLE_PASS,
    llm_config={"temperature": 0.2},
)

result = processor.extract(
    "https://example-files.online-convert.com/document/pdf/example.pdf",
    License,
)

# Or from a local file:
# result = processor.extract("path/to/your/document.pdf", License)

# Different schema, same processor
# result = processor.extract("receipt.pdf", Receipt, page_numbers=[0])

print(f"Extracted data: {result.data}")
print(f"Metadata: {result.metadata}")
```

### Sample Output

The `extract` method returns an `ExtractionResult` with two attributes:

1. **`.data`**: The extracted data as a Pydantic model instance
2. **`.metadata`**: Citation information for each field with values and bounding boxes

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

### Configuration Reference

**`PDFProcessor` constructor** — pipeline config, set once:

| Parameter | Description | Default |
|---|---|---|
| `provider` | LLM provider (`"openai"`, `"anthropic"`, `"gemini"`, `"ollama"`) | required\* |
| `model` | Model name for the provider | provider default |
| `include_citations` | Include citation bounding boxes in results | `True` |
| `extraction_mode` | `SINGLE_PASS` or `MULTI_PASS` | `SINGLE_PASS` |
| `llm_config` | Generation parameters (e.g. `{"temperature": 0.2}`) | `None` |

\* Or pass a pre-built `llm=` instance instead of `provider`.

**`processor.extract()`** — document-specific, vary per call:

| Parameter | Description |
|---|---|
| `uri` | Path or URL of the PDF (required) |
| `response_format` | Pydantic model class for the extraction schema (required) |
| `page_numbers` | List of 0-indexed page numbers to process (default: all pages) |
