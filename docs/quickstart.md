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

### Processing PDFs from URLs or Local Files

Document AI supports both local file paths and URLs:

```python
from dotenv import load_dotenv
from pydantic import BaseModel

from doc_intelligence import PDFProcessor

# Load environment variables
load_dotenv()

# Define your data model
class License(BaseModel):
    license_name: str

# Create a processor — supports "openai", "anthropic", "gemini", "ollama"
processor = PDFProcessor(provider="openai")

# Extract from a URL
result = processor.extract(
    uri="https://example-files.online-convert.com/document/pdf/example.pdf",
    response_format=License,
    include_citations=True,
    extraction_mode="single_pass",
    model="gpt-4o-mini",
)

# Or from a local file:
# result = processor.extract(
#     uri="path/to/your/document.pdf",
#     response_format=License,
# )

# Access the extracted data and citations
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

### Configuration Options

The `extract()` method accepts these keyword arguments:

- **`uri`**: Path to a local PDF file or a URL
- **`response_format`**: Your Pydantic model class defining the extraction schema
- **`include_citations`**: Set to `True` to get citation information with bounding boxes (default: `True`)
- **`extraction_mode`**: `"single_pass"` or `"multi_pass"` (default: `"single_pass"`)
- **`page_numbers`**: Optional list of page indices to process (0-indexed)
- **`model`**: Override the default model for this call (e.g., `"gpt-4o"`)
- **`llm_config`**: Additional LLM configuration as a dict (e.g., `{"reasoning": {"effort": "minimal"}}`)
