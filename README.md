# Document AI

**Documentation:** [https://zeel-04.github.io/doc-intelligence/](https://zeel-04.github.io/doc-intelligence/)

A library for parsing, formatting, and processing documents that can be used to build AI-powered document processing pipelines with structured data extraction and citation support.

![Document AI](./docs/static/doc_intelligence.jpg)

## Features

- Extract structured data from PDF documents using LLMs
- Automatic citation tracking with page numbers, line numbers, and bounding boxes
- Support for digital PDFs (local files and URLs)
- Type-safe data models using Pydantic
- OpenAI integration with support for reasoning models

## Installation

### Requirements

- Python >= 3.10
- OpenAI API key

### Install with uv

```bash
uv pip install doc-intelligence
```

Or with pip:

```bash
pip install doc-intelligence
```

## Quick Start

Set up your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Here's a simple example to extract structured data from a PDF:

```python
from dotenv import load_dotenv
from doc_intelligence.processer import DocumentProcessor
from doc_intelligence.llm import OpenAILLM
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = OpenAILLM()

# Create a processor from a PDF file (local or URL)
processor = DocumentProcessor.from_digital_pdf(
    uri="https://example-files.online-convert.com/document/pdf/example.pdf",  # Can also be a local path
    llm=llm,
)

# Define your data model
class License(BaseModel):
    license_name: str

# Configure extraction with citations
config = {
    "response_format": License,
    "llm_config": {
        "model": "gpt-5-mini",
        "reasoning": {"effort": "minimal"},
    },
    "extraction_config": {
        "include_citations": True,
        "extraction_mode": "single_pass",
        "page_numbers": [0, 1],  # Optional: specify which pages to process
    }
}

# Extract structured data
response = processor.extract(config)

# Access the extracted data and citations
extracted_data = response["extracted_data"]
metadata = response["metadata"]
print(f"Extracted data: {extracted_data}")
print(f"Metadata: {metadata}")
```

### Sample Output

The `extract` method returns a dictionary containing the extracted data and metadata with citation information:

```python
{
    'extracted_data': License(license_name='Attribution-ShareAlike 3.0 Unported'),
    'metadata': {
        'license_name': {
            'value': 'Attribution-ShareAlike 3.0 Unported',
            'citations': [{
                'page': 0,
                'bboxes': [{
                    'x0': 0.20106913928643427,
                    'top': 0.8587326111744586,
                    'x1': 0.5648947389639185,
                    'bottom': 0.8718454960091222
                }]
            }]
        }
    }
}
```

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
