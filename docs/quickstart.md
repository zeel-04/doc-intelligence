# Quickstart

This guide will help you get started with Document AI.

## Installation

## Requirements

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

## Environment Setup

Document AI uses OpenAI's API for document processing. Set up your API key:

```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Basic Usage

Here's a simple example to extract structured data from a PDF document.

### Processing PDFs from URLs or Local Files

Document AI supports both local file paths and URLs:

```python
from dotenv import load_dotenv
from doc_intelligence.processer import DocumentProcessor
from doc_intelligence.llm import OpenAILLM
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = OpenAILLM()

# Create a processor from a PDF URL
processor = DocumentProcessor.from_digital_pdf(
    uri="https://example-files.online-convert.com/document/pdf/example.pdf",
    llm=llm,
)

# Or use a local file path:
# processor = DocumentProcessor.from_digital_pdf(
#     uri="path/to/your/document.pdf",
#     llm=llm,
# )

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

The `extract` method returns a dictionary containing:
1. **extracted_data**: The extracted data as a Pydantic model instance
2. **metadata**: Citation information for each field with values and bounding boxes

```python
# Example output
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

### Configuration Options

- **response_format**: Your Pydantic model class
- **llm_config**: 
  - `model`: The OpenAI model to use (e.g., "gpt-5-mini", "gpt-4o")
  - `reasoning`: Optional reasoning configuration with `effort` level ("minimal", "low", "medium", "high")
- **extraction_config**:
  - `include_citations`: Set to `True` to get citation information with bounding boxes
  - `extraction_mode`: "single_pass" for single-pass extraction
  - `page_numbers`: Optional list of page indices to process (0-indexed)
