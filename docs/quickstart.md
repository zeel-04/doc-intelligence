# Quickstart

This guide will help you get started with Document AI.

## Installation

## Requirements

- Python >= 3.10
- OpenAI API key

### Install uv

First, install [uv](https://docs.astral.sh/uv/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install from Source

Clone the repository and install the package:

```bash
# Clone the repository
git clone https://github.com/zeel-04/document-ai.git
cd document-ai

# Install the package with uv
uv sync
```

### Install from Git (Alternative)

You can also install directly from the git repository:

```bash
uv pip install git+https://github.com/zeel-04/document-ai.git
```

## Environment Setup

Document AI uses OpenAI's API for document processing. Set up your API key:

```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Basic Usage

Here's a simple example to extract structured data from a PDF document:

```python
from dotenv import load_dotenv
from document_ai.processer import DocumentProcessor
from document_ai.llm import OpenAILLM
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = OpenAILLM()

# Create a processor from a PDF file
processor = DocumentProcessor.from_digital_pdf(
    uri="path/to/your/document.pdf",
    llm=llm,
)

# Define your data model
class Balance(BaseModel):
    ending_balance: float

# Configure extraction with citations
config = {
    "response_format": Balance,
    "llm_config": {
        "model": "gpt-5",
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

# Get the extracted data and citations
data, citations = response
print(f"Extracted data: {data}")
print(f"Citations: {citations}")
```

### Sample Output

The `extract` method returns a tuple containing:
1. The extracted data as a Pydantic model instance
2. A dictionary with citation information for each field

```python
# Example output
(Balance(ending_balance=111.61),
 {'ending_balance': {'value': 111.61,
   'citations': [{'page': 0,
     'bboxes': [{'x0': 0.058823529411764705,
       'top': 0.6095707475757575,
       'x1': 0.5635455037254902,
       'bottom': 0.6221969596969696}]}]}})
```

### Configuration Options

- **response_format**: Your Pydantic model class
- **llm_config**: 
  - `model`: The OpenAI model to use (e.g., "gpt-5", "gpt-4o")
  - `reasoning`: Optional reasoning configuration with `effort` level ("minimal", "low", "medium", "high")
- **extraction_config**:
  - `include_citations`: Set to `True` to get citation information
  - `extraction_mode`: "single_pass" for single-pass extraction
  - `page_numbers`: Optional list of page indices to process (0-indexed)
