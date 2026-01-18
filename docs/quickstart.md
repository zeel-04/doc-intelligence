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

# Define your data model with citations
# If you want to include citations for any field, 
# Use the `processor.citation_type` as the type.
class MyData(BaseModel):
    my_data: str
    my_data_citation: processor.citation_type

# Extract structured data
response = processor.extract(
    model="gpt-5-mini",
    reasoning={"effort": "low"},
    response_format=MyData,
)

# Get the extracted data
data = response.model_dump()
print(data)
```

### Sample Output

```json
{
    "my_data": "my data",
    "my_data_citation": [{
        "page": 0,
        "lines": [10],
        "bboxes": [{
            "x0": 0.058823529411764705,
            "top": 0.6095707475757575,
            "x1": 0.5635455037254902,
            "bottom": 0.6221969596969696
        }]
    }]
}
```
