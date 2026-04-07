"""Data-driven test case definitions for integration tests.

Each list contains dicts that are consumed by ``@pytest.mark.parametrize``
in the corresponding test module. Adding a new dict automatically creates
a new test — no changes to test files required.
"""

import json
from typing import Any

from doc_intelligence.pdf.types import PDFExtractionMode

# ---------------------------------------------------------------------------
# Parse cases — tests/integration/test_parse.py
# ---------------------------------------------------------------------------
PARSE_CASES: list[dict[str, Any]] = [
    {
        "id": "basic_one_page",
        "description": "Parse a simple one-page PDF with 3 lines",
        "pdf": "simple_one_page.pdf",
        "expected": {
            "num_pages": 1,
            "pages": [
                {
                    "page_index": 0,
                    "num_lines": 3,
                    "lines": [
                        {"contains": "Name: John"},
                        {"contains": "Age: 30"},
                        {"contains": "City: Springfield"},
                    ],
                }
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Format cases — tests/integration/test_format.py
# ---------------------------------------------------------------------------
FORMAT_CASES: list[dict[str, Any]] = [
    {
        "id": "with_block_indices",
        "description": "Format with block indices (citations mode)",
        "pdf": "simple_one_page.pdf",
        "include_citations": True,
        "expected": {
            "contains": [
                '<block index="0" type="text">',
                "Name: John",
                "Age: 30",
                '<page number="0">',
            ],
        },
    },
    {
        "id": "without_block_indices",
        "description": "Format without block indices (no citations)",
        "pdf": "simple_one_page.pdf",
        "include_citations": False,
        "expected": {
            "contains": ["Name: John", "Age: 30", '<page number="0">'],
            "not_contains": ["<block"],
        },
    },
]

# ---------------------------------------------------------------------------
# Extract cases (mocked LLM) — tests/integration/test_extract.py
# ---------------------------------------------------------------------------
EXTRACT_CASES: list[dict[str, Any]] = [
    {
        "id": "single_pass_no_citations",
        "description": "Single-pass extraction without citations",
        "pdf": "simple_one_page.pdf",
        "schema": "SimpleExtraction",
        "config": {
            "include_citations": False,
            "extraction_mode": PDFExtractionMode.SINGLE_PASS,
        },
        "mock_llm_response": '{"name": "John", "age": 30}',
        "expected_data": {"name": "John", "age": 30},
    },
    {
        "id": "single_pass_with_citations",
        "description": "Single-pass extraction with citations enabled",
        "pdf": "simple_one_page.pdf",
        "schema": "SimpleExtraction",
        "config": {
            "include_citations": True,
            "extraction_mode": PDFExtractionMode.SINGLE_PASS,
        },
        "mock_llm_response": (
            '{"name": {"value": "John", "citations": '
            '[{"page": 0, "blocks": [0]}]}, '
            '"age": {"value": 30, "citations": '
            '[{"page": 0, "blocks": [1]}]}}'
        ),
        "expected_data": {"name": "John", "age": 30},
    },
]

# ---------------------------------------------------------------------------
# Live extract cases — tests/integration/test_extract.py (requires --run-live)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VLM parse cases (mocked) — tests/integration/test_parse.py
# ---------------------------------------------------------------------------
VLM_PARSE_CASES: list[dict[str, Any]] = [
    {
        "id": "vlm_basic_one_page",
        "description": "VLM parse of a simple one-page PDF (mocked)",
        "pdf": "simple_one_page.pdf",
        "mock_vlm_response": json.dumps(
            {
                "pages": [
                    {
                        "page_index": 0,
                        "blocks": [
                            {
                                "block_type": "text",
                                "bbox": {"x0": 10, "y0": 20, "x1": 400, "y1": 40},
                                "text": "Name: John",
                            },
                            {
                                "block_type": "text",
                                "bbox": {"x0": 10, "y0": 50, "x1": 400, "y1": 70},
                                "text": "Age: 30",
                            },
                            {
                                "block_type": "text",
                                "bbox": {"x0": 10, "y0": 80, "x1": 400, "y1": 100},
                                "text": "City: Springfield",
                            },
                        ],
                    }
                ]
            }
        ),
        "expected": {
            "num_pages": 1,
            "pages": [
                {
                    "page_index": 0,
                    "num_blocks": 3,
                    "blocks": [
                        {"contains": "Name: John"},
                        {"contains": "Age: 30"},
                        {"contains": "City: Springfield"},
                    ],
                }
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Live extract cases — tests/integration/test_extract.py (requires --run-live)
# ---------------------------------------------------------------------------
LIVE_EXTRACT_CASES: list[dict[str, Any]] = [
    {
        "id": "live_openai_single_pass",
        "description": "Real OpenAI extraction of name and age",
        "pdf": "simple_one_page.pdf",
        "schema": "SimpleExtraction",
        "provider": "openai",
        "config": {
            "include_citations": False,
            "extraction_mode": PDFExtractionMode.SINGLE_PASS,
        },
        "expected_data": {"name": "John", "age": 30},
    },
]
