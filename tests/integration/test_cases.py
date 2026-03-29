"""Data-driven test case definitions for integration tests.

Each list contains dicts that are consumed by ``@pytest.mark.parametrize``
in the corresponding test module. Adding a new dict automatically creates
a new test — no changes to test files required.
"""

from typing import Any

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
        "id": "with_line_numbers",
        "description": "Format with line numbers (citations mode)",
        "pdf": "simple_one_page.pdf",
        "include_citations": True,
        "expected": {
            "contains": ["0: Name: John", "1: Age: 30", "<page number=0>"],
        },
    },
    {
        "id": "without_line_numbers",
        "description": "Format without line numbers (no citations)",
        "pdf": "simple_one_page.pdf",
        "include_citations": False,
        "expected": {
            "contains": ["Name: John", "Age: 30", "<page number=0>"],
            "not_contains": ["0: Name"],
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
            "extraction_mode": "single_pass",
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
            "extraction_mode": "single_pass",
        },
        "mock_llm_response": (
            '{"name": {"value": "John", "citations": '
            '[{"page": 0, "lines": [0]}]}, '
            '"age": {"value": 30, "citations": '
            '[{"page": 0, "lines": [1]}]}}'
        ),
        "expected_data": {"name": "John", "age": 30},
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
            "extraction_mode": "single_pass",
        },
        "expected_data": {"name": "John", "age": 30},
    },
]
