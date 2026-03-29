# Integration & E2E Testing Spec — doc_intelligence

**Version:** 1.0
**Status:** Approved
**Last updated:** 2026-03-29

---

## 1. Motivation

All existing tests use synthetic in-memory fixtures and mock every external dependency. This is good for unit testing but leaves gaps:

- No real PDF parsing is ever exercised in the test suite.
- The full pipeline (parse → format → extract) is never tested end-to-end with real data.
- There is no way to verify extraction accuracy against a known PDF with expected answers.
- Adding a new test case requires writing Python code instead of just adding data.

This spec introduces a **data-driven integration test framework** where test cases are Python dicts, real PDFs are used, and both mocked-LLM and live-LLM modes are supported.

---

## 2. Directory Structure

```
tests/integration/
├── __init__.py
├── conftest.py          # --run-live flag, markers, pdf_path fixture, schema registry
├── test_cases.py        # All test case dicts: PARSE_CASES, FORMAT_CASES, EXTRACT_CASES, LIVE_EXTRACT_CASES
├── pdfs/                # Real test PDF files (<100KB each, committed to repo)
│   └── simple_one_page.pdf
├── test_parse.py        # Data-driven parse-only tests
├── test_format.py       # Data-driven format-only tests
└── test_extract.py      # Data-driven full extraction tests (mocked + live)
```

---

## 3. Test Case Format

Test cases are Python dicts in `test_cases.py`, organized into lists by pipeline stage. Every case must have an `id` (used as the pytest ID) and a `description`.

### 3.1 Parse Cases

```python
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
                    {"contains": "Name"},
                    {"contains": "Age"},
                    {"contains": "City"},
                ],
            }
        ],
    },
}
```

**Assertions:** Page count, line count per page, substring match on each line's text.

### 3.2 Format Cases

```python
{
    "id": "with_line_numbers",
    "description": "Format with line numbers (citations mode)",
    "pdf": "simple_one_page.pdf",
    "include_citations": True,
    "expected": {
        "contains": ["0:", "Name"],
        "not_contains": [],  # optional
    },
}
```

**Assertions:** Output string contains all `expected["contains"]` substrings. If `not_contains` is present, assert those substrings are absent.

### 3.3 Extract Cases (Mocked LLM)

```python
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
}
```

**Assertions:** Exact match — `result.data.model_dump() == case["expected_data"]`.

### 3.4 Live Extract Cases

```python
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
}
```

**Assertions:** Same exact match, but with a real LLM. Marked `@pytest.mark.live`, skipped without `--run-live`.

---

## 4. Infrastructure (`conftest.py`)

| Component | Purpose |
|---|---|
| `--run-live` CLI flag | `pytest_addoption` — enables live LLM tests |
| `@pytest.mark.live` marker | `pytest_configure` — registered; `pytest_collection_modifyitems` auto-skips without flag |
| `pdf_path` fixture | Callable returning absolute path: `pdf_path("foo.pdf")` → `tests/integration/pdfs/foo.pdf` |
| `SCHEMA_REGISTRY` dict | Maps string names to Pydantic models: `{"SimpleExtraction": SimpleExtraction, ...}` |
| `resolve_schema(name)` | Looks up from registry, raises `KeyError` if missing |
| `is_live` fixture | Returns `True` if `--run-live` was passed |

Schema models (`SimpleExtraction`, `NestedExtraction`) are imported from `tests/conftest.py` to avoid duplication.

---

## 5. Test PDF Files

PDFs in `tests/integration/pdfs/` are small (<100KB), deterministic, and committed to the repo. The first PDF (`simple_one_page.pdf`) is generated with `fpdf2` and contains:

```
Name: John
Age: 30
City: Springfield
```

Each line is a separate text line in the PDF so the parser produces exactly 3 `Line` objects.

---

## 6. Phased Implementation

### Phase 1: Scaffolding & Infrastructure

**Goal:** Directory structure, conftest, PDF generation.

**Files created:**
- `tests/integration/__init__.py` — empty
- `tests/integration/conftest.py` — all infrastructure from §4
- `tests/integration/pdfs/simple_one_page.pdf` — generated via `fpdf2`

**Dev dependency added:** `fpdf2` (for PDF generation script or test-time generation)

**Verification:** `uv run pytest tests/integration/ --collect-only` loads clean with no errors.

---

### Phase 2: Parse Tests

**Goal:** Data-driven tests for `DigitalPDFParser.parse()` with real PDFs.

**Files created:**
- `tests/integration/test_cases.py` — starts with `PARSE_CASES`
- `tests/integration/test_parse.py` — parametrized parse tests

**Test logic:**
1. Instantiate `DigitalPDFParser()`
2. Create `PDFDocument(uri=pdf_path(case["pdf"]))`
3. Call `parser.parse(doc)`
4. Assert page count, line count, line text content per `case["expected"]`

**Verification:** `uv run pytest tests/integration/test_parse.py -v` — green.

---

### Phase 3: Format Tests

**Goal:** Data-driven tests for `DigitalPDFFormatter.format_document_for_llm()`.

**Files modified/created:**
- `tests/integration/test_cases.py` — add `FORMAT_CASES`
- `tests/integration/test_format.py` — parametrized format tests

**Test logic:**
1. Parse the PDF with `DigitalPDFParser`
2. Set `doc.include_citations` from the case dict
3. Call `DigitalPDFFormatter().format_document_for_llm(doc)`
4. Assert `contains` / `not_contains` substrings

**Verification:** `uv run pytest tests/integration/test_format.py -v` — green.

---

### Phase 4: Extract Tests (Mocked LLM)

**Goal:** Full pipeline extraction with `FakeLLM`.

**Files modified/created:**
- `tests/integration/test_cases.py` — add `EXTRACT_CASES`
- `tests/integration/test_extract.py` — parametrized extraction tests

**Test logic:**
1. Look up schema from `SCHEMA_REGISTRY` via `case["schema"]`
2. Build `DocumentProcessor.from_digital_pdf(llm=FakeLLM(text_response=case["mock_llm_response"]))`
3. Call `processor.extract(uri, schema, **case["config"])`
4. Assert `result.data.model_dump() == case["expected_data"]`

**Verification:** `uv run pytest tests/integration/test_extract.py -v` — green.

---

### Phase 5: Live LLM Tests

**Goal:** Real LLM extraction tests behind `--run-live` flag.

**Files modified:**
- `tests/integration/test_cases.py` — add `LIVE_EXTRACT_CASES`
- `tests/integration/test_extract.py` — add live-mode parametrized test function

**Test logic:**
1. `@pytest.mark.live` + `@pytest.mark.parametrize`
2. Use `PDFProcessor(provider=case["provider"])`
3. Call `processor.extract(uri, schema, **case["config"])`
4. Assert `result.data.model_dump() == case["expected_data"]`

**Verification:** `uv run pytest tests/integration/ -v --run-live` — live tests execute with API key.

---

### Phase 6: QA

**Goal:** Full quality pass, no regressions.

**Steps:**
1. `uv run pytest tests/ -v` — all tests pass (integration + existing unit tests)
2. `uv run ruff check . && uv run ruff format --check .` — clean
3. `uv run pyrefly check .` — clean
4. Verify adding a new dict to `PARSE_CASES` auto-discovers without code changes

---

## 7. Adding New Test Cases

To add a new test case:

1. Add a dict to the appropriate list in `test_cases.py` (`PARSE_CASES`, `FORMAT_CASES`, `EXTRACT_CASES`, or `LIVE_EXTRACT_CASES`).
2. If needed, place a new PDF in `tests/integration/pdfs/`.
3. If needed, add a new Pydantic model to `tests/conftest.py` and register it in `SCHEMA_REGISTRY`.
4. No changes to test files — `@pytest.mark.parametrize` picks up new cases automatically.

---

## 8. Running Integration Tests

```bash
# All integration tests (mocked LLM, no API keys needed)
uv run pytest tests/integration/ -v

# Include live LLM tests (requires API keys)
uv run pytest tests/integration/ -v --run-live

# Only parse tests
uv run pytest tests/integration/test_parse.py -v

# Only live tests
uv run pytest tests/integration/ -v --run-live -m live
```
