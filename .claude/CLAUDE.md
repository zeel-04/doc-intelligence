# CLAUDE.md — doc_intelligence

## Project Overview

`doc_intelligence` is a library for AI-powered document processing pipelines. It parses PDFs, formats content for LLMs, and extracts structured data using Pydantic models with optional citation support.

## Package Structure

```
doc_intelligence/
├── pdf/                    # All PDF-specific logic
│   ├── parser.py           # PDFParser (abstract), DigitalPDFParser — parse PDFs into structured PDFDocument objects
│   ├── extractor.py        # PDFExtractor — single-pass & multi-pass extraction with citation enrichment
│   ├── formatter.py        # PDFFormatter — format PDF content for LLM consumption with line numbers
│   ├── processor.py        # DocumentProcessor (reusable pipeline), PDFProcessor (convenience wrapper with LLM factory)
│   ├── schemas.py          # PDF, PDFDocument, PDFExtractionConfig — PDF-specific document models (block types and Page live in schemas/core.py)
│   ├── types.py            # PDFExtractionMode enum (SINGLE_PASS, MULTI_PASS)
│   └── utils.py            # enrich_citations_with_bboxes() — add bounding boxes to citation metadata
├── schemas/
│   └── core.py             # BoundingBox, Line, Cell, TextBlock, TableBlock, ImageBlock, ChartBlock, ContentBlock, Page, BaseCitation, Document, ExtractionResult, ExtractionConfig, PydanticModel TypeVar
├── base.py                 # Abstract bases: BaseParser, BaseFormatter, BaseLLM, BaseExtractor
├── llm.py                  # OpenAILLM, OllamaLLM, AnthropicLLM, GeminiLLM + create_llm() factory
├── extract.py              # extract() — top-level one-liner convenience function wrapping PDFProcessor
├── restrictions.py         # check_pdf_size(), check_page_count(), check_schema_depth() — hard-limit validators
├── config.py               # DocIntelligenceConfig — per-provider default models, size/page/depth limits, async settings
├── utils.py                # normalize_bounding_box, strip_citations — bbox transforms and citation stripping
└── pydantic_to_json_instance_schema.py  # Convert Pydantic models to JSON instance schemas with citation wrappers

tests/                      # Mirrors doc_intelligence/ structure exactly
├── pdf/
│   ├── test_parser.py
│   ├── test_extractor.py
│   ├── test_formatter.py
│   ├── test_processor.py
│   ├── test_schemas.py
│   ├── test_types.py
│   └── test_utils.py
├── schemas/
│   └── test_core.py
├── conftest.py             # Shared fixtures: FakeLLM, FakeParser, FakeExtractor, sample_pdf, etc.
├── test_base.py
├── test_llm.py
├── test_extract.py
├── test_restrictions.py
├── test_utils.py
└── test_pydantic_to_json_instance_schema.py
```

**Rule:** New document types (e.g. `docx/`, `image/`) get their own folder under `doc_intelligence/` with the same internal layout as `pdf/`.

## Public API

Everything in `__all__` from `doc_intelligence/__init__.py` is the public surface. Currently:

- **One-liner:** `extract()`
- **Processors:** `PDFProcessor`, `DocumentProcessor`
- **LLMs:** `create_llm()`, `BaseLLM`, `OpenAILLM`, `OllamaLLM`, `AnthropicLLM`, `GeminiLLM`
- **PDF types:** `PDFDocument`, `PDFExtractionMode`, `PDFExtractionConfig`
- **Primitives:** `ExtractionResult`, `BoundingBox`, `BaseCitation`

Everything else is internal. Users should never import from submodules directly (e.g. `from doc_intelligence.pdf.parser import ...`).

## Specs

`specs/prd.md` and `specs/engineering_design.md` are the source of truth for planned features and architecture decisions. Consult them before starting new feature work.

## Workflow

1. **Plan first.** Before writing code, outline the steps. Ask for clarification on ambiguous trade-offs.
2. **One step at a time.** Implement and verify each step by writing relevant test if not already, before moving on.
3. **Verify after every step.** Run `uv run pytest tests/` and `ruff check` + `ruff format --check` after each change.
4. **Never leave the codebase broken** between steps.
5. **Bump spec versions after every spec change** After any modification to the codebase if it requires to change plan, then, increment the minor version (e.g. `1.1` → `1.2`) in both `specs/prd.md` and `specs/engineering_design.md`. Do this automatically — never wait to be asked.
6. **Ping me if CLAUDE.md needs to be updated** If during code modifications/planning if there's anything worth adding/updating/deleting from CLAUDE.md then propose me with a summarized pros & cons.

## Git Conventions

- **Branch naming:** `feature/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`.
- **Commit messages:** Imperative mood, concise first line (e.g. "Add multi-pass extraction for tables"). Body for context if needed.
- **PRs:** Target `main` from feature branches. One logical change per PR.

## Dependencies

- Minimize runtime dependencies. New deps require justification.
- `uv add <pkg>` for runtime, `uv add --dev <pkg>` for dev-only.
- Pin exact versions for runtime deps in `pyproject.toml`.

## Tooling

| Tool | Purpose |
|---|---|
| `uv` | Dependency management |
| `ruff` | Linting and formatting |
| `pyrefly` | Type checking |
| `pytest` | Testing |
| `mkdocs-material` | Documentation |
| `asyncio` | Async support for LLM calls and processing |
| `loguru` | Logging |
| `tenacity` | Retry logic for LLM/network calls |
| `pydantic` | Data validation and modeling |
| `pydantic-settings` | Configuration management |

```bash
uv run pytest tests/          # run all tests
uv run pytest tests/pdf/      # run pdf tests only
uv run ruff check .           # lint
uv run ruff format --check .  # format check
uv run pyrefly check .        # type checking
```

- Use **Context7 MCP** to fetch current library/framework documentation instead of relying on training data.

## Coding Standards

### SOLID Principles

- **Single Responsibility:** Each module/class has one job — parser parses, formatter formats, extractor extracts, processor orchestrates. Never mix concerns across these boundaries.
- **Open/Closed:** Extend via subclassing `BaseParser`, `BaseFormatter`, `BaseLLM`, `BaseExtractor` in `base.py`. New LLM providers go in `_LLM_REGISTRY` in `llm.py`. Never add `if doc_type == ...` conditionals to existing implementations.
- **Liskov Substitution:** All subclasses must match their base interface exactly — same signatures, same return types, same exception semantics. `generate_structured_output()` on `BaseLLM` is opt-in (raises `NotImplementedError` by default).
- **Interface Segregation:** Abstract bases stay minimal — one abstract method each (`parse`, `format_document_for_llm`, `generate_text`, `extract`). Don't force methods that subclasses don't need.
- **Dependency Inversion:** `DocumentProcessor` accepts abstract types via constructor injection. Concrete wiring happens at call sites or in factory methods like `from_digital_pdf()` and `create_llm()`. `PDFProcessor` is a convenience wrapper that delegates to `DocumentProcessor`.

### Type Annotations

- All function signatures must have full type annotations (parameters and return types).
- Modern union syntax: `str | None`, `int | float` (not `Optional`, `Union`).
- Lowercase generics: `list[str]`, `dict[str, Any]`, `type[PydanticModel]`.
- Docstrings: Google-style with Args/Returns/Raises for all public functions.

### Error Handling

- Use `loguru` for logging — never `print()` or stdlib `logging`.
- Use stdlib exceptions (`ValueError`, `TypeError`) with descriptive messages. No custom exception hierarchy for now — revisit if users need granular error handling.
- Use `tenacity` for retry logic on LLM/network calls.

### Configuration

`DocIntelligenceConfig` in `config.py` uses `pydantic-settings`. All settings are overridable via `DOC_INTEL_*` env vars or `.env` file. The module-level `settings` singleton is the single source of truth — import and use it, don't instantiate your own.

### Citation Architecture

Citations are block-level: `{"page": <int>, "blocks": [<int>]}`. `ContentBlock` (discriminated union in `schemas/core.py`) is the universal citable unit. For digital PDFs, each text line becomes its own `TextBlock`, so block-level addressing gives line-level precision. `ImageBlock` and `ChartBlock` are excluded from formatter output and block index numbering — they are placeholders for future VLM support.

### General Rules

- Always use absolute imports.
- Update @docs/ and @README.md if API level changes are made.
- Use proper docstrings
  - Functions and classes where required
  - Module level

## Testing Conventions

- **Mirror structure:** every `doc_intelligence/foo/bar.py` has a corresponding `tests/foo/test_bar.py`.
- **Group tests into classes:** `TestDigitalPDFParser`, `TestExtractSinglePass`, etc.
- **Section separators** between test classes:
  ```python
  # ---------------------------------------------------------------------------
  # DigitalPDFParser
  # ---------------------------------------------------------------------------
  class TestDigitalPDFParser:
  ```
- **One behavior per test**, named descriptively: `test_bboxes_are_normalized`, `test_http_error_propagates`.
- Use `@pytest.mark.parametrize` for data-driven cases.
- Use `pytest.raises` for expected exceptions.
- Absolute imports: `from doc_intelligence.pdf.parser import DigitalPDFParser`.
- Shared fixtures live in `tests/conftest.py` (`FakeLLM`, `FakeParser`, `sample_pdf`, `sample_pdf_document`, etc.).
- Target **100% coverage** — every public function, branch, and edge case must be tested.
- Add module-level docstring: `"""Tests for pdf.parser module."""`
- **Async tests:** Use `pytest-asyncio` with `@pytest.mark.asyncio` for async code paths. Keep async fixtures in `conftest.py` alongside sync ones.

## Integration & E2E Testing

Integration and end-to-end tests live in `tests/integration/` and are **data-driven** — test cases are Python dicts, not inline assertions.

### Structure

```
tests/integration/
├── conftest.py          # --run-live flag, markers, pdf_path fixture, schema registry
├── test_cases.py        # All test case dicts: PARSE_CASES, FORMAT_CASES, EXTRACT_CASES
├── pdfs/                # Real test PDF files (<100KB each)
├── test_parse.py        # Parse-only tests
├── test_format.py       # Format-only tests
└── test_extract.py      # Full extraction pipeline tests
```

### Adding a Test Case

1. Add a dict to the appropriate list in `test_cases.py` (`PARSE_CASES`, `FORMAT_CASES`, or `EXTRACT_CASES`).
2. Place any new PDF in `tests/integration/pdfs/`.
3. No code changes needed in test files — `@pytest.mark.parametrize` picks up new cases automatically.

### Mocked vs Live LLM

- **Default:** `FakeLLM` with `mock_llm_response` from the test case dict. No API keys needed.
- **`--run-live` flag:** Uses real LLM via `create_llm()`. Requires API keys. Tests marked `@pytest.mark.live` are skipped without this flag.

```bash
uv run pytest tests/integration/ -v              # mocked (default)
uv run pytest tests/integration/ -v --run-live   # real LLM calls
```

### Assertions

- **Exact match** by default: `result.data.model_dump() == case["expected_data"]`.
- Parse tests assert page count, line count, and line text content.
- Format tests assert output string contains expected substrings.
