# CLAUDE.md — doc_intelligence

## Project Overview

`doc_intelligence` is a library for AI-powered document processing pipelines. It parses PDFs, formats content for LLMs, and extracts structured data using Pydantic models with optional citation support.

## Package Structure

```
doc_intelligence/
├── pdf/                    # All PDF-specific logic
│   ├── parser.py           # PDFParser (abstract), DigitalPDFParser — parse PDFs into structured PDFDocument objects
│   ├── extractor.py        # DigitalPDFExtractor — single-pass & multi-pass extraction with citation enrichment
│   ├── formatter.py        # DigitalPDFFormatter — format PDF content for LLM consumption with line numbers
│   ├── processor.py        # DocumentProcessor (reusable pipeline), PDFProcessor (convenience wrapper with LLM factory)
│   ├── schemas.py          # Line, Page, PDF, PDFDocument, PDFExtractionConfig — PDF structure models
│   ├── types.py            # PDFExtractionMode enum (SINGLE_PASS, MULTI_PASS)
│   └── utils.py            # enrich_citations_with_bboxes() — add bounding boxes to citation metadata
├── schemas/
│   └── core.py             # BoundingBox, BaseCitation, Document, ExtractionResult, ExtractionConfig, PydanticModel TypeVar
├── base.py                 # Abstract bases: BaseParser, BaseFormatter, BaseLLM, BaseExtractor
├── llm.py                  # OpenAILLM, OllamaLLM, AnthropicLLM, GeminiLLM + create_llm() factory
├── extract.py              # extract() — top-level one-liner convenience function wrapping PDFProcessor
├── restrictions.py         # check_pdf_size(), check_page_count(), check_schema_depth() — hard-limit validators
├── config.py               # DocIntelligenceConfig — per-provider default models, size/page/depth limits, async settings
├── utils.py                # normalize/denormalize bounding boxes, strip_citations, find_citation_fields, etc.
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

## Workflow

1. **Plan first.** Before writing code, outline the steps. Ask for clarification on ambiguous trade-offs.
2. **One step at a time.** Implement and verify each step by writing relevant test if not already, before moving on.
3. **Verify after every step.** Run `uv run pytest tests/` and `ruff check` + `ruff format --check` after each change.
4. **Never leave the codebase broken** between steps.
5. **Bump spec versions after every spec change** After any modification to the codebase if it requires to change plan, then, increment the minor version (e.g. `1.1` → `1.2`) in both `specs/prd.md` and `specs/engineering_design.md`. Do this automatically — never wait to be asked.
6. **Ping me if CLAUDE.md needs to be updated** If during code modifications/planning if there's anything worth adding/updating/deleting from CLAUDE.md then propose me with a summarized pros & cons.

## Tooling

| Tool | Purpose |
|---|---|
| `uv` | Dependency management |
| `hatchling` | Build backend |
| `ruff` | Linting and formatting |
| `pyrefly` | Type checking |
| `pytest` | Testing |
| `pre-commit` | Git hooks (ruff + pyrefly) |
| `mkdocs-material` | Documentation |

```bash
uv run pytest tests/          # run all tests
uv run pytest tests/pdf/      # run pdf tests only
uv run ruff check .           # lint
uv run ruff format --check .  # format check
uv run pyrefly check .        # type checking
```

- You can use context7 for getting up to date Documentation For libraries.

## Coding Standards

### SOLID Principles

- **Single Responsibility:** Each module, class, and function has one clear job. Parser parses, formatter formats, extractor extracts.
- **Open/Closed:** Add new document types by subclassing the abstract bases in `base.py`. Never add conditionals to existing implementations.
- **Liskov Substitution:** All subclasses of `BaseParser`, `BaseFormatter`, `BaseLLM`, `BaseExtractor` must honor the interface contracts exactly.
- **Interface Segregation:** Keep abstract interfaces minimal — don't force methods that aren't needed.
- **Dependency Inversion:** `DocumentProcessor` depends on abstract types; concrete implementations are injected at construction time.

### Type Annotations

- All function signatures must have full type annotations (parameters and return types).
- Modern union syntax: `str | None`, `int | float` (not `Optional`, `Union`).
- Lowercase generics: `list[str]`, `dict[str, Any]`, `type[PydanticModel]`.
- Docstrings: Google-style with Args/Returns/Raises for all public functions.

### Error Handling

- Use `loguru` for logging — never `print()` or stdlib `logging`.
- Raise `ValueError` / `TypeError` with descriptive messages.
- Use `tenacity` for retry logic on LLM/network calls.

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
