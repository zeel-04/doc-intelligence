# Project Status

> Tracks implementation progress against the [Engineering Design](engineering_design.md) and [PRD](prd.md).
> Updated: 2026-03-29

---

## Phase Summary

| Phase | Scope | Status |
|-------|-------|--------|
| Baseline | Digital PDF parsing, single-pass extraction, citation enrichment | Done |
| Phase 1 | Multi-pass extraction, pydantic-settings config, restriction validators | Done |
| Phase 2 | Multi-LLM providers (OpenAI, Ollama, Anthropic, Gemini) | Done |
| Client API Redesign | Decouple processor, three-layer API, `create_llm()` factory | Done |
| Phase 3 | Scanned PDF support (OCR pipeline) | Not started |
| Phase 4 | Batch and async processing | Not started |

---

## Baseline — Core Architecture

- [x] `DigitalPDFParser` — parse text-based PDFs into `PDFDocument`
- [x] `PDFFormatter` — format pages for LLM with/without line numbers
- [x] `PDFExtractor` — single-pass extraction with citation enrichment
- [x] `DocumentProcessor` — orchestrates parse → format → extract
- [x] Schemas: `BoundingBox`, `Line`, `Page`, `PDF`, `PDFDocument`, `Document`
- [x] `pydantic_to_json_instance_schema` — schema generation for LLM prompts
- [x] Bounding box normalization utilities
- [x] Citation stripping utilities

## Phase 1 — Multi-pass Extraction + Restrictions

- [x] `DocIntelligenceConfig` (pydantic-settings with `DOC_INTEL_` env prefix)
- [x] `restrictions.py` — `check_pdf_size`, `check_page_count`, `check_schema_depth`
- [x] Three-pass extraction in `PDFExtractor`:
  - [x] Pass 1 — raw extraction (no citations)
  - [x] Pass 2 — page grounding (field → page mapping)
  - [x] Pass 3 — line grounding on relevant pages only
- [x] `PDFDocument.pass1_result` and `pass2_page_map` intermediate storage
- [x] `PDFExtractionConfig` schema with `page_numbers` support
- [x] `DocumentProcessor.extract()` runs restriction checks before LLM calls

## Phase 2 — Multi-LLM Provider Support

- [x] `OpenAILLM` — Responses API (`client.responses.create`)
- [x] `OllamaLLM` — Native Ollama SDK (`ollama.Client.chat`)
- [x] `AnthropicLLM` — Messages API (optional dependency)
- [x] `GeminiLLM` — Google GenAI SDK (optional dependency)
- [x] `BaseLLM.generate_structured_output` made non-abstract (raises `NotImplementedError`)
- [x] Optional extras in `pyproject.toml` (`anthropic`, `gemini`, `ocr`)
- [x] Integration test suite (`tests_integration/`) with dynamic Ollama model discovery
- [x] `_LLM_REGISTRY` and `create_llm()` factory function

## Client API Redesign

- [x] `DocumentProcessor` decoupled from document (no `document` in `__init__`)
- [x] `DocumentProcessor.extract()` takes `uri`, `response_format`, keyword-only options
- [x] `PDFProcessor` convenience wrapper (`provider` + `model` or `llm` instance)
- [x] Top-level `extract()` one-liner in `doc_intelligence/extract.py`
- [x] `create_llm()` registry-based factory in `doc_intelligence/llm.py`
- [x] `ExtractionResult` typed return (`.data` / `.metadata`)
- [x] `BaseLLM` stores per-provider default `model`; per-call override via kwargs
- [x] Removed dead `response` and `response_metadata` fields from `Document`
- [x] Removed cross-provider `default_llm_model` bug from `config.py`
- [x] Public API exported via `doc_intelligence/__init__.py` with `__all__`
- [x] `tests/test_init.py` — public API contract tests

## Phase 3 — Scanned PDF Support (OCR Pipeline)

- [x] `BaseLayoutDetector` and `BaseOCREngine` abstract bases
- [x] `ScannedPDFParser` — image → layout → OCR → `PDFDocument`
- [x] Async OCR regions within page via `asyncio.gather`
- [x] `DocumentProcessor.from_scanned_pdf()` factory method (requires user-supplied detector/engine)
- [x] Tests for OCR base classes and scanned parser

## Phase 4 — Batch and Async Processing

- [ ] `BaseLLM.agenerate_text()` with default thread-pool implementation
- [ ] Native async overrides in provider LLMs
- [ ] `AsyncDocumentProcessor` — async parse, extract, factory methods
- [ ] `batch_extract()` — concurrent multi-document processing with semaphore
- [ ] Progress callbacks (`queued → parsing → extracting → done | failed`)
- [ ] Concurrency config: `max_concurrent_documents`, `max_concurrent_pages`, `max_concurrent_regions`
- [ ] Tests for async processor and batch API
