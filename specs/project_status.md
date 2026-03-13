# Project Status

> Tracks implementation progress against the [Engineering Design](engineering_design.md) and [PRD](prd.md).
> Updated: 2026-03-12

---

## Phase Summary

| Phase | Scope | Status |
|-------|-------|--------|
| Baseline | Digital PDF parsing, single-pass extraction, citation enrichment | Done |
| Phase 1 | Multi-pass extraction, pydantic-settings config, restriction validators | Done |
| Phase 2 | Multi-LLM providers (OpenAI, Ollama, Anthropic, Gemini) | Done |
| Phase 3 | Scanned PDF support (OCR pipeline) | Not started |
| Phase 4 | Batch and async processing | Not started |

---

## Baseline — Core Architecture

- [x] `DigitalPDFParser` — parse text-based PDFs into `PDFDocument`
- [x] `DigitalPDFFormatter` — format pages for LLM with/without line numbers
- [x] `DigitalPDFExtractor` — single-pass extraction with citation enrichment
- [x] `DocumentProcessor` — orchestrates parse → format → extract
- [x] Schemas: `BoundingBox`, `Line`, `Page`, `PDF`, `PDFDocument`, `Document`
- [x] `pydantic_to_json_instance_schema` — schema generation for LLM prompts
- [x] Bounding box normalization utilities
- [x] Citation stripping utilities

## Phase 1 — Multi-pass Extraction + Restrictions

- [x] `DocIntelligenceConfig` (pydantic-settings with `DOC_INTEL_` env prefix)
- [x] `restrictions.py` — `check_pdf_size`, `check_page_count`, `check_schema_depth`
- [x] Three-pass extraction in `DigitalPDFExtractor`:
  - [x] Pass 1 — raw extraction (no citations)
  - [x] Pass 2 — page grounding (field → page mapping)
  - [x] Pass 3 — line grounding on relevant pages only
- [x] `PDFDocument.pass1_result` and `pass2_page_map` intermediate storage
- [x] `PDFExtractionConfig` schema with `page_numbers` support
- [x] `DocumentProcessor.extract()` runs restriction checks before LLM calls

## Phase 2 — Multi-LLM Provider Support

- [x] `OpenAILLM` — Responses API (`client.responses.create`)
- [x] `OllamaLLM` — Chat Completions API (`client.chat.completions.create`)
- [x] `AnthropicLLM` — Messages API (optional dependency)
- [x] `GeminiLLM` — Google GenAI SDK (optional dependency)
- [x] `BaseLLM.generate_structured_output` made non-abstract (raises `NotImplementedError`)
- [x] Optional extras in `pyproject.toml` (`anthropic`, `gemini`, `ocr`)
- [x] Integration test suite (`tests_integration/`) with dynamic Ollama model discovery

## Phase 3 — Scanned PDF Support (OCR Pipeline)

- [ ] `BaseLayoutDetector` and `BaseOCREngine` abstract bases
- [ ] `PaddleLayoutDetector` — PPStructure layout detection
- [ ] `PaddleOCREngine` — PaddleOCR text recognition
- [ ] `ScannedPDFParser` — image → layout → OCR → `PDFDocument`
- [ ] Async OCR regions within page via `asyncio.gather`
- [ ] `DocumentProcessor.from_scanned_pdf()` factory method
- [ ] Tests for OCR base classes, Paddle implementations, scanned parser

## Phase 4 — Batch and Async Processing

- [ ] `BaseLLM.agenerate_text()` with default thread-pool implementation
- [ ] Native async overrides in provider LLMs
- [ ] `AsyncDocumentProcessor` — async parse, extract, factory methods
- [ ] `batch_extract()` — concurrent multi-document processing with semaphore
- [ ] Progress callbacks (`queued → parsing → extracting → done | failed`)
- [ ] Concurrency config: `max_concurrent_documents`, `max_concurrent_pages`, `max_concurrent_regions`
- [ ] Tests for async processor and batch API
