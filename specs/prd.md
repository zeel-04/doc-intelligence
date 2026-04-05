# Product Requirements Document — doc_intelligence

**Version:** 0.3.1
**Status:** Living document
**Last updated:** 2026-04-05

---

## 1. Overview

`doc_intelligence` is a Python library for extracting structured, typed data from documents using large language models. Developers describe the data they want as a Pydantic model; the library handles parsing the document, formatting it for the LLM, prompting, and returning a populated instance of that model — optionally with source citations pinpointing where every piece of data came from.

The library is designed to be embedded inside larger applications: pipelines, APIs, notebooks, or CLI tools. It is not an end-user product on its own, though a reference Streamlit frontend is included for demonstration and prototyping.

---

## 2. Problem Statement

Extracting structured information from documents — PDFs, scanned forms, invoices, statements, contracts — is a common need but hard to do reliably. Ad-hoc LLM prompting produces inconsistent formats. OCR tools produce text but no structure. Parsing libraries produce structure but no semantics. Existing document AI platforms are either cloud-locked, expensive, or require giving up raw control over the extraction schema.

Developers need a library that:

- Accepts any document and any output schema they define.
- Is transparent about where in the document each extracted value came from.
- Works with the LLM and infrastructure they already use.
- Handles the messy parts (chunking, formatting, retries, citation grounding) without requiring the developer to think about them.

---

## 3. Goals

- Provide a clean, composable Python API for document extraction that follows SOLID principles.
- Support multiple document formats (starting with PDFs, digital and scanned).
- Support multiple LLM providers through a single consistent interface.
- Deliver extraction results with optional, fine-grained source citations (page → block → bounding box).
- Be configurable enough for production use (limits, async, batching) without being over-engineered for simple cases.
- Remain framework-agnostic — no FastAPI, no Django, no mandatory cloud service.

## 3.1 Non-Goals

- This library does not build or host an API service.
- It does not store documents or extracted results (no database layer).
- It does not provide a pre-built UI (the Streamlit frontend is a demo only).
- It does not train or fine-tune models.
- It does not handle document authentication, access control, or billing.

---

## 4. User Personas

**The ML Engineer / Data Scientist**
Builds one-off extraction pipelines for internal datasets. Wants to iterate quickly in a notebook. Cares about accuracy and citation quality. May swap LLMs to compare results.

**The Backend Engineer**
Integrates extraction into a production service (invoices, contracts, forms). Needs reliability, limits to prevent abuse, async support for throughput, and a clear interface that doesn't break between versions.

**The Platform Engineer**
Runs extraction at scale across thousands of documents per day. Needs batch processing, async concurrency controls, observability hooks, and the ability to swap infrastructure (LLM provider, OCR engine) without rewriting application code.

---

## 5. Features

### 5.1 Digital PDF Extraction ✅ (Complete)

The library parses text-based (digital/native) PDFs and extracts structured data from them.

**How it works for the developer:**

```python
from doc_intelligence.pdf.processor import DocumentProcessor
from doc_intelligence.llm import OpenAILLM
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    date: str

processor = DocumentProcessor.from_digital_pdf(llm=OpenAILLM())
result = processor.extract(
    uri="invoice.pdf",
    response_format=Invoice,
    include_citations=True,
    llm_config={"model": "gpt-4o-mini"},
)
# result.data     → Invoice instance
# result.metadata → citation bounding boxes per field
```

**What happens under the hood:**

1. The PDF is parsed page by page; each text line becomes its own content block with a bounding box, normalized to a 0–1 coordinate scale.
2. The document is formatted as block-aware markup for the LLM (each block tagged with an index and type).
3. The LLM returns JSON matching the output schema (with optional block-level citation wrappers if citations are enabled).
4. Citations are enriched with bounding boxes by resolving block indices from the parser output.
5. The result is returned as a typed Pydantic instance plus raw citation metadata.

**Extraction modes:**

- *Single-pass:* One LLM call returns both the extracted data and its citations in a single response.
- *Multi-pass:* Three focused LLM calls for higher accuracy. See §5.2.

---

### 5.2 Multi-pass Extraction ✅ (Complete)

Single-pass extraction asks the LLM to do two things at once: understand the document and locate the source. For complex documents, this degrades accuracy. Multi-pass splits the job into three focused steps.

**Pass 1 — Data extraction (no grounding):**
The LLM receives the full document text and returns a clean Pydantic model instance with no citation fields. The schema it sees is the developer's schema exactly, with no citation wrappers added. The goal is pure accuracy — the LLM is not distracted by location.

**Pass 2 — Page grounding:**
The LLM receives the Pass 1 answer (as context) plus the full document and returns a mapping of field paths to the page numbers where the data appears. This step is intentionally coarse — pages are cheap to locate.

**Pass 3 — Block and bbox grounding:**
The library reduces the document to only the pages identified in Pass 2 and sends that smaller context to the LLM, asking for specific block indices. Bounding boxes are resolved deterministically from the parser output, not by the LLM.

**Intermediate results** are stored on the document object so callers can inspect or cache partial results if the pipeline is interrupted.

---

### 5.3 Extraction Limits and Restrictions ✅ (Complete)

The library enforces configurable hard limits to prevent runaway usage and to make extraction predictable in production.


| Limit                    | Default      | Description                                                          |
| ------------------------ | ------------ | -------------------------------------------------------------------- |
| Max PDF file size        | 10 MB        | PDFs above this size are rejected before parsing.                    |
| Max pages processed      | Configurable | PDFs with more pages than the limit are rejected with a `ValueError` before parsing begins. |
| Max schema nesting depth | 5            | Pydantic schemas deeper than this raise an error at extraction time. |


Limits are defined in a central configuration file (`config.py`) so they can be changed once for the whole application. Violations raise a descriptive `ValueError` before any LLM call is made, so they fail fast and cheaply.

---

### 5.4 Multiple LLM Providers ✅ (Complete)

The library ships with an OpenAI implementation. Phase 2 adds first-class support for three additional providers, all sharing the same interface.

**OpenAI** (existing) — Uses the Responses API with native structured output support.

**Ollama** — Connects to a locally-running Ollama server via the native Ollama Python SDK (`ollama.Client`). The native SDK is used instead of the OpenAI-compatible `/v1` endpoint because it correctly supports Ollama-specific parameters (e.g., `think=False`) that the `/v1` endpoint silently ignores. Any model that Ollama supports (Llama, Mistral, Qwen, etc.) works without provider-specific code.

**Anthropic (Claude)** — Uses Anthropic's Messages API. Structured output is achieved by including the JSON schema in the prompt and parsing the response, since Claude does not have a native structured-output API identical to OpenAI's. Retries on malformed JSON are handled automatically.

**Google Gemini** — Uses the Gemini API (or Vertex AI). Same JSON-in-prompt approach as Anthropic.

All providers implement the same two methods: `generate_text` (used by extractors) and `generate_structured_output` (available for direct use). Swapping providers requires only changing which LLM class is instantiated — the rest of the pipeline is unaffected.

---

### 5.5 Scanned PDF Support — OCR Pipeline (Phase 3)

Many real-world documents are scanned images embedded in PDF containers. pdfplumber cannot extract text from these. Phase 3 adds a full OCR pipeline that produces the same `PDF → Page → Line` data structure as the digital parser, so the existing formatter and extractor work without modification.

**The OCR pipeline has three steps:**

1. **Layout detection:** A user-supplied layout detector (implementing `BaseLayoutDetector`) runs on each page image and identifies distinct regions — paragraphs, tables, headers, figures. The library is designed so any layout model can be plugged in without changing the rest of the pipeline.
2. **OCR per region:** Each detected region is sent to a user-supplied OCR engine (implementing `BaseOCREngine`) to extract its text and character-level bounding boxes. Regions within a page are processed in parallel to minimize latency.
3. **Assembly:** The OCR results are assembled into the standard `PDF → Page → Line` schema. From this point forward, the scanned PDF goes through the same formatter and extractor as a digital PDF.

**Naming:** The digital parser remains `DigitalPDFParser`. The new OCR-based parser is `ScannedPDFParser`. Both produce `PDFDocument` objects; a `DocumentProcessor.from_scanned_pdf()` factory creates the appropriate pipeline.

**Async concurrency** within the OCR step (regions per page, pages per document) is controlled by a configuration block so teams can tune it to their hardware.

---

### 5.6 Batch and Async Processing (Phase 4)

For production use, processing documents one at a time synchronously is a bottleneck. Phase 4 makes the entire pipeline natively async and adds a batch API.

**Async pipeline:** All components (`AsyncDocumentProcessor`) expose `async` variants of `parse`, `format`, and `extract`. IO-bound operations (HTTP calls to LLMs, PDF downloads from URLs) are properly awaited. CPU-bound operations (PDF parsing, OCR) run in thread pools to avoid blocking the event loop.

**Batch API:** A `batch_extract` function accepts a list of documents (or document URIs) and an extraction config. It processes them concurrently up to a configurable `max_concurrent_documents` limit and returns results in the same order as the inputs.

**Progress callbacks:** A `on_progress` callback parameter lets callers track per-document status (`queued → parsing → extracting → done | failed`). This enables streaming progress to a UI or logging system.

**Configuration:** All concurrency limits live in the config file:

```python
# config.py — async concurrency limits (declared in Phase 1, used in Phase 4)
# Override via environment variables or .env:
#   DOC_INTEL_MAX_CONCURRENT_DOCUMENTS=10
#   DOC_INTEL_MAX_CONCURRENT_PAGES=4
#   DOC_INTEL_MAX_CONCURRENT_REGIONS=8
class DocIntelligenceConfig(BaseSettings):
    max_concurrent_documents: int = Field(default=10)
    max_concurrent_pages: int = Field(default=4)    # OCR only
    max_concurrent_regions: int = Field(default=8)  # OCR only
```

---

## 6. Non-Functional Requirements

**Reliability:**
LLM calls include automatic retries (up to 3 attempts via `tenacity`) with exponential backoff. Extraction failures return structured error information rather than raising unhandled exceptions.

**Observability:**
All significant events are logged with `loguru` at appropriate levels (DEBUG for LLM inputs/outputs, INFO for pipeline stages, ERROR for failures). No `print()` calls anywhere in the library.

**Correctness:**
Bounding boxes are always normalized to a 0–1 scale relative to page dimensions, making them usable regardless of the original PDF's coordinate system.

**Testability:**
All components depend on abstract base classes. Fake implementations (`FakeLLM`, `FakeParser`, etc.) are provided in the test suite. New features must reach 100% test coverage.

**Extensibility:**
New document types follow the same folder pattern (`doc_intelligence/<type>/`). New LLM providers subclass `BaseLLM`. New parsers subclass `BaseParser`. The core pipeline (`DocumentProcessor`) does not change when new implementations are added.

---

## 7. Future Considerations

The following are not planned for the current roadmap but inform design decisions to avoid dead ends.

**Third-party integrations:**

- *Landing AI* — plug in as an alternative layout detection / OCR backend.
- *Docling* — use as an alternative PDF parsing backend, particularly for complex table extraction.

These integrations will follow the same swappable-component pattern established in Phases 3 and 4: they become alternative implementations of `BaseParser` or a new `BaseOCREngine` abstraction, not special-cased additions to the existing code.

**Progress callbacks (async):** Already planned as a Phase 4 feature but may be simplified for an initial async release and enhanced later.

**Streaming extraction:** Long documents may benefit from streaming partial results as each page is processed. Not planned now, but the async architecture of Phase 4 should not preclude it.