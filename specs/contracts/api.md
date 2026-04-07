# API Design — doc_intelligence

---

## 1. Design Philosophy

The API follows an **opinionated split** between configuration and invocation:

- **Constructor = pipeline configuration.** LLM provider, model, extraction mode, citation preferences, parsing strategy, and generation parameters are all fixed when you create a processor. They define _how_ the pipeline behaves.
- **`processor.extract()` = per-document intent.** It accepts only the document URI, the output schema, and an optional page filter. These define _what_ to extract from _which_ document.
- **One processor, one configuration.** If you need different settings (e.g. multi-pass for contracts, single-pass for invoices), create a separate processor. There are no per-call overrides.

This design eliminates merge logic, sentinel handling, and ambiguity about which value wins. It also makes processors safe to reuse across calls — the LLM client and pipeline components are initialized once.

---

## 2. Public Surface

Everything in `__all__` from `doc_intelligence/__init__.py`. Users should only import from the top-level package.

| Category | Symbols |
|---|---|
| **Processors** | `PDFProcessor`, `DocumentProcessor` |
| **LLMs** | `create_llm`, `BaseLLM`, `OpenAILLM`, `OllamaLLM`, `AnthropicLLM`, `GeminiLLM` |
| **PDF types** | `PDFDocument`, `PDFExtractionMode`, `PDFExtractionRequest`, `PDFParser`, `ParseStrategy` |
| **Result & primitives** | `ExtractionResult`, `BoundingBox`, `BaseCitation` |
| **OCR hooks** | `BaseLayoutDetector`, `BaseOCREngine`, `LayoutRegion` |

Everything else is internal. Direct submodule imports (e.g. `from doc_intelligence.pdf.parser import ...`) are unsupported.

---

## 3. Entry Points

### 3.1 Processor: `PDFProcessor`

For production use — create once, extract many. Reuses the LLM client across calls.

**Constructor:**
```python
PDFProcessor(
    llm: BaseLLM | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    strategy: ParseStrategy = ParseStrategy.DIGITAL,
    include_citations: bool = True,
    extraction_mode: PDFExtractionMode = PDFExtractionMode.SINGLE_PASS,
    llm_config: dict[str, Any] | None = None,
    layout_detector: BaseLayoutDetector | None = None,
    ocr_engine: BaseOCREngine | None = None,
    dpi: int = 150,
)
```

LLM identity is provided via one of two paths:
- `llm` — pass a pre-built `BaseLLM` instance (useful for testing or sharing a client).
- `provider` + `model` — delegates to `create_llm()`. If `model` is `None`, the provider's default from `DocIntelligenceConfig` is used.

Exactly one of `llm` or `provider` must be specified. Neither → `ValueError`.

**extract():**
```python
def extract(
    self,
    uri: str,
    response_format: type[PydanticModel],
    *,
    page_numbers: list[int] | None = None,
) -> ExtractionResult
```

`response_format` is required and positional. It is always per-call because different documents typically need different schemas. `page_numbers` is optional (0-indexed, defaults to all pages).

**Usage:**
```python
from doc_intelligence import PDFProcessor, PDFExtractionMode

processor = PDFProcessor(
    provider="openai",
    model="gpt-4o",
    extraction_mode=PDFExtractionMode.MULTI_PASS,
)

r1 = processor.extract("jan.pdf", InvoiceSchema)
r2 = processor.extract("feb.pdf", InvoiceSchema)
r3 = processor.extract("receipt.pdf", ReceiptSchema)
```

### 3.2 Components: `DocumentProcessor` + Base Classes

For advanced use — build a custom pipeline from individual components.

```python
DocumentProcessor(
    parser: BaseParser,
    formatter: BaseFormatter,
    extractor: BaseExtractor,
)
```

`DocumentProcessor` is document-type agnostic. It orchestrates `parse → extract` and knows nothing about PDFs. Users wire their own parser, formatter, and extractor implementations. `PDFProcessor` is a convenience wrapper that wires the PDF-specific components and delegates to `DocumentProcessor`.

```python
def extract(self, request: ExtractionRequest) -> ExtractionResult
```

Takes a fully-built `ExtractionRequest` (or `PDFExtractionRequest`). No restrictions are applied — that is `PDFProcessor`'s responsibility.

---

## 4. Parameter Placement Rules

| Concern | Where | Rationale |
|---|---|---|
| LLM identity (`provider`, `model`, `llm`) | Constructor | Provider and model define the pipeline — they don't change per document |
| Parsing strategy (`strategy`, `layout_detector`, `ocr_engine`, `dpi`) | Constructor | A processor is configured for one document type (digital or scanned) |
| Extraction behavior (`include_citations`, `extraction_mode`) | Constructor | Behavioral policy, not document-specific intent |
| Generation parameters (`llm_config`) | Constructor | Temperature, max tokens, etc. are pipeline tuning, not per-document |
| Document URI (`uri`) | `extract()` | Changes every call |
| Output schema (`response_format`) | `extract()` | Different documents may need different schemas |
| Page filter (`page_numbers`) | `extract()` | Document-specific scoping |

**Decision rule:** If a parameter answers "how should this pipeline behave?", it goes on the constructor. If it answers "what should I extract from this document?", it goes on `extract()`.

---

## 5. Type Contracts

### 5.1 Input Types

| Parameter | Type | Notes |
|---|---|---|
| `uri` | `str` | Local file path or URL. Non-local URIs skip size/page restrictions. |
| `response_format` | `type[PydanticModel]` | Must be a `BaseModel` subclass. Validated at call time — `ValueError` if not. |
| `provider` | `str` | Must match a key in `_LLM_REGISTRY` (`"openai"`, `"ollama"`, `"anthropic"`, `"gemini"`). |
| `model` | `str \| None` | Provider-specific model identifier. `None` → provider default from config. |
| `llm_config` | `dict[str, Any] \| None` | Forwarded as `**kwargs` to `BaseLLM.generate()`. A `"model"` key here overrides the constructor `model` per LLM call. |
| `page_numbers` | `list[int] \| None` | 0-indexed. `None` → all pages. |

### 5.2 Return Types

`PDFProcessor.extract()` returns `ExtractionResult`:

```python
class ExtractionResult(BaseModel):
    data: Any          # Populated instance of the response_format model
    metadata: dict[str, Any] | None  # Citation dict, or None if citations disabled
```

`data` is typed as `Any` at the schema level but will always be an instance of the `response_format` class passed to `extract()`.

`metadata` contains citation information when `include_citations=True`. Structure: field paths as keys, `{"page": int, "blocks": [int]}` citations as values. When citation enrichment runs, block indices are resolved to bounding boxes.

### 5.3 Error Types

No custom exception hierarchy. All errors are stdlib exceptions.

| Error | When | Source |
|---|---|---|
| `ValueError("Either \`llm\` or \`provider\` must be specified")` | Neither LLM path provided | `PDFProcessor.__init__` |
| `ValueError("response_format must be a Pydantic model")` | `response_format` is not a `BaseModel` subclass | `PDFProcessor.extract` |
| `ValueError("PDF exceeds max size ...")` | File too large | `check_pdf_size` |
| `ValueError("PDF has N pages, limit is M")` | Too many pages | `check_page_count` |
| `ValueError("Schema depth N exceeds limit M")` | Pydantic model too deeply nested | `check_schema_depth` |
| Provider-specific exceptions | Network errors, auth failures, rate limits | `OpenAILLM`, `AnthropicLLM`, etc. (retried via `tenacity`) |

---

## 6. Configuration Precedence

Settings resolve in this order (highest priority first):

```
1. Explicit constructor argument (e.g. PDFProcessor(model="gpt-4o"))
2. llm_config dict (e.g. llm_config={"model": "gpt-4o-mini"} overrides at LLM call time)
3. DocIntelligenceConfig / environment variable (e.g. DOC_INTEL_OPENAI_DEFAULT_MODEL)
4. Hardcoded default in DocIntelligenceConfig
```

For LLM model specifically: `model` on the constructor sets the model at `BaseLLM` initialization. A `"model"` key inside `llm_config` overrides it per `generate()` call (popped from kwargs by the LLM implementation). Both are set once at construction — there is no per-`extract()` override.

Restriction thresholds (`max_pdf_size_mb`, `max_pdf_pages`, `max_schema_depth`) are read from the `settings` singleton at call time. They are not exposed on `PDFProcessor`'s constructor — override via env vars or `.env` file.

---

## 7. Stability & Versioning

- **Stable:** Everything in `__all__`. These symbols, their signatures, and their return types are the public contract.
- **Internal:** Anything imported from a submodule directly (e.g. `doc_intelligence.pdf.extractor`). May change without notice.
- **Breaking changes:** Any change to a stable symbol's signature, return type, or error behavior is a breaking change and requires a major version bump.
- **Additions:** New symbols in `__all__`, new optional parameters on existing functions, and new enum members are non-breaking.
