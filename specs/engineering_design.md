# Engineering Design Document — doc_intelligence

**Version:** 0.1.9
**Status:** Living document
**Last updated:** 2026-03-29

---

## Baseline — Current Architecture (Complete)

Everything described in this section is already implemented and tested.

### Package layout

```
doc_intelligence/
├── pdf/
│   ├── parser.py       # PDFParser (ABC), DigitalPDFParser
│   ├── formatter.py    # DigitalPDFFormatter
│   ├── extractor.py    # DigitalPDFExtractor
│   ├── processor.py    # DocumentProcessor, PDFProcessor
│   ├── schemas.py      # PDF, Line, Page, PDFDocument, PDFExtractionConfig
│   ├── types.py        # PDFExtractionMode (SINGLE_PASS, MULTI_PASS)
│   └── utils.py        # enrich_citations_with_bboxes
├── schemas/
│   └── core.py         # Document, BoundingBox, BaseCitation, ExtractionConfig, ExtractionResult, PydanticModel TypeVar
├── base.py             # BaseParser, BaseFormatter, BaseLLM, BaseExtractor
├── llm.py              # OpenAILLM, OllamaLLM, AnthropicLLM, GeminiLLM, create_llm()
├── extract.py          # Top-level extract() convenience function
├── utils.py            # normalize_bounding_box, denormalize_bounding_box, strip_citations
├── pydantic_to_json_instance_schema.py
└── config.py           # pydantic-settings backed configuration
```

### Data flow (single-pass with citations)

```
PDF file / URL
    │
    ▼
DigitalPDFParser.parse()
    → PDFDocument(content=PDF(pages=[Page(lines=[Line(text, bbox)])]))
    │
    ▼
DigitalPDFFormatter.format_document_for_llm()
    → str: <page number=0>\n0: line text\n1: line text\n…\n</page>
    │
    ▼
DigitalPDFExtractor.extract()
    → calls llm.generate_text(system_prompt, user_prompt)
    → json_parser.parse(response)            # langchain JsonOutputParser
    → enrich_citations_with_bboxes(…)        # resolve line → bbox
    → strip_citations(…)                     # unwrap {value, citations} → plain value
    │
    ▼
ExtractionResult(data=<PydanticModel instance>, metadata=<citation dict>)
```

### Key design decisions

- `Document.content` is `BaseModel | None`. `PDFDocument` narrows it to `PDF | None`.
- `Document.extraction_mode` is declared on the base `Document` model; `PDFDocument` defaults it to `PDFExtractionMode.SINGLE_PASS`.
- All LLM interaction goes through `BaseLLM.generate_text`; `generate_structured_output` exists but is not used by the current extractor.
- Citation enrichment is a pure function over a dict — no mutation of the Pydantic model.
- `PDFExtractionMode.MULTI_PASS` is fully implemented via three-pass extraction in `DigitalPDFExtractor` (Phase 1).
- `config.py` uses `pydantic-settings` (`DocIntelligenceConfig`) with `DOC_INTEL_` env prefix (Phase 1).
- `BaseParser` is generic: `BaseParser(ABC, Generic[TDocument])` where `TDocument = TypeVar("TDocument", bound=Document)`. `PDFParser` narrows to `BaseParser[PDFDocument]`.
- `DocumentProcessor` is stateless w.r.t. documents — a fresh `PDFDocument` is created per `extract()` call (Client API Redesign).

---

## Phase 1 — Multi-pass Extraction + Restrictions (Complete)

### 1.1 Goals

- Implement the three-pass extraction flow for `PDFExtractionMode.MULTI_PASS`.
- Replace `config.py` with a validated, file-backed configuration model.
- Enforce hard limits (file size, page count, schema depth) before any expensive work.

---

### 1.2 Configuration overhaul

**Replace** the current `config.py` bare dict with a `pydantic-settings`-backed model. Settings are read from environment variables or a `.env` file, with sensible defaults hard-coded.

```python
# doc_intelligence/config.py  (new shape)
from pydantic import Field
from pydantic_settings import BaseSettings

class DocIntelligenceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOC_INTEL_", env_file=".env")

    # Per-provider default models — overridable via DOC_INTEL_<PROVIDER>_DEFAULT_MODEL env vars
    openai_default_model: str = Field(default="gpt-4o-mini")
    anthropic_default_model: str = Field(default="claude-sonnet-4-20250514")
    gemini_default_model: str = Field(default="gemini-2.0-flash")
    ollama_default_model: str = Field(default="llama3.2")

    # Limits
    max_pdf_size_mb: float = Field(default=10.0)
    max_pdf_pages: int = Field(default=100)
    max_schema_depth: int = Field(default=5)

    # Async (used in Phase 4, declared here for single source of truth)
    max_concurrent_documents: int = Field(default=10)
    max_concurrent_pages: int = Field(default=4)
    max_concurrent_regions: int = Field(default=8)

settings = DocIntelligenceConfig()   # singleton, imported everywhere
```

Environment variable names are `DOC_INTEL_MAX_PDF_SIZE_MB`, `DOC_INTEL_MAX_PDF_PAGES`, etc.

---

### 1.3 Restriction validators

A new `doc_intelligence/restrictions.py` module contains pure functions called at the start of the pipeline, before parsing or LLM calls.

```
check_pdf_size(uri: str, max_mb: float) -> None
    raises ValueError if file size > max_mb

check_page_count(page_count: int, max_pages: int) -> None
    raises ValueError if page_count > max_pages

check_schema_depth(model: type[BaseModel], max_depth: int, _current: int = 0) -> None
    raises ValueError if nesting depth > max_depth (recursive walk of model_fields)
```

`DocumentProcessor.extract()` calls all three checks at the top of the method before delegating to the extractor. This keeps the extractor itself free of limit-checking logic.

---

### 1.4 Multi-pass extraction

`PDFExtractionMode.MULTI_PASS` is implemented in `DigitalPDFExtractor`. The three passes are private methods; `extract()` orchestrates them.

#### Pass 1 — raw extraction

```
_extract_pass1(document, formatter, response_format, llm_config)
    → response_format instance (plain, no citations)
```

- The formatter is called with `include_citations=False` on the document (temporarily override `document.include_citations`).
- The LLM prompt uses `pydantic_to_json_instance_schema(response_format, citation=False)` so no citation wrappers appear in the schema.
- The JSON response is parsed and validated into `response_format(**response_dict)`.

#### Pass 2 — page grounding

```
_extract_pass2(document, formatter, pass1_result, llm_config)
    → dict[str, list[int]]   # field_path → [page_numbers]
```

- A new prompt template is used (different from the extraction prompt). It shows the full document text and the Pass 1 answer and asks: *"For each field in the answer, on which page(s) does the supporting text appear? Respond as JSON: {field_path: [page_numbers]}"*
- No Pydantic schema is needed for this pass; a plain JSON dict is parsed.

> **`field_path` grammar:** top-level fields use their name directly (e.g., `"vendor"`); nested fields use dot notation (e.g., `"address.city"`); list elements are not individually addressed — the entire list field maps to its page set (e.g., `"line_items"`).

#### Pass 3 — line grounding on relevant pages only

```
_extract_pass3(document, formatter, pass1_result, page_mapping, llm_config)
    → dict[str, Any]   # same structure as single-pass citation metadata
```

- The formatter is called with only the pages identified in `page_mapping` (the existing `page_numbers` kwarg handles this).
- The LLM prompt asks for line numbers for each field. The same citation-aware schema from single-pass is used, restricted to the relevant pages.
- `enrich_citations_with_bboxes` is called on the result to resolve line → bbox.

#### Intermediate result storage

Passes 1–3 results are stored on `PDFDocument`:

```python
class PDFDocument(Document):
    content: PDF | None = None
    extraction_mode: Enum = PDFExtractionMode.SINGLE_PASS
    page_numbers: list[int] | None = None       # optional page filter
    pass1_result: BaseModel | None = None       # raw extraction (multi-pass)
    pass2_page_map: dict[str, list[int]] | None = None   # field → pages (multi-pass)
```

The final `extract()` return shape is unchanged: `ExtractionResult(data=…, metadata=…)` — access via `.data` and `.metadata`.

> **Note on `extraction_config`:** `BaseExtractor.extract()` accepts `extraction_config: dict[str, Any]` for interface compatibility. `DocumentProcessor` builds it from `PDFExtractionConfig` and passes it through, but `DigitalPDFExtractor` reads extraction behaviour (mode, page filtering, citation flag) from the `document` object directly. `extraction_config` is not consumed by the current extractor implementation.

---

### 1.5 Files changed / created


| File                                | Change                                                |
| ----------------------------------- | ----------------------------------------------------- |
| `doc_intelligence/config.py`        | Rewrite with `pydantic-settings`                      |
| `doc_intelligence/restrictions.py`  | New — limit validators                                |
| `doc_intelligence/pdf/extractor.py` | Add `_extract_pass1/2/3`, implement MULTI_PASS branch |
| `doc_intelligence/pdf/schemas.py`   | Add `pass1_result`, `pass2_page_map` to `PDFDocument` |
| `doc_intelligence/pdf/processor.py` | Call restriction checks at top of `extract()`         |
| `tests/test_restrictions.py`        | New                                                   |
| `tests/pdf/test_extractor.py`       | Add multi-pass tests                                  |


---

## Phase 2 — Multi-LLM Provider Support(Completed)

### 2.1 Goals

- Add `OllamaLLM`, `AnthropicLLM`, `GeminiLLM` implementations.
- Make `generate_structured_output` optional (not all providers support it natively).
- Ensure swapping providers requires zero changes outside the instantiation line.

---

### 2.2 BaseLLM changes

`generate_structured_output` is moved from abstract to a concrete default that raises `NotImplementedError` with a helpful message. Only providers that natively support structured output override it.

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str: ...

    def generate_structured_output(self, …) -> PydanticModel | None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output. "
            "Use generate_text with a JSON schema prompt instead."
        )
```

This is a non-breaking change: `OpenAILLM` continues to override both methods.

---

### 2.3 OllamaLLM

Uses the native Ollama Python SDK (`ollama.Client`) rather than the OpenAI-compatible `/v1` endpoint, because the native SDK correctly supports Ollama-specific parameters (e.g., `think=False`) that the `/v1` endpoint silently ignores. `OllamaLLM` subclasses `BaseLLM` directly.

```python
class OllamaLLM(BaseLLM):
    def __init__(self, host: str = "http://localhost:11434", model: str | None = None):
        super().__init__(model=model or settings.ollama_default_model)
        import ollama  # optional dependency
        self.client = ollama.Client(host=host)

    @retry(stop=stop_after_attempt(3))
    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        model = kwargs.pop("model", self.model)
        response = self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            **kwargs,
        )
        return response.message.content
```

`generate_structured_output` is inherited from `BaseLLM` as the `NotImplementedError` default.

---

### 2.4 AnthropicLLM

Uses the `anthropic` SDK. Only `generate_text` is implemented; `generate_structured_output` is inherited as the `NotImplementedError` default.

```python
class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(model=model or settings.anthropic_default_model)
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3))
    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        model = kwargs.pop("model", self.model)
        response = self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=kwargs.pop("max_tokens", 4096),
            **kwargs,
        )
        return response.content[0].text
```

---

### 2.5 GeminiLLM

Uses the `google-genai` SDK (already a project dependency).

```python
class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(model=model or settings.gemini_default_model)
        self.client = genai.Client(api_key=api_key)

    @retry(stop=stop_after_attempt(3))
    def generate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        model = kwargs.pop("model", self.model)
        response = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=GenerateContentConfig(system_instruction=system_prompt, **kwargs),
        )
        return response.text
```

---

### 2.6 Files changed / created


| File                       | Change                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `doc_intelligence/base.py` | Make `generate_structured_output` non-abstract with `NotImplementedError` default     |
| `doc_intelligence/llm.py`  | Add `OllamaLLM`, `AnthropicLLM`, `GeminiLLM` (or split into `doc_intelligence/llms/`) |
| `tests/test_llm.py`        | Add tests for all three providers (using mocks)                                       |


> **Note on file organisation:** If `llm.py` grows past ~150 lines, split into `doc_intelligence/llms/__init__.py`, `openai.py`, `ollama.py`, `anthropic.py`, `gemini.py`.

---

- Also update the @docs/ folder according to the features

## Client API Redesign (Completed)

### Goals

- Decouple `DocumentProcessor` from any specific document (remove `document` from `__init__`).
- Fix the cross-provider bug where a global `default_llm_model = "gpt-4o-mini"` was used as fallback for Anthropic/Gemini/Ollama.
- Replace untyped `dict[str, Any]` extraction config with typed keyword arguments.
- Add `ExtractionResult` typed return (`.data` / `.metadata` instead of dict keys).
- Add `create_llm()` factory function for provider-agnostic LLM creation.
- Add `PDFProcessor` convenience class and top-level `extract()` one-liner.

### Key changes


| Change                        | Details                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `BaseLLM` stores `model`      | Each provider has its own default; per-call override via `kwargs.pop("model", self.model)` — transient, never mutates |
| `create_llm()` factory        | Registry-based: `create_llm("openai", model="gpt-4o")`                                                                |
| `ExtractionResult`            | `data: Any`, `metadata: dict                                                                                          |
| `DocumentProcessor.__init_`_  | Takes `parser`, `formatter`, `extractor` only — no `document`                                                         |
| `DocumentProcessor.extract()` | Takes `uri`, `response_format`, keyword-only options — creates fresh `PDFDocument` per call                           |
| `PDFProcessor`                | High-level wrapper accepting `llm` or `provider+model`                                                                |
| `extract()` top-level         | One-liner: `extract("file.pdf", Schema, provider="openai")`                                                           |
| `Document` schema cleanup     | Removed dead `response` and `response_metadata` fields                                                                |
| `config.py` cleanup           | Removed `default_llm_model` (was a cross-provider bug)                                                                |


### Three-layer API

```python
# Layer 1 — one-liner
from doc_intelligence import extract
result = extract("invoice.pdf", InvoiceSchema, provider="openai")

# Layer 2 — reusable processor
from doc_intelligence import PDFProcessor
proc = PDFProcessor(provider="openai", model="gpt-4o")
r1 = proc.extract("a.pdf", Schema)
r2 = proc.extract("b.pdf", Schema)

# Layer 3 — full control
from doc_intelligence import DocumentProcessor
proc = DocumentProcessor.from_digital_pdf(llm=my_llm)
result = proc.extract(uri="a.pdf", response_format=Schema, include_citations=True)
```

---

## Phase 3 — Scanned PDF Support (OCR Pipeline)

### 3.1 Goals

- Parse scanned PDFs (image-based) into the same `PDFDocument` schema as digital PDFs.
- Layout detection and OCR engines are swappable without changing the pipeline.
- OCR of regions within a page runs in parallel via `asyncio`.
- The existing `DigitalPDFFormatter` and `DigitalPDFExtractor` work without modification on OCR output.

---

### 3.2 Component naming

The existing digital components keep their names. New OCR components are named to be explicit:


| Component         | Digital (existing)                     | Scanned (new)                                       |
| ----------------- | -------------------------------------- | --------------------------------------------------- |
| Parser            | `DigitalPDFParser`                     | `ScannedPDFParser`                                  |
| Layout detector   | —                                      | `BaseLayoutDetector` (ABC) + `PaddleLayoutDetector` |
| OCR engine        | —                                      | `BaseOCREngine` (ABC) + `PaddleOCREngine`           |
| Formatter         | `DigitalPDFFormatter`                  | *reused as-is*                                      |
| Extractor         | `DigitalPDFExtractor`                  | *reused as-is*                                      |
| Processor factory | `DocumentProcessor.from_digital_pdf()` | `DocumentProcessor.from_scanned_pdf()`              |


> **Decision:** `DigitalPDFFormatter` and `DigitalPDFExtractor` keep their current names. Both components operate on the `PDF → Page → Line` schema, which both the digital and scanned parsers produce. No rename is needed — "Digital" refers to the origin of the implementation, not a restriction on the data it can process.

---

### 3.3 New abstract base classes

```python
# doc_intelligence/ocr/base.py

class BaseLayoutDetector(ABC):
    @abstractmethod
    def detect(self, page_image: np.ndarray) -> list[BoundingBox]:
        """Return a list of region bboxes for one page image (pixel coords)."""
        ...

class BaseOCREngine(ABC):
    @abstractmethod
    def ocr(self, region_image: np.ndarray) -> list[Line]:
        """Return extracted lines (with normalized bboxes) for one region image."""
        ...
```

Both ABCs work on `numpy` arrays so they are independent of the image-loading library.

---

### 3.4 ScannedPDFParser data flow

```
PDF file / URL
    │
    ▼ (pypdfium2)
list[np.ndarray]  — one array per page
    │
    ▼ BaseLayoutDetector.detect(page_image)  — per page
list[list[BoundingBox]]  — regions per page
    │
    ▼ asyncio.gather(*[ocr_engine.ocr(crop) for crop in page_regions])  — per page
list[list[Line]]  — lines per region, per page
    │
    ▼ assemble
PDFDocument(content=PDF(pages=[Page(lines=[…])]))
```

Parallelism is at the region level within a page. Pages are processed sequentially by default in Phase 3; full page-level parallelism arrives in Phase 4.

The `page_images` generation step (PDF → images) uses `pypdfium2` (already a project dependency) to render pages as arrays.

---

### 3.5 PaddleOCR implementations

```python
# doc_intelligence/ocr/paddle.py

class PaddleLayoutDetector(BaseLayoutDetector):
    def __init__(self, model: str = "picodet_lcnet_x1_0_fgd_layout"):
        from paddleocr import PPStructure
        self._model = PPStructure(layout=True, ocr=False, table=False)

    def detect(self, page_image: np.ndarray) -> list[BoundingBox]:
        result = self._model(page_image)
        return [_to_bounding_box(r["bbox"]) for r in result]

class PaddleOCREngine(BaseOCREngine):
    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def ocr(self, region_image: np.ndarray) -> list[Line]:
        result = self._ocr.ocr(region_image, cls=True)
        return [_to_line(r) for r in (result[0] or [])]
```

The `paddleocr` import is deferred (inside `__init__`) so the library is importable without PaddleOCR installed unless a `ScannedPDFParser` is actually instantiated.

---

### 3.6 ScannedPDFParser

```python
# doc_intelligence/pdf/parser.py  (alongside DigitalPDFParser)

class ScannedPDFParser(BaseParser):
    def __init__(
        self,
        layout_detector: BaseLayoutDetector,
        ocr_engine: BaseOCREngine,
        dpi: int = 150,
    ):
        self.layout_detector = layout_detector
        self.ocr_engine = ocr_engine
        self.dpi = dpi

    def parse(self, document: Document) -> PDFDocument:  # type: ignore[override]
        page_images = _render_pdf_to_images(document.uri, dpi=self.dpi)
        pages = []
        for img in page_images:
            regions = self.layout_detector.detect(img)
            lines = asyncio.run(self._ocr_regions(img, regions))
            pages.append(Page(lines=lines, width=img.shape[1], height=img.shape[0]))
        return PDFDocument(uri=document.uri, content=PDF(pages=pages))

    async def _ocr_regions(
        self, page_image: np.ndarray, regions: list[BoundingBox]
    ) -> list[Line]:
        semaphore = asyncio.Semaphore(settings.max_concurrent_regions)
        async def _ocr_one(bbox: BoundingBox) -> list[Line]:
            async with semaphore:
                crop = _crop(page_image, bbox)
                return await asyncio.to_thread(self.ocr_engine.ocr, crop)
        results = await asyncio.gather(*[_ocr_one(r) for r in regions])
        return [line for region_lines in results for line in region_lines]
```

`asyncio.to_thread` wraps the synchronous PaddleOCR call to avoid blocking the event loop.

---

### 3.7 Factory method addition

```python
# doc_intelligence/pdf/processor.py

@classmethod
def from_scanned_pdf(
    cls,
    llm: BaseLLM,
    layout_detector: BaseLayoutDetector | None = None,
    ocr_engine: BaseOCREngine | None = None,
    **kwargs,
) -> "DocumentProcessor":
    from ..ocr.paddle import PaddleLayoutDetector, PaddleOCREngine
    return cls(
        parser=ScannedPDFParser(
            layout_detector=layout_detector or PaddleLayoutDetector(),
            ocr_engine=ocr_engine or PaddleOCREngine(),
        ),
        formatter=DigitalPDFFormatter(),
        extractor=DigitalPDFExtractor(llm),
        **kwargs,
    )
```

The URI is supplied per-call via `processor.extract(uri=…)`, consistent with `from_digital_pdf`.

---

### 3.8 Files changed / created


| File                                 | Change                                          |
| ------------------------------------ | ----------------------------------------------- |
| `doc_intelligence/ocr/__init__.py`   | New package                                     |
| `doc_intelligence/ocr/base.py`       | New — `BaseLayoutDetector`, `BaseOCREngine`     |
| `doc_intelligence/ocr/paddle.py`     | New — `PaddleLayoutDetector`, `PaddleOCREngine` |
| `doc_intelligence/pdf/parser.py`     | Add `ScannedPDFParser` (merged from ocr_parser) |
| `doc_intelligence/pdf/processor.py`  | Add `from_scanned_pdf()` factory                |
| `tests/ocr/test_base.py`             | New                                             |
| `tests/ocr/test_paddle.py`           | New (mocked PaddleOCR)                          |
| `tests/pdf/test_ocr_parser.py`       | New                                             |


---

## Phase 4 — Batch and Async Processing

### 4.1 Goals

- Provide a fully async pipeline: `AsyncDocumentProcessor`.
- Support batch processing of multiple documents concurrently.
- Expose per-document progress callbacks.
- All concurrency limits configurable via `settings` (already declared in Phase 1).

---

### 4.2 Async component strategy

Rather than duplicating every component, the async pipeline wraps the existing sync components with `asyncio.to_thread` for CPU-bound steps and uses native `async` for IO-bound steps (LLM HTTP calls).

New async LLM interface (added to `BaseLLM`):

```python
class BaseLLM(ABC):
    # existing sync methods stay
    @abstractmethod
    def generate_text(self, …) -> str: ...

    async def agenerate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Default: runs generate_text in a thread. Override for native async."""
        return await asyncio.to_thread(self.generate_text, system_prompt, user_prompt, **kwargs)
```

Providers that have native async SDKs (OpenAI, Anthropic) can override `agenerate_text` directly.

---

### 4.3 AsyncDocumentProcessor

```python
# doc_intelligence/pdf/async_processor.py

class AsyncDocumentProcessor:
    """Async wrapper over the same parse → format → extract pipeline."""

    def __init__(self, parser, formatter, extractor):
        ...  # mirrors DocumentProcessor — no document field

    @classmethod
    def from_digital_pdf(cls, llm: BaseLLM) -> "AsyncDocumentProcessor": ...

    @classmethod
    def from_scanned_pdf(cls, llm: BaseLLM, …) -> "AsyncDocumentProcessor": ...

    async def extract(
        self,
        uri: str,
        response_format: type[PydanticModel],
        *,
        include_citations: bool = True,
        extraction_mode: str = "single_pass",
        llm_config: dict[str, Any] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        document = PDFDocument(uri=uri, include_citations=include_citations, ...)
        if on_progress:
            asyncio.create_task(on_progress(uri, "parsing"))
        document.content = (
            await asyncio.to_thread(self.parser.parse, document)
        ).content
        if on_progress:
            asyncio.create_task(on_progress(uri, "extracting"))
        result = await asyncio.to_thread(self.extractor.extract, document, llm_config or {}, …)
        if on_progress:
            asyncio.create_task(on_progress(uri, "done"))
        return result
```

A fresh `PDFDocument` is created per `extract()` call; the processor holds no document state. Progress callbacks are fire-and-forget via `asyncio.create_task` (consistent with §4.5) so a slow callback cannot block the pipeline.

---

### 4.4 Batch API

```python
# doc_intelligence/batch.py

ProgressCallback = Callable[[str, str], Awaitable[None]]  # (uri, status) → None

async def batch_extract(
    documents: list[str | Document],
    response_format: type[PydanticModel],
    llm: BaseLLM,
    *,
    include_citations: bool = True,
    extraction_mode: str = "single_pass",
    llm_config: dict[str, Any] | None = None,
    document_type: Literal["digital", "scanned"] = "digital",
    on_progress: ProgressCallback | None = None,
) -> list[ExtractionResult | Exception]:
    """
    Process multiple documents concurrently.
    Returns results in input order; failed documents return the Exception, not raise.
    """
    semaphore = asyncio.Semaphore(settings.max_concurrent_documents)

    async def _process_one(doc_uri: str | Document) -> ExtractionResult | Exception:
        async with semaphore:
            try:
                processor = AsyncDocumentProcessor.from_digital_pdf(llm)
                return await processor.extract(
                    str(doc_uri),
                    response_format,
                    include_citations=include_citations,
                    extraction_mode=extraction_mode,
                    llm_config=llm_config,
                    on_progress=on_progress,
                )
            except Exception as e:
                logger.error(f"batch_extract: failed for {doc_uri}: {e}")
                return e

    return await asyncio.gather(*[_process_one(d) for d in documents])
```

Failures are returned as `Exception` objects (not re-raised) so one bad document does not abort the batch.

---

### 4.5 Progress callback shape

```python
# Status values emitted in order:
# "queued" → "parsing" → "extracting" → "done"
# On failure: "queued" → "parsing"|"extracting" → "failed"

async def my_progress(uri: str, status: str) -> None:
    print(f"{uri}: {status}")

results = await batch_extract(uris, config, llm, on_progress=my_progress)
```

The callback is fire-and-forget (`asyncio.create_task`) so a slow callback cannot block the pipeline.

---

### 4.6 Files changed / created


| File                                      | Change                                                                     |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| `doc_intelligence/base.py`                | Add `agenerate_text` default to `BaseLLM`                                  |
| `doc_intelligence/llm.py`                 | Override `agenerate_text` in `OpenAILLM`, `AnthropicLLM` with native async |
| `doc_intelligence/pdf/async_processor.py` | New — `AsyncDocumentProcessor`                                             |
| `doc_intelligence/batch.py`               | New — `batch_extract`                                                      |
| `tests/pdf/test_async_processor.py`       | New                                                                        |
| `tests/test_batch.py`                     | New                                                                        |


---

## Cross-cutting Concerns

### Dependency management

New optional dependencies are declared as extras in `pyproject.toml`, not required:

```toml
[project.optional-dependencies]
ollama     = ["ollama>=0.4.0"]              # native Ollama SDK
anthropic  = ["anthropic>=0.40.0"]
gemini     = ["google-genai>=1.57.0"]       # already in main deps
ocr        = ["paddleocr>=2.9.0", "paddlepaddle>=3.0.0"]
```

This keeps the base install small. A user who only needs OpenAI + digital PDFs does not install PaddleOCR.

### Error taxonomy


| Situation                         | Exception                                      |
| --------------------------------- | ---------------------------------------------- |
| File too large                    | `ValueError: PDF exceeds max size (…MB > …MB)` |
| Too many pages                    | `ValueError: PDF has … pages, limit is …`      |
| Schema too deep                   | `ValueError: Schema depth … exceeds limit …`   |
| LLM parse failure (after retries) | `ExtractionError` (custom, Phase 1+ — not yet implemented; currently propagates from JSON parser) |
| OCR engine failure                | `OCRError` (custom, Phase 3+ — not yet implemented; currently propagates from PaddleOCR)          |


### Testing conventions (unchanged from baseline)

- Mirror structure: `tests/ocr/`, `tests/pdf/test_async_processor.py`, etc.
- All OCR and LLM calls are mocked; no network in unit tests.
- 100% coverage target on all new modules.

