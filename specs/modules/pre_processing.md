# Module: Pre-Processing

**Source:** `doc_intelligence/restrictions.py`, `doc_intelligence/config.py`
**Diagram ref:** "Restrictions — size | pages | schema depth" (top of pipeline, before type detection)

---

## 1. Purpose

Guard the pipeline against inputs that are too expensive or too complex to process. Every restriction runs before any parsing, formatting, or LLM work begins. The module exists to fail fast and cheaply — a rejected file never touches pdfplumber, OCR, or an LLM.

---

## 2. Pipeline Position

```
PDF Input
    │
    ▼
► Restrictions (this module)     ← runs inside PDFProcessor.extract(), before parse
    │
    ▼
  Type? (Digital / Scanned)
    │
    ▼
  Parsing Phase ...
```

**Called by:** `PDFProcessor.extract()` in `pdf/processor.py` (lines 175–177).
**Runs before:** `DocumentProcessor.extract()` — no document is parsed if restrictions reject.
**Depends on:** `DocIntelligenceConfig` for default thresholds.

---

## 3. Inputs & Outputs

### 3.1 Inputs

| Check | Input | Source |
|---|---|---|
| `check_pdf_size` | `uri: str`, `max_mb: float` | File system (`os.path.getsize`) |
| `check_page_count` | `uri: str`, `max_pages: int` | PDF metadata (`pdfplumber.open`) |
| `check_schema_depth` | `model: type[BaseModel]`, `max_depth: int` | Pydantic model class introspection |

### 3.2 Outputs

None. Each function returns `None` on success or raises `ValueError` on violation. There is no data transformation — this module is a gate, not a pipeline stage.

---

## 4. Public API

### 4.1 `check_pdf_size(uri: str, max_mb: float) -> None`

Compares file size on disk against `max_mb`. Reads only file metadata (`os.path.getsize`), never opens the PDF.

### 4.2 `check_page_count(uri: str, max_pages: int) -> None`

Opens the PDF with `pdfplumber.open()` to read page count, then closes it. Does not extract any text or content — only `len(pdf.pages)`.

### 4.3 `check_schema_depth(model: type[BaseModel], max_depth: int) -> None`

Recursively walks `model.model_fields` to measure nesting depth. The root model is depth 0. Each nested `BaseModel` increments by 1.

### 4.4 `_walk_annotation(annotation: Any, max_depth: int, current: int) -> None`

Internal helper. Traverses generic type annotations (`list[T]`, `T | None`, `Union[T, U]`) to find nested `BaseModel` subclasses within wrappers.

---

## 5. Internal Design

### 5.1 Fail-Fast Philosophy

Restrictions are ordered cheapest-first in `PDFProcessor.extract()`:

1. **Size** — `os.path.getsize`, no I/O beyond a stat call.
2. **Page count** — opens the PDF but only reads metadata.
3. **Schema depth** — pure in-memory type introspection.

If any check fails, subsequent checks do not run.

### 5.2 Restriction Types

| Restriction | What it prevents | Default threshold |
|---|---|---|
| File size | Oversized files consuming memory during parsing | `10.0 MB` |
| Page count | Long documents driving up LLM token cost | `100 pages` |
| Schema depth | Deeply nested schemas that produce complex, unreliable LLM output | `5 levels` |

### 5.3 Non-Local URI Handling

Both `check_pdf_size` and `check_page_count` silently pass for non-local URIs (URLs, missing files). The guard is `os.path.isfile(uri)` — if it returns `False`, the function returns without checking. This means:

- Remote URLs skip size and page checks entirely.
- Non-existent paths are not rejected here — they will fail at parse time with a more descriptive error.

### 5.4 Schema Depth Walking Algorithm

`check_schema_depth` uses mutual recursion between itself and `_walk_annotation`:

```
check_schema_depth(Model, max_depth, current=0)
  └─ for each field in Model.model_fields:
       └─ _walk_annotation(field.annotation, max_depth, current)
            ├─ if generic (list[T], T | None): recurse into each type arg
            └─ if BaseModel subclass: call check_schema_depth(sub, max_depth, current + 1)
```

- `list[InnerModel]` counts as one level of nesting (the list wrapper itself does not add depth).
- `InnerModel | None` unwraps the Optional — `NoneType` is skipped, the model is walked.
- Depth is checked at entry (`_current > max_depth`), so the limit is inclusive: `max_depth=1` allows one level of nesting.

### 5.5 Boundary Semantics

All checks use strict inequality for rejection:

- Size: rejects when `size_mb > max_mb` (exactly-at-limit passes).
- Pages: rejects when `page_count > max_pages` (exactly-at-limit passes).
- Depth: rejects when `_current > max_depth` (exactly-at-limit passes).

---

## 6. Configuration

All thresholds are set in `DocIntelligenceConfig` (`config.py`) and overridable via environment variables:

| Setting | Env var | Default |
|---|---|---|
| `max_pdf_size_mb` | `DOC_INTEL_MAX_PDF_SIZE_MB` | `10.0` |
| `max_pdf_pages` | `DOC_INTEL_MAX_PDF_PAGES` | `100` |
| `max_schema_depth` | `DOC_INTEL_MAX_SCHEMA_DEPTH` | `5` |

The `settings` singleton in `config.py` is the single source of truth. `PDFProcessor` reads from it at call time — thresholds are never hardcoded at the call site.

---

## 7. Constraints & Invariants

- **No side effects.** Restriction functions are pure validators — they read state but never modify it. No files are created, no content is extracted, no network calls are made.
- **All rejections are `ValueError`.** No custom exception types. Error messages include the measured value, the limit, and (for size) the file path.
- **Restrictions run as a group.** `PDFProcessor.extract()` calls all three sequentially. If a new restriction is added, it must be added to that call sequence.
- **Schema depth is document-type agnostic.** `check_schema_depth` works on any `BaseModel` — it has no PDF-specific logic and is reusable for future document types.

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| File exceeds size limit | `ValueError` with measured size, limit, and file path |
| Page count exceeds limit | `ValueError` with page count and limit |
| Schema too deep | `ValueError` with depth and limit |
| URI is a URL (not local file) | Size and page checks silently pass |
| URI points to non-existent file | Size and page checks silently pass; parse will fail later |
| `response_format` is not a `BaseModel` subclass | `ValueError` raised by `PDFProcessor.extract()` before restrictions run |

---

## 9. Extension Points

### 9.1 Adding New Restriction Types

Add a new function to `restrictions.py` following the same pattern: accept the input and threshold, return `None` or raise `ValueError`. Then add the call to `PDFProcessor.extract()` in the restriction block, and add the corresponding threshold to `DocIntelligenceConfig`.

### 9.2 Per-Document-Type Restrictions

`check_pdf_size` and `check_page_count` are PDF-specific (the latter uses `pdfplumber`). Future document types (DOCX, images) will need their own size/page validators. `check_schema_depth` is already document-type agnostic.
