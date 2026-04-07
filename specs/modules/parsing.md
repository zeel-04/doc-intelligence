# Module: Parsing

**Source:** `doc_intelligence/pdf/parser.py`, `doc_intelligence/base.py`
**Diagram ref:** "Parsing Phase" (step 3 in the pipeline, after restrictions and type selection)

---

## 1. Purpose

Convert a raw PDF file into a structured `PDFDocument` — a list of `Page` objects, each containing an ordered sequence of `ContentBlock` items. The parser is the single entry point for all PDF-to-structure conversion, hiding the strategy (digital vs. scanned) behind a unified interface so that downstream formatting and extraction work identically regardless of source type.

---

## 2. Pipeline Position

```
Restrictions (pre-processing)
    │
    ▼
► Parsing (this module)     ← PDFParser.parse(uri)
    │
    ▼
  PDFDocument
    │
    ▼
  Formatting Phase ...
```

**Called by:** `DocumentProcessor.extract()` in `pdf/processor.py`.
**Runs after:** Restriction checks (size, page count, schema depth).
**Produces:** `PDFDocument` consumed by `PDFFormatter`.

---

## 3. Inputs & Outputs

### 3.1 Inputs

| Parameter | Type | Description |
|---|---|---|
| `uri` | `str` | Local file path or HTTP(S) URL of the PDF |
| `strategy` | `ParseStrategy` | `DIGITAL` (default) or `SCANNED` — set at construction time |
| `layout_detector` | `BaseLayoutDetector \| None` | Required for `SCANNED` strategy |
| `ocr_engine` | `BaseOCREngine \| None` | Required for `SCANNED` strategy |
| `dpi` | `int` | Page rendering resolution for scanned PDFs (default `150`) |

### 3.2 Outputs

| Output | Type | Description |
|---|---|---|
| Return value | `PDFDocument` | Contains `uri` and `content: PDF` with a list of `Page` objects |

Each `Page` contains:
- `blocks: list[ContentBlock]` — ordered sequence of `TextBlock`, `TableBlock`, `ImageBlock`, or `ChartBlock`.
- `width`, `height` — page dimensions.

---

## 4. Public API

### 4.1 `PDFParser(strategy, layout_detector?, ocr_engine?, dpi?)`

Constructor. Validates that `layout_detector` and `ocr_engine` are provided when `strategy` is `SCANNED`. Raises `ValueError` otherwise.

### 4.2 `PDFParser.parse(uri: str) -> PDFDocument`

Entry point. Dispatches to the appropriate internal strategy based on the `strategy` set at construction.

---

## 5. Internal Design

### 5.1 Strategy Dispatch

`parse()` delegates to one of two private methods based on `self._strategy`:

- `_parse_digital(uri)` — native text extraction via pdfplumber.
- `_parse_scanned(uri)` — image rendering + layout detection + OCR.

No `if doc_type == ...` conditionals exist outside these two paths — the strategy is fixed at construction.

### 5.2 Digital Parsing (`_parse_digital`)

Uses `pdfplumber.open()` to iterate over pages. For each page:

1. Calls `page.extract_text_lines(return_chars=False)` to get line-level text with bounding boxes.
2. Normalizes each bounding box to `[0, 1]` coordinates via `normalize_bounding_box()`.
3. Wraps each line as a `TextBlock` containing a single `Line`.

This one-line-per-block design gives block-level citation addressing the precision of line-level addressing.

**URI handling:** HTTP(S) URLs are downloaded to a `BytesIO` buffer first; local paths are passed directly to pdfplumber.

### 5.3 Scanned Parsing (`_parse_scanned`)

The OCR pipeline supports two architectures for extracting structured text from scanned page images. Both produce the same output — pages with layout regions containing recognized text and spatial metadata. Only `TextBlock` and `TableBlock` are fully supported end-to-end. Image and chart regions are stored as content-less placeholder blocks (bbox only) — no OCR is performed on them. These placeholders preserve layout structure but are excluded from formatting and extraction until VLM support is added.

#### Pipeline Type 1: Two-Stage (Separate Layout + OCR) — current implementation

This pipeline uses two distinct models in sequence:

1. **Render:** `_render_pdf_to_images(uri, dpi)` converts each page to an HxWxC uint8 numpy array using pypdfium2 at the configured DPI.
2. **Layout Detection (batched by page):** `self._layout_detector.detect(image)` segments each page image into typed `LayoutRegion` objects — bounding boxes identifying text blocks, tables, figures, etc. Each region carries metadata linking it back to its source page.
3. **OCR (batched by region):** For text and table regions, crops the image and runs `self._ocr_engine.ocr(cropped)` to produce `Line` objects with text and bounding boxes. All detected regions across all pages are collected into a flat list and processed independently.
4. **Reassembly:** Since each region carries its page metadata, the OCR'd regions are mapped back to their originating pages to reconstruct the full document structure.

**Concurrency model:**
- Pages are processed **sequentially** (`_parse_all_pages`).
- Within each page, regions are processed **concurrently** via `asyncio.gather` with a `Semaphore` capping concurrent OCR calls to `settings.max_concurrent_regions` (default 8).
- OCR calls are offloaded to threads via `asyncio.to_thread` since most OCR engines are synchronous.

#### Pipeline Type 2: Single-Stage (Unified Layout + OCR) — implemented

This pipeline uses a single vision-capable LLM — via `BaseLLM.generate(images=...)` — that performs both layout detection and text recognition in one call.

Each page image is rendered, encoded to a base64 PNG data URL, and sent to the LLM with a prompt requesting structured JSON containing typed blocks (text, table, figure, etc.) with bounding boxes and OCR'd text. Pages are batched according to `vlm_batch_size` (default 1). No reassembly step is needed since results are already organized by page.

**Selection:** Set `scanned_pipeline=ScannedPipelineType.VLM` on `PDFParser` or `PDFProcessor`. The LLM is passed via the `llm` constructor parameter (or reused from the processor's extraction LLM in `PDFProcessor`).

**Implementation:** `PDFParser._parse_scanned_vlm()` in `parser.py`. Response parsing via `_parse_vlm_response()` handles varying JSON shapes from different VLM providers.

#### Pipeline Comparison

| | Type 1 (Two-Stage) | Type 2 (Single-Stage) |
|---|---|---|
| **Models** | Separate layout + OCR models | Single visual language model |
| **Inference passes** | Two (layout, then OCR) | One |
| **Batch unit** | Stage 1: pages, Stage 2: regions | Pages (configurable batch size) |
| **Flexibility** | Swap layout or OCR model independently | Coupled to one model |
| **Reassembly** | Required (regions → pages) | Not needed |
| **Status** | Implemented (orchestration only, no concrete engines) | Implemented |

### 5.4 Region-to-Block Mapping

| Region type | Block type | OCR performed? | Supported now? |
|---|---|---|---|
| text (default) | `TextBlock` | Yes — lines from OCR | Yes |
| table | `TableBlock` | Yes — each line becomes a single-cell row | Yes |
| image, figure, picture, photo | `ImageBlock` | No — bbox only | No — placeholder for future VLM support |
| chart, diagram, plot, graph | `ChartBlock` | No — bbox only | No — placeholder for future VLM support |

**Current scope:** Only `TextBlock` and `TableBlock` are fully supported end-to-end (parsing → formatting → extraction). `ImageBlock` and `ChartBlock` are created during parsing to preserve layout structure, but carry no content — they are excluded from formatter output and block index numbering, and are invisible to the extraction phase. Full support is planned when VLM integration is added.

Region type matching is case-insensitive. The recognized label sets (`_IMAGE_REGION_TYPES`, `_CHART_REGION_TYPES`) are frozen sets that can be expanded as layout detectors adopt new vocabularies.

### 5.5 Module-Level Helpers

- `_render_pdf_to_images(uri, dpi)` — Downloads URL if needed, renders via pypdfium2 at `dpi / 72.0` scale. Returns ordered list of numpy arrays.
- `_crop(image, bbox)` — Pixel-coordinate crop using numpy slicing.

---

## 6. Configuration

| Setting | Env var | Default | Used by |
|---|---|---|---|
| `max_concurrent_regions` | `DOC_INTEL_MAX_CONCURRENT_REGIONS` | `8` | Semaphore in `_parse_page` |

DPI is a constructor parameter (default `150`), not a global config setting.

---

## 7. Constraints & Invariants

- **Strategy is immutable.** Set at construction, never changed. No runtime strategy switching.
- **Output schema is strategy-agnostic.** Both strategies produce identical `PDFDocument` structure — downstream code never needs to know which strategy was used.
- **One `TextBlock` per line (digital).** For digital PDFs, each extracted text line becomes its own `TextBlock`. This is a deliberate design choice for citation granularity.
- **Bounding boxes are normalized (digital).** All bounding boxes from digital parsing are in `[0, 1]` relative coordinates. Scanned parsing preserves the coordinates returned by the layout detector/OCR engine.
- **No content mutation.** The parser creates a new `PDFDocument` — it never mutates input data.
- **Sequential pages, parallel regions.** Pages are always in document order. Region concurrency within a page does not affect block ordering (results are gathered in region order).

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| `SCANNED` strategy without `layout_detector` | `ValueError` at construction |
| `SCANNED` strategy without `ocr_engine` | `ValueError` at construction |
| Unsupported `ParseStrategy` value | `ValueError` from `parse()` |
| HTTP URL returns error | `requests.HTTPError` propagated from `requests.get().raise_for_status()` |
| Corrupt or unreadable PDF | Exception from pdfplumber (digital) or pypdfium2 (scanned) propagated |
| OCR engine failure | Exception propagated from `ocr_engine.ocr()` |

---

## 9. Extension Points

### 9.1 Adding New Parse Strategies

Add a new `ParseStrategy` enum value in `types.py` and a corresponding `_parse_<strategy>` method in `PDFParser`. Update `parse()` dispatch.

### 9.2 Custom Layout Detectors and OCR Engines

Implement `BaseLayoutDetector` or `BaseOCREngine` from `doc_intelligence/ocr/base.py` and pass to the `PDFParser` constructor. No changes to parser code needed.

### 9.3 New Document Types

New document types (DOCX, images) create their own parser subclass of `BaseParser[TDocument]` in their respective module folder, following the same pattern.

