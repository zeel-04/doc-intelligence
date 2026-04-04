# OCR Pipeline — Phased Implementation Plan

**Version:** 1.1
**Status:** Draft
**Last updated:** 2026-03-29
**Parent spec:** `specs/engineering_design.md` §Phase 3

---

## Overview

This document breaks down the OCR pipeline (Phase 3 of the engineering design) into isolated, incrementally deliverable sub-phases. Each sub-phase produces working, tested code and leaves the codebase green.

The end goal: `ScannedPDFParser` produces the same `PDFDocument` schema as `DigitalPDFParser`, so the existing formatter and extractor work without modification.

---

## Decisions

These decisions were made during planning and apply across all sub-phases:

1. **`LayoutRegion` location:** Lives in `ocr/base.py`, co-located with the `BaseLayoutDetector` ABC that returns it. Moved to `schemas/core.py` only if non-OCR code needs it later.

2. **Table regions — structured representation:** The layout detector identifies tables as a region type. Tables are represented as structured `TableBlock(rows=list[list[Cell]])` in the page schema, not flattened into lines. A `table_block_to_text_block()` utility converts tables to text for the formatter, preserving the structured data for consumers who need it.

3. **Image rendering library:** `pypdfium2` (already a project dependency at `>=4.30.0`). Purpose-built for rendering, good DPI control, returns PIL Images convertible to numpy arrays via `np.array()`.

---

## Sub-phase 3.0 — Schema Migration (Line → ContentBlock) [DONE]

**Goal:** Evolve the page data model from flat `list[Line]` to structured `list[ContentBlock]` that supports both text regions and tables, for both digital and scanned pipelines.

**Why first:** Building the OCR pipeline on the right schema avoids a retrofit. The digital pipeline benefits too — pdfplumber already supports table extraction.

### New schemas

```python
# doc_intelligence/pdf/schemas.py

class Cell(BaseModel):
    """A single cell in a table."""
    text: str
    bounding_box: BoundingBox | None = None

class TextBlock(BaseModel):
    """A contiguous block of text lines."""
    block_type: Literal["text"] = "text"
    bounding_box: BoundingBox | None = None
    lines: list[Line]

class TableBlock(BaseModel):
    """A structured table region."""
    block_type: Literal["table"] = "table"
    bounding_box: BoundingBox | None = None
    rows: list[list[Cell]]

ContentBlock = TextBlock | TableBlock

class Page(BaseModel):
    blocks: list[ContentBlock]
    width: int | float
    height: int | float
```

### Conversion utilities

```python
# doc_intelligence/pdf/schemas.py (or a dedicated utils if it grows)

def table_block_to_text_block(table: TableBlock) -> TextBlock:
    """Convert a TableBlock into a TextBlock for LLM consumption.

    Each row becomes a pipe-delimited line: "| cell1 | cell2 | cell3 |"
    """
    ...

def blocks_to_lines(blocks: list[ContentBlock]) -> list[Line]:
    """Flatten all blocks into a flat list of Lines (backward compat helper).

    TextBlocks contribute their lines directly.
    TableBlocks are converted via table_block_to_text_block() first.
    """
    ...
```

### Migration path

The `Page.lines` field is replaced by `Page.blocks`. This ripples through:

| Component | Change needed |
|---|---|
| `Page` schema | `lines: list[Line]` → `blocks: list[ContentBlock]` |
| `DigitalPDFParser` | Wrap extracted lines in `TextBlock`. Table extraction is **not** added in this sub-phase — all content becomes `TextBlock`s. Table extraction from pdfplumber is a follow-up enhancement. |
| `DigitalPDFFormatter` | Iterate over `page.blocks` instead of `page.lines`. For `TextBlock`, format lines as before. For `TableBlock`, call `table_block_to_text_block()` first, then format. |
| `DigitalPDFExtractor` | No change — operates on formatter output (strings), not on `Page` directly. |
| `enrich_citations_with_bboxes` (pdf/utils.py) | Update line lookups to work with blocks. Lines are addressed by a global line index across all blocks on a page, preserving backward compat with citation line numbers. |
| Tests | Update all tests that construct `Page(lines=[...])` to use `Page(blocks=[TextBlock(lines=[...])])`. |

### What to build / change

```
doc_intelligence/pdf/schemas.py        — add Cell, TextBlock, TableBlock, ContentBlock;
                                         change Page.lines → Page.blocks;
                                         add table_block_to_text_block(), blocks_to_lines()
doc_intelligence/pdf/parser.py         — wrap lines in TextBlock
doc_intelligence/pdf/formatter.py      — iterate blocks, handle TextBlock and TableBlock
doc_intelligence/pdf/utils.py          — update enrich_citations_with_bboxes line resolution
tests/pdf/test_schemas.py             — new tests for Cell, TextBlock, TableBlock, conversion utils
tests/pdf/test_parser.py              — update assertions for blocks
tests/pdf/test_formatter.py           — update assertions, add TableBlock formatting test
tests/pdf/test_utils.py               — update citation enrichment tests
tests/conftest.py                     — update sample_pdf fixtures
```

**Verification:** `uv run pytest tests/` (full suite) + `ruff check` + `ruff format --check` + `pyrefly check .`

**Exit criteria:** All existing tests pass with the new schema. The digital pipeline works exactly as before. `TableBlock` is defined and the formatter knows how to render it, even though no parser produces `TableBlock`s yet.

---

## Sub-phase 3.1 — Abstract Base Classes [DONE]

**Goal:** Define the contracts for layout detection and OCR engines.

**What to build:**

```
doc_intelligence/ocr/__init__.py       — package marker, re-exports
doc_intelligence/ocr/base.py           — BaseLayoutDetector, BaseOCREngine, LayoutRegion
tests/ocr/__init__.py                  — test package marker
tests/ocr/test_base.py                 — tests for ABCs
```

**Details:**

- `BaseLayoutDetector` — single abstract method:
  ```python
  def detect(self, page_image: np.ndarray) -> list[LayoutRegion]: ...
  ```
  Returns a list of detected regions with bounding boxes, region type labels, and confidence scores.

- `BaseOCREngine` — single abstract method:
  ```python
  def ocr(self, region_image: np.ndarray) -> list[Line]: ...
  ```
  Returns lines with normalized bounding boxes for a single cropped region image.

- `LayoutRegion` schema (in `ocr/base.py`):
  ```python
  class LayoutRegion(BaseModel):
      bounding_box: BoundingBox    # pixel coordinates within the page image
      region_type: str             # e.g. "text", "table", "header", "figure"
      confidence: float            # detection confidence score
  ```
  `region_type` is a plain string so third-party layout detectors can return their own labels without being constrained to a fixed set.

- Both ABCs accept `np.ndarray` inputs — framework-agnostic.

**Tests:**
- Verify ABCs cannot be instantiated directly.
- Concrete test subclasses (`FakeLayoutDetector`, `FakeOCREngine`) confirm the interface works.
- Add fakes to `tests/conftest.py` for reuse in later sub-phases.

**Verification:** `uv run pytest tests/ocr/` + `ruff check` + `ruff format --check`

**Exit criteria:** ABCs exist, are importable, and have 100% test coverage.

---

## Sub-phase 3.2 — PaddleOCR Implementations [DONE]

**Goal:** Implement `PaddleLayoutDetector` and `PaddleOCREngine` using PaddleOCR.

**What to build:**

```
doc_intelligence/ocr/paddle.py         — PaddleLayoutDetector, PaddleOCREngine
tests/ocr/test_paddle.py              — tests (mocked PaddleOCR)
```

**Details:**

- `PaddleLayoutDetector`:
  - Deferred import of `paddleocr.PPStructure` inside `__init__`.
  - `detect()` runs the layout model and converts results to `list[LayoutRegion]`.
  - Helper `_to_layout_region()` maps PaddleOCR's raw bbox + type output to `LayoutRegion`.

- `PaddleOCREngine`:
  - Deferred import of `paddleocr.PaddleOCR` inside `__init__`.
  - `ocr()` runs OCR on a cropped region image, returns `list[Line]`.
  - Helper `_to_line()` maps PaddleOCR's `[[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, confidence)` to `Line(text=..., bounding_box=...)` with normalized bounding boxes.
  - Bounding box normalization: convert from pixel coordinates to 0–1 scale relative to region image dimensions.

- Deferred imports ensure the library is importable without PaddleOCR installed.

- Optional dependency already declared in `pyproject.toml`:
  ```toml
  ocr = ["paddleocr>=2.9.0", "paddlepaddle>=3.0.0"]
  ```

**Tests:**
- All PaddleOCR calls are mocked — no real ML inference in tests.
- Test `detect()` with mocked PPStructure output → verify correct `LayoutRegion` list.
- Test `ocr()` with mocked PaddleOCR output → verify correct `Line` list with normalized bboxes.
- Test edge cases: empty results, single-line regions, multi-line regions.
- Test deferred import behavior: importing `doc_intelligence.ocr` does not require PaddleOCR.

**Verification:** `uv run pytest tests/ocr/` + `ruff check` + `ruff format --check`

**Exit criteria:** Both implementations pass all tests with mocked PaddleOCR. Library remains importable without PaddleOCR installed.

---

## Sub-phase 3.3 — ScannedPDFParser [DONE]

**Goal:** Build the parser that wires layout detection → OCR → assembly into `PDFDocument`.

**What to build:**

```
doc_intelligence/pdf/parser.py         — ScannedPDFParser (merged alongside DigitalPDFParser)
tests/pdf/test_ocr_parser.py          — tests
```

**Details:**

- `ScannedPDFParser(BaseParser[PDFDocument])`:
  - Constructor accepts `layout_detector`, `ocr_engine`, and `dpi` (default 150).
  - `parse(document) -> PDFDocument`:
    1. Render PDF pages to `list[np.ndarray]` using `pypdfium2` (via `_render_pdf_to_images(uri, dpi)`).
    2. For each page image: run `layout_detector.detect(image)` → `list[LayoutRegion]`.
    3. For each page: run OCR on all regions in parallel via `asyncio.gather` + `asyncio.to_thread` with a semaphore (`settings.max_concurrent_regions`).
    4. Assemble `Page(blocks=[...], width=..., height=...)` per page. Text regions become `TextBlock`, table regions become `TableBlock` (table structure extracted by the OCR engine or a dedicated table handler).
    5. Return `PDFDocument(uri=..., content=PDF(pages=[...]))`.

- Helper `_render_pdf_to_images(uri, dpi) -> list[np.ndarray]`:
  - Uses `pypdfium2` (already a dependency) to render each page.
  - `pypdfium2` renders to a `PdfBitmap` → convert to PIL Image → `np.array()`.
  - Handles both local paths and URLs (download first, consistent with `DigitalPDFParser` pattern).

- Helper `_crop(image, bbox) -> np.ndarray`:
  - Crops a region from the page image using a `BoundingBox` (pixel coordinates).

- Region-level parallelism within each page. Pages processed sequentially in this sub-phase (page-level parallelism deferred to Phase 4).

- `asyncio.to_thread` wraps synchronous OCR calls so they don't block the event loop.

**Tests:**
- Use `FakeLayoutDetector` and `FakeOCREngine` from `tests/conftest.py`.
- Test full `parse()` flow with a mock PDF rendering step.
- Test that output `PDFDocument` has correct block structure: `TextBlock`s and `TableBlock`s.
- Test parallel OCR: verify semaphore limits concurrent calls.
- Test edge cases: empty pages (no regions detected), single-region pages, pages with many regions.
- Test URL vs local path handling in `_render_pdf_to_images`.

**Verification:** `uv run pytest tests/pdf/test_ocr_parser.py tests/pdf/test_parser.py` + `ruff check` + `ruff format --check`

**Exit criteria:** `ScannedPDFParser` produces valid `PDFDocument` objects with `ContentBlock`-based pages. Existing formatter and extractor accept its output unchanged.

---

## Sub-phase 3.4 — Factory Method & Integration [DONE]

**Goal:** Wire `ScannedPDFParser` into `DocumentProcessor` and verify end-to-end.

**What to change:**

```
doc_intelligence/pdf/processor.py      — add from_scanned_pdf() factory
tests/pdf/test_processor.py            — add scanned pipeline tests
```

**Details:**

- `DocumentProcessor.from_scanned_pdf()`:
  ```python
  @classmethod
  def from_scanned_pdf(
      cls,
      llm: BaseLLM,
      layout_detector: BaseLayoutDetector | None = None,
      ocr_engine: BaseOCREngine | None = None,
      dpi: int = 150,
  ) -> "DocumentProcessor":
  ```
  - Defaults to `PaddleLayoutDetector()` and `PaddleOCREngine()` via deferred import.
  - Reuses `DigitalPDFFormatter` and `DigitalPDFExtractor` — no new formatter or extractor needed.

- `PDFProcessor` update: add `document_type` parameter (or a `from_scanned_pdf` convenience) so the high-level API also supports scanned PDFs.

- Top-level `extract()` function: add `document_type: Literal["digital", "scanned"] = "digital"` parameter.

**Tests:**
- Test `from_scanned_pdf()` returns a correctly wired `DocumentProcessor`.
- Test with custom `layout_detector` and `ocr_engine` injection.
- Test with defaults (mocked PaddleOCR imports).
- End-to-end test: `processor.extract(uri, Schema)` with `FakeLLM` + `FakeLayoutDetector` + `FakeOCREngine` → verify `ExtractionResult` shape.
- Test that existing `from_digital_pdf()` tests still pass (no regressions).

**Verification:** `uv run pytest tests/` (full suite) + `ruff check` + `ruff format --check` + `pyrefly check .`

**Exit criteria:** A developer can do:
```python
processor = DocumentProcessor.from_scanned_pdf(llm=OpenAILLM())
result = processor.extract(uri="scanned.pdf", response_format=MySchema)
```
and get the same `ExtractionResult` shape as the digital pipeline. All existing tests pass.

---

## Sub-phase 3.5 — Documentation & Exports [DONE]

**Goal:** Update public API surface, docs, and package exports.

**What to change:**

```
doc_intelligence/ocr/__init__.py       — public re-exports
doc_intelligence/__init__.py           — add OCR exports if needed
docs/                                  — update user-facing documentation
```

**Details:**

- `ocr/__init__.py` exports: `BaseLayoutDetector`, `BaseOCREngine`, `LayoutRegion`, `PaddleLayoutDetector`, `PaddleOCREngine`, `ScannedPDFParser`.
- Update `docs/` with:
  - Scanned PDF quickstart guide.
  - Custom layout detector / OCR engine extension guide.
  - API reference for new classes.
- Update `README.md` if API-level changes affect the getting-started flow.

**Verification:** `mkdocs build` (no warnings) + manual review.

**Exit criteria:** A new user can discover and use the scanned PDF pipeline from the docs alone.

---

## Dependency Graph

```
Sub-phase 3.0 (Schema migration)
    │
    ▼
Sub-phase 3.1 (ABCs)
    │
    ▼
Sub-phase 3.2 (Paddle implementations)
    │
    ▼
Sub-phase 3.3 (ScannedPDFParser)
    │
    ▼
Sub-phase 3.4 (Factory & integration)
    │
    ▼
Sub-phase 3.5 (Docs & exports)
```

Each sub-phase depends only on the previous one. The codebase is green after every sub-phase.

---

## Files Summary

| Sub-phase | New files | Modified files |
|---|---|---|
| 3.0 | — | `pdf/schemas.py`, `pdf/parser.py`, `pdf/formatter.py`, `pdf/utils.py`, `tests/pdf/test_schemas.py`, `tests/pdf/test_parser.py`, `tests/pdf/test_formatter.py`, `tests/pdf/test_utils.py`, `tests/conftest.py` |
| 3.1 | `ocr/__init__.py`, `ocr/base.py`, `tests/ocr/__init__.py`, `tests/ocr/test_base.py` | `tests/conftest.py` (add fakes) |
| 3.2 | `ocr/paddle.py`, `tests/ocr/test_paddle.py` | — |
| 3.3 | `pdf/parser.py` (ScannedPDFParser), `tests/pdf/test_ocr_parser.py` | — |
| 3.4 | — | `pdf/processor.py`, `extract.py`, `tests/pdf/test_processor.py` |
| 3.5 | — | `ocr/__init__.py`, `docs/`, `README.md` |
