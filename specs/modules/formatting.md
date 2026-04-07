# Module: Formatting

**Source:** `doc_intelligence/pdf/formatter.py`, `doc_intelligence/base.py`
**Diagram ref:** "Formatting" (step 5 in the pipeline, between PDFDocument and Extraction)

---

## 1. Purpose

Convert a parsed `PDFDocument` into an LLM-consumable string with optional block-level citation markup. The formatter is the bridge between structured document data and the text prompts sent to the LLM — it controls what the model sees and how it can reference content back.

---

## 2. Pipeline Position

```
PDFDocument (from parser)
    │
    ▼
► Formatting (this module)     ← PDFFormatter.format_document_for_llm(document)
    │
    ▼
  Formatted string
    │
    ▼
  Extraction Phase (LLM calls) ...
```

**Called by:** `PDFExtractor.extract()` in `pdf/extractor.py` (for both single-pass and multi-pass modes).
**Runs after:** Parsing phase produces a `PDFDocument`.
**Produces:** A string consumed by the LLM via system/user prompts in the extraction phase.

---

## 3. Inputs & Outputs

### 3.1 Inputs

| Parameter | Type | Description |
|---|---|---|
| `document` | `Document` | Must have non-`None` `content` (a `PDF` object with pages) |
| `page_numbers` | `list[int] \| None` (kwarg) | Optional 0-indexed page filter — only format these pages |
| `include_citations` | `bool` (kwarg) | Whether to emit `<block>` tags with indices (default `True`) |

### 3.2 Outputs

| Output | Type | Description |
|---|---|---|
| Return value | `str` | Formatted text with `<page>` tags, optionally containing `<block>` tags |

---

## 4. Public API

### 4.1 `PDFFormatter.format_document_for_llm(document, **kwargs) -> str`

Main entry point. Accepts a `Document` and optional keyword arguments:

- `page_numbers: list[int] | None` — restricts output to specific pages (0-indexed). Duplicates are removed and pages are sorted.
- `include_citations: bool` — toggles between block-indexed output (default) and plain text output.

Returns pages joined by double newlines (`\n\n`).

### 4.2 Module-Level Helper: `_render_block_text(block) -> str`

Renders a single `ContentBlock` to its text representation:

- `TextBlock` — lines joined by `\n`.
- `TableBlock` — rows rendered as pipe-delimited markdown table rows (`| cell1 | cell2 |`).
- `ImageBlock` / `ChartBlock` — returns empty string (silently skipped).

---

## 5. Internal Design

### 5.1 Two Formatting Modes

The formatter has two internal rendering paths, selected by the `include_citations` kwarg:

#### With citations (`_format_with_block_indices`) — default

```xml
<page number="0">
<block index="0" type="text">
Line one of first block
Line two of first block
</block>
<block index="1" type="table">
| col1 | col2 |
| val1 | val2 |
</block>
</page>
```

- Each visible block gets a sequential `index` attribute (0-based per page).
- The `type` attribute reflects the block's `block_type` field.
- `ImageBlock` and `ChartBlock` are skipped entirely — they do not receive an index and do not appear in output.
- Block indices are contiguous: if blocks 0 and 2 are text and block 1 is an image, the output indices are 0 and 1.

#### Without citations (`_format_without_block_indices`)

```xml
<page number="0">
Line one of first block
Line two of first block
| col1 | col2 |
| val1 | val2 |
</page>
```

- Plain text with `<page>` wrappers only — no `<block>` tags.
- Same skip behavior for `ImageBlock` / `ChartBlock`.

### 5.2 Page Filtering

When `page_numbers` is provided:

1. Deduplicated via `set()` and sorted.
2. Only pages whose 0-based index appears in the list are included.
3. The original `PDFDocument` is **never mutated** — a temporary `PDF` view is created with the filtered page list.
4. Page `number` attributes in the output reflect the original page index, not the position in the filtered list.

### 5.3 Block Index Assignment

Block indices are assigned per-page during formatting, not stored on the block model. The index counter skips `ImageBlock` and `ChartBlock` instances, ensuring that citation block indices map directly to the visible blocks in the formatted output.

This means the same block may receive different indices if different page subsets are formatted — block indices are output-relative, not document-absolute.

---

## 6. Configuration

The formatter has no global configuration settings. All behavior is controlled via constructor defaults and keyword arguments at call time.

---

## 7. Constraints & Invariants

- **No mutation.** The formatter never modifies the input `Document` or its content. Page filtering creates a shallow copy.
- **Content must be set.** `format_document_for_llm` raises `ValueError` if `document.content` is `None`.
- **Pages must be set.** Both internal methods raise `ValueError` if `content.pages` is empty/`None`.
- **Only `TextBlock` and `TableBlock` are supported.** These are the only block types that produce output. `ImageBlock` and `ChartBlock` are excluded from all formatted output and do not consume block indices — they are placeholders created during parsing to preserve layout structure, but carry no content. Full support is planned when VLM integration is added.
- **Block indices are dense.** No gaps in the index sequence — every supported block gets the next integer. This simplifies LLM citation parsing.
- **Page numbers are 0-indexed.** Both in `page_numbers` filter and in the `<page number="...">` attribute.

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| `document.content` is `None` | `ValueError` with message to parse first |
| `content.pages` is empty or `None` | `ValueError` from internal formatting methods |
| `page_numbers` contains out-of-range indices | Those indices are silently ignored (no matching page) |
| `page_numbers` is empty list | All pages are formatted (falsy check passes through) |
| Unknown block type | `_render_block_text` returns empty string — block is effectively skipped |

---

## 9. Extension Points

### 9.1 New Block Types / VLM Support

When `ImageBlock` and `ChartBlock` gain content (via future VLM integration), add rendering branches to `_render_block_text` and include them in block index assignment. The same applies to any new block types added to `ContentBlock` (e.g., `CodeBlock`).

### 9.2 Alternative Output Formats

Subclass `BaseFormatter` to produce different output formats (e.g., JSON, markdown with headers). The `BaseFormatter` interface requires only `format_document_for_llm(document, **kwargs) -> str`.

### 9.3 New Document Types

New document types create their own formatter subclass of `BaseFormatter` in their respective module folder. The formatter interface is document-type agnostic — only the internal rendering logic changes.
