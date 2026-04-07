# doc_intelligence Pipeline

> A concise walkthrough of the PDF processing pipeline from input to structured output.

## 1. Input & Validation

The pipeline accepts a PDF file path and a Pydantic schema defining the desired output structure. Before any processing begins, the restrictions layer validates file size, page count, and schema nesting depth against configurable limits to reject oversized or overly complex requests early.

## 2. Strategy Selection

The caller specifies the parse strategy (`DIGITAL` or `SCANNED`) when constructing `PDFParser`. This selection decides the internal parsing path. Digital PDFs contain an extractable text layer; scanned PDFs are image-based and require OCR.

## 3. Parsing Phase

`PDFParser` is a unified, strategy-based parser that dispatches to the appropriate internal method based on the configured `ParseStrategy`.

### 3.1 Digital Strategy

Uses pdfplumber to extract text lines along with their bounding boxes directly from the PDF's text layer. Each line becomes a `TextBlock` with precise positional metadata.

### 3.2 Scanned Strategy

Renders each page to an image via pypdfium2 and then runs an OCR pipeline to extract structured text. The current implementation uses the **Two-Stage (Separate Layout + OCR)** architecture:

- A `BaseLayoutDetector` segments each page into typed regions (text, table, image, chart).
- A `BaseOCREngine` recognizes text within each non-image/non-chart region.
- Pages are processed sequentially; regions within a page are OCR'd concurrently.

A **Single-Stage (Unified Layout + OCR)** architecture — where a single visual language model performs both layout detection and text recognition in one call — is described in `specs/architecture/ocr_pipeline.md` but not yet implemented.

Both architectures produce the same output — `TextBlock` and `TableBlock` content with spatial metadata. Only these two block types are fully supported end-to-end. `ImageBlock` and `ChartBlock` are created as content-less placeholders preserving layout structure, pending future VLM support.

## 4. PDFDocument Model

`PDFParser` produces a `PDFDocument` — a list of `Page` objects, each containing an ordered sequence of `ContentBlock` items (text, table, image, chart). This unified structure is the handoff point between parsing and downstream stages.

## 5. Formatting

`PDFFormatter` converts the `PDFDocument` into a block-indexed, `<page>`-tagged string optimized for LLM consumption. Each block is assigned an index so the LLM can reference specific content in its citations. Image and chart blocks are excluded from the output (reserved for future VLM support).

## 6. Extraction

### 6.1 Single Pass

The formatted text, the target Pydantic schema (augmented with citation fields), and a system prompt are sent to the LLM in a single call. The LLM returns structured data conforming to the schema, with block-level citations attached.

### 6.2 Multi Pass

Extraction happens in three sequential LLM calls: **Pass 1** extracts the raw data, **Pass 2** grounds each field to specific pages, and **Pass 3** narrows citations to specific block indices within the cited pages. This staged approach improves citation accuracy for complex documents.

## 7. Citation Enrichment

After extraction, `enrich_citations_with_bboxes()` maps block-level citations back to bounding box coordinates from the parsed `PDFDocument`. This converts abstract block references into pixel-level regions on each page.

## 8. ExtractionResult

The final output is an `ExtractionResult` containing `data` (the populated Pydantic model) and `metadata` (citation coordinates). Consumers use the data directly and the metadata for UI features like source highlighting.
