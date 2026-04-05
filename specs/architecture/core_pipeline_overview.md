# doc_intelligence Pipeline

> A concise walkthrough of the PDF processing pipeline from input to structured output.

## 1. Input & Validation

The pipeline accepts a PDF file path and a Pydantic schema defining the desired output structure. Before any processing begins, the restrictions layer validates file size, page count, and schema nesting depth against configurable limits to reject oversized or overly complex requests early.

## 2. PDF Type Detection

The pipeline determines whether the PDF is digital (contains extractable text) or scanned (image-based). This classification decides which parser handles the document.

## 3. Parsing Phase

### 3.1 Digital PDF Parsing

`DigitalPDFParser` uses pdfplumber to extract text lines along with their bounding boxes directly from the PDF's text layer. Each line becomes a `TextBlock` with precise positional metadata.

### 3.2 Scanned PDF Parsing

`ScannedPDFParser` renders each page to an image via pypdfium2 and then runs an OCR pipeline to extract structured text. Two pipeline architectures are supported:

- **Two-Stage (Separate Layout + OCR):** Page images are batched through a layout detection model first, producing bounding boxes for text blocks, tables, and figures. All detected regions are then collected into a flat list and batched through a separate OCR model for text recognition. Results are reassembled back to their originating pages using region metadata.
- **Single-Stage (Unified Layout + OCR):** A single visual language model performs both layout detection and text recognition in one batched call per page, returning regions with spatial coordinates and recognized text together. No reassembly step is needed.

Both architectures produce the same output — `TextBlock` and `TableBlock` content with spatial metadata. The two-stage approach allows swapping layout or OCR models independently, while the single-stage approach is simpler with fewer inference passes.

## 4. PDFDocument Model

Both parsers produce a `PDFDocument` — a list of `Page` objects, each containing an ordered sequence of `ContentBlock` items (text, table, image, chart). This unified structure is the handoff point between parsing and downstream stages.

## 5. Formatting

`PDFFormatter` converts the `PDFDocument` into a line-numbered, `<page>`-tagged string optimized for LLM consumption. Each block is assigned an index so the LLM can reference specific content in its citations. Image and chart blocks are excluded from the output (reserved for future VLM support).

## 6. Extraction

### 6.1 Single Pass

The formatted text, the target Pydantic schema (augmented with citation fields), and a system prompt are sent to the LLM in a single call. The LLM returns structured data conforming to the schema, with block-level citations attached.

### 6.2 Multi Pass

Extraction happens in three sequential LLM calls: **Pass 1** extracts the raw data, **Pass 2** grounds each field to specific pages, and **Pass 3** narrows citations to specific block indices within the cited pages. This staged approach improves citation accuracy for complex documents.

## 7. Citation Enrichment

After extraction, `enrich_citations_with_bboxes()` maps block-level citations back to bounding box coordinates from the parsed `PDFDocument`. This converts abstract block references into pixel-level regions on each page.

## 8. ExtractionResult

The final output is an `ExtractionResult` containing `data` (the populated Pydantic model) and `metadata` (citation coordinates, token usage, processing info). Consumers use the data directly and the metadata for UI features like source highlighting.
