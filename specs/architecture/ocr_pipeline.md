# OCR Pipeline Architectures

The OCR pipeline supports two architectures for extracting structured text from scanned page images. Both produce the same output — pages with layout regions containing recognized text and spatial metadata.

### Pipeline Type 1: Two-Stage (Separate Layout + OCR)

This pipeline uses two distinct models in sequence:

**Stage 1 — Layout Detection (batched by page)**
Page images are batched (e.g., 10 at a time) through a layout detection model. Each page produces a list of regions — bounding boxes identifying text blocks, tables, figures, etc. Each region carries metadata linking it back to its source page.

**Stage 2 — OCR (batched by region)**
All detected regions across all pages are collected into a flat list and batched (e.g., 10 at a time) through an OCR model. Each region image is recognized independently, producing the text content for that region.

**Reassembly**
Since each region carries its page metadata, the OCR'd regions are mapped back to their originating pages to reconstruct the full document structure.

### Pipeline Type 2: Single-Stage (Unified Layout + OCR)

This pipeline uses a single model — typically a visual language model — that performs both layout detection and text recognition in one call.

**Single Stage — Layout + OCR (batched by page)**
Each page image is sent to the model with a prompt requesting bounding boxes and their contained text. The model returns regions with both spatial coordinates and recognized text in a single response. These calls are batched (e.g., 10 pages at a time).

No reassembly step is needed since results are already organized by page.

### Comparison

| | Type 1 (Two-Stage) | Type 2 (Single-Stage) |
|---|---|---|
| **Models** | Separate layout + OCR models | Single visual language model |
| **Inference passes** | Two (layout, then OCR) | One |
| **Batch unit** | Stage 1: pages, Stage 2: regions | Pages |
| **Flexibility** | Swap layout or OCR model independently | Coupled to one model |
| **Reassembly** | Required (regions → pages) | Not needed |
