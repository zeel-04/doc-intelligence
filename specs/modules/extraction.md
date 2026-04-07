# Module: Extraction

**Source:** `doc_intelligence/pdf/extractor.py`, `doc_intelligence/base.py`, `doc_intelligence/pydantic_to_json_instance_schema.py`, `doc_intelligence/pdf/utils.py`, `doc_intelligence/utils.py`
**Diagram ref:** "Extraction" (step 6 in the pipeline, after Formatting and before ExtractionResult)

---

## 1. Purpose

Convert formatted document text into structured data conforming to a caller-supplied Pydantic schema, optionally enriched with block-level citation metadata. The extractor is the LLM-facing layer of the pipeline — it builds prompts from formatted content, sends them to the LLM, parses JSON responses, and post-processes citations into bounding box coordinates. It is the only module that communicates with the LLM.

---

## 2. Pipeline Position

```
PDFDocument (from parser)
    │
    ▼
  Formatting Phase
    │
    ▼
  Formatted string
    │
    ▼
► Extraction (this module)     ← PDFExtractor.extract(document, request, formatter)
    │
    ▼
  ExtractionResult (data + metadata)
```

**Called by:** `DocumentProcessor.extract()` in `pdf/processor.py`.
**Runs after:** Formatting phase converts `PDFDocument` into an LLM-consumable string (the formatter is called _within_ the extractor, not before it).
**Produces:** `ExtractionResult` — the final pipeline output returned to the caller.

---

## 3. Inputs & Outputs

### 3.1 Inputs

| Parameter | Type | Description |
|---|---|---|
| `document` | `Document` | Parsed document (e.g. `PDFDocument`) with populated `content` |
| `request` | `ExtractionRequest` | Carries `uri`, `response_format`, `include_citations`, `llm_config`, and (for `PDFExtractionRequest`) `extraction_mode` and `page_numbers` |
| `formatter` | `BaseFormatter` | Used internally to render the document for each LLM call |

### 3.2 Outputs

| Output | Type | Description |
|---|---|---|
| Return value | `ExtractionResult` | `data`: populated Pydantic model instance; `metadata`: citation dict with bounding boxes (or `None` if citations disabled) |

---

## 4. Public API

### 4.1 `PDFExtractor(llm: BaseLLM)`

Constructor. Accepts a `BaseLLM` instance and initializes the system/user prompt templates. Also inherits a `JsonOutputParser` (`self.json_parser`) from `BaseExtractor` for parsing raw LLM text into Python dicts.

### 4.2 `PDFExtractor.extract(document, request, formatter) -> ExtractionResult`

Main entry point. Narrows the generic `ExtractionRequest` to `PDFExtractionRequest` (via `_as_pdf_request`), then dispatches to single-pass or multi-pass based on `request.extraction_mode`.

### 4.3 Module-Level Helper: `_as_pdf_request(request: ExtractionRequest) -> PDFExtractionRequest`

If the request is already a `PDFExtractionRequest`, returns it directly. Otherwise wraps it with PDF defaults (`SINGLE_PASS` mode, no page filter).

---

## 5. Internal Design

### 5.1 Schema Transformation

Before any LLM call, the caller's Pydantic schema is transformed into a **JSON instance schema** via `pydantic_to_json_instance_schema()`. This produces a dict that looks like the expected JSON output rather than a JSON Schema definition — it uses `<type>` placeholders, inline comments from field descriptions/examples/defaults, and (when citations are enabled) wraps every leaf field in a `{"value": "<type>", "citations": [...]}` structure.

`stringify_schema()` renders this dict into a formatted string with inline `# comments`, which becomes part of the user prompt.

Two schema variants are used across the pipeline:

| Variant | `citation` param | Used in |
|---|---|---|
| Citation-aware | `citation=True, citation_level="block"` | Single-pass; Multi-pass Pass 3 |
| Plain (no citations) | `citation=False` | Multi-pass Pass 1 |

**Example — citation-aware schema:**

```
{
    "name": {
        "value": <string>,  # desc: person name
        "citations": [{"page": <integer>, "blocks": [<integer>]}]
    }
}
```

**Example — plain schema:**

```
{
    "name": <string>
}
```

### 5.2 Single-Pass Mode (`_run_single_pass`)

Everything happens in one LLM call:

1. **Schema transformation** — `pydantic_to_json_instance_schema(response_format, citation=include_citations, citation_level="block")`.
2. **Format document** — `formatter.format_document_for_llm(document, page_numbers=..., include_citations=...)`. The document is formatted with block tags when citations are enabled.
3. **Build prompt** — Interpolates `content_text` and `schema` into the user prompt template.
4. **LLM call** — `self.llm.generate(system_prompt, user_prompt, **llm_config)`.
5. **Parse response** — `self.json_parser.parse(response)` extracts a Python dict from the LLM's JSON text.
6. **Post-process citations** — If citations are enabled: `enrich_citations_with_bboxes()` resolves block indices to bounding box coordinates, then `strip_citations()` unwraps `{"value": ..., "citations": [...]}` structures into plain values for the `data` field.
7. **Build result** — `ExtractionResult(data=response_format(**response_dict), metadata=response_metadata)`.

```
Schema + Formatted text + Prompts
    │
    ▼
  LLM call (1 call)
    │
    ▼
  JSON response
    │
    ├── citations enabled ──► enrich_citations_with_bboxes() → metadata
    │                          strip_citations() → plain dict → data
    │
    └── citations disabled ──► plain dict → data, metadata=None
```

### 5.3 Multi-Pass Mode (`_run_multi_pass`)

Extraction is split into three sequential LLM calls. This staged approach improves citation accuracy for complex documents by separating the extraction task from the grounding task.

#### Pass 1 — Raw Extraction

Extracts the data without any citation overhead.

- Schema: **plain** (no citation wrappers).
- Document formatting: **without** block tags (`include_citations=False`).
- Prompt: standard system + user prompt (same templates as single-pass).
- Output: A populated `PydanticModel` instance (`pass1_result`).
- If `include_citations` is `False`, this is the final result — Passes 2 and 3 are skipped entirely.

#### Pass 2 — Page Grounding

Identifies which page(s) contain the supporting text for each extracted field.

- Schema: none — the LLM is asked to return a simple `{"field_path": [page_numbers]}` mapping.
- Document formatting: **with** block tags (`include_citations=True`) — the LLM can see the page structure.
- Prompt: dedicated `_PASS2_SYSTEM_PROMPT` and `_PASS2_USER_PROMPT` templates. The user prompt includes both the formatted document and the Pass 1 JSON output.
- Output: `page_map: dict[str, list[int]]` — field paths mapped to lists of 0-indexed page numbers.

#### Pass 3 — Block Grounding

Narrows citations from page-level to block-level, operating only on the relevant pages identified in Pass 2.

- **Page scoping:** All unique pages from `page_map` are collected. If the original request had `page_numbers`, they are intersected with the Pass 2 pages to avoid processing irrelevant content.
- Schema: **citation-aware** (`citation=True, citation_level="block"`).
- Document formatting: **with** block tags, restricted to the scoped pages only.
- Prompt: dedicated `_PASS3_USER_PROMPT` template. Includes the Pass 1 JSON, the scoped document text, and the citation-aware schema.
- Output: `enrich_citations_with_bboxes()` is called directly on the response dict to produce the final metadata.

The `data` field of the final `ExtractionResult` is the Pass 1 result (the clean Pydantic model without citation wrappers). The `metadata` field is the enriched citation dict from Pass 3.

```
Pass 1: Extract data (no citations)
    │
    ▼
  pass1_result (PydanticModel)
    │
    ├── citations disabled ──► ExtractionResult(data=pass1_result, metadata=None)
    │
    └── citations enabled
         │
         ▼
       Pass 2: Page grounding
         │
         ▼
       page_map: {"field": [page_numbers]}
         │
         ▼
       Pass 3: Block grounding (scoped to relevant pages)
         │
         ▼
       metadata (enriched with bboxes)
         │
         ▼
       ExtractionResult(data=pass1_result, metadata=metadata)
```

### 5.4 Prompt Templates

| Template | Role | Used in |
|---|---|---|
| `self.system_prompt` | System | Single-pass, Pass 1, Pass 3 |
| `self.user_prompt` | User | Single-pass, Pass 1 |
| `_PASS2_SYSTEM_PROMPT` | System | Pass 2 |
| `_PASS2_USER_PROMPT` | User | Pass 2 |
| `_PASS3_USER_PROMPT` | User | Pass 3 |

System prompts are static. User prompts are formatted with `str.format()` using these variables:

| Variable | Source |
|---|---|
| `{content_text}` | `formatter.format_document_for_llm(...)` output |
| `{schema}` | `stringify_schema(pydantic_to_json_instance_schema(...))` output |
| `{pass1_json}` | `pass1_result.model_dump_json()` |

### 5.5 Citation Post-Processing

Two utilities handle the transformation from raw LLM citation output to the final metadata structure:

#### `enrich_citations_with_bboxes(response, document)` (`pdf/utils.py`)

Recursively traverses the response dict to find citation dicts (objects with both `page` and `blocks` keys). For each citation:

1. Looks up the page in `document.content.pages` by the `page` index.
2. Builds a **citable blocks** list for that page — all blocks excluding `ImageBlock` and `ChartBlock` (matching the formatter's numbering convention).
3. Resolves each block index to its bounding box from the citable blocks list.
4. Replaces the `blocks` key with a `bboxes` key containing a list of `BoundingBox` dicts.

Out-of-range page or block indices are silently ignored (no bbox added). Blocks without a bounding box are also skipped.

#### `strip_citations(response)` (`utils.py`)

Recursively unwraps `{"value": ..., "citations": [...]}` structures into just the value. Used in single-pass mode to separate the data dict (for Pydantic model instantiation) from the metadata (already enriched).

### 5.6 JSON Parsing

`BaseExtractor` initializes a `JsonOutputParser` from `langchain_core.output_parsers`. This parser extracts valid JSON from the LLM's raw text response, handling markdown code fences and surrounding text. The parser is used after every LLM call to convert the response string into a Python dict.

---

## 6. Configuration

The extractor has no global configuration settings of its own. All behavior is controlled via:

| Setting | Source | Effect |
|---|---|---|
| `extraction_mode` | `PDFExtractionRequest` | Selects single-pass vs. multi-pass |
| `include_citations` | `PDFExtractionRequest` | Toggles citation schema wrapping, block tags in formatting, and post-processing |
| `page_numbers` | `PDFExtractionRequest` | Restricts which pages are formatted for the LLM |
| `llm_config` | `PDFExtractionRequest` | Forwarded as `**kwargs` to `BaseLLM.generate()` on every call |

LLM generation parameters (temperature, max tokens, etc.) are passed through `llm_config` — the extractor does not interpret them.

---

## 7. Constraints & Invariants

- **Extractor owns LLM communication.** No other module calls the LLM. The extractor is the sole boundary between the pipeline and the LLM provider.
- **Formatter is called within the extractor.** The extractor receives the formatter as a dependency and calls it internally — potentially multiple times (once per pass in multi-pass mode) with different parameters.
- **Single-pass: one LLM call.** Multi-pass: exactly three LLM calls when citations are enabled, one when citations are disabled.
- **Pass 1 result is the data, Pass 3 result is the metadata.** In multi-pass mode, the `data` field is always the Pass 1 Pydantic model instance. Citation metadata comes from Pass 3 only. The `data` never contains citation wrappers.
- **Block indices are formatter-relative.** Citation block indices reference the same numbering the formatter produces — `ImageBlock` and `ChartBlock` are excluded from the count.
- **Schema transformation is deterministic.** The same Pydantic model always produces the same JSON instance schema for a given `citation` and `citation_level` setting.
- **No document mutation.** The extractor never modifies the `Document` or its content. Formatting creates a view; extraction creates new dicts and model instances.
- **Request narrowing is safe.** `_as_pdf_request` preserves all fields when the request is already a `PDFExtractionRequest`. For generic `ExtractionRequest`, defaults are applied (`SINGLE_PASS`, no page filter).

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| Unsupported `extraction_mode` value | `ValueError` from `extract()` |
| `document.content` is `None` during citation enrichment | `ValueError` from `enrich_citations_with_bboxes()` |
| LLM returns invalid/non-JSON response | Exception from `JsonOutputParser.parse()` |
| LLM response does not match expected schema | `ValidationError` from `response_format(**response_dict)` (Pydantic) |
| Out-of-range page index in citation | Silently ignored — no bbox added |
| Out-of-range block index in citation | Silently ignored — no bbox added |
| LLM provider errors (auth, rate limit, network) | Provider-specific exceptions, retried via `tenacity` in the LLM layer |

---

## 9. Extension Points

### 9.1 Custom Prompts

The system and user prompts are instance attributes set in `__init__`. A subclass or post-construction override can replace them for domain-specific extraction (e.g. legal documents, medical records) without changing the extraction logic.

### 9.2 New Extraction Modes

Add a new `PDFExtractionMode` enum value in `types.py` and a corresponding `_run_<mode>` method in `PDFExtractor`. Update the dispatch in `extract()`.

### 9.3 Structured Output

A future mode could leverage provider-native structured output to return a `PydanticModel` directly, bypassing the JSON instance schema transformation. This would require extending `BaseLLM.generate()` or adding a dedicated method. The current extractor uses `generate()` + `JsonOutputParser`.

### 9.4 New Document Types

New document types create their own extractor subclass of `BaseExtractor` in their respective module folder. The `BaseExtractor` interface requires only `extract(document, request, formatter) -> ExtractionResult`.

### 9.5 Citation Levels

The schema transformation already supports `citation_level="page"` (page-only citations without block indices). This could be exposed as a user-facing option if block-level precision is unnecessary for certain use cases.
