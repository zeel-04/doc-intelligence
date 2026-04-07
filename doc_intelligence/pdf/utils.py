"""PDF-specific utilities for citation enrichment."""

from typing import Any

from doc_intelligence.pdf.schemas import PDF
from doc_intelligence.schemas.core import ChartBlock, Document, ImageBlock


def enrich_citations_with_bboxes(
    response: dict[str, Any], document: Document
) -> dict[str, Any]:
    """Enrich citation fields in the response dict with bounding boxes.

    Traverses the response dictionary recursively to find all citation
    dictionaries (identified by having both ``page`` and ``blocks`` keys),
    then replaces ``blocks`` with ``bboxes`` looked up from the
    corresponding content blocks in the parsed document.

    Block indices refer to the *citable* block ordering — i.e. the order
    that ``ImageBlock`` and ``ChartBlock`` are excluded from (matching the
    formatter's numbering).

    Args:
        response: The response dictionary (e.g. from LLM structured output).
        document: The Document instance (e.g. PDFDocument) with parsed content.

    Returns:
        A dictionary with bboxes added to all citation fields and ``blocks``
        removed.  Each citation will have ``page`` and ``bboxes`` (a list of
        normalized BoundingBox dicts).

    Raises:
        ValueError: If document.content is None (document not parsed yet).
    """
    if document.content is None:
        raise ValueError(
            "Document content is None. Parse the document before enriching citations."
        )

    parsed_pdf: PDF = document.content  # type: ignore[assignment]

    def _citable_blocks(page_idx: int) -> list:
        """Return the citable blocks for a page, excluding image/chart."""
        if page_idx < 0 or page_idx >= len(parsed_pdf.pages):
            return []
        return [
            b
            for b in parsed_pdf.pages[page_idx].blocks
            if not isinstance(b, (ImageBlock, ChartBlock))
        ]

    def _is_citation_dict(obj: Any) -> bool:
        """Check if an object is a citation dictionary."""
        return (
            isinstance(obj, dict)
            and "page" in obj
            and "blocks" in obj
            and isinstance(obj.get("page"), int)
            and isinstance(obj.get("blocks"), list)
        )

    def _enrich_citation(citation: dict[str, Any]) -> dict[str, Any]:
        """Add bboxes to a single citation dictionary."""
        page_idx = citation["page"]
        block_indices = citation["blocks"]

        citable = _citable_blocks(page_idx)
        bboxes = []

        for block_idx in block_indices:
            if 0 <= block_idx < len(citable):
                block = citable[block_idx]
                if block.bounding_box is not None:
                    bboxes.append(block.bounding_box.model_dump())

        enriched = {k: v for k, v in citation.items() if k != "blocks"}
        enriched["bboxes"] = bboxes
        return enriched

    def _traverse_and_enrich(obj: Any) -> Any:
        """Recursively traverse and enrich citation dictionaries."""
        if _is_citation_dict(obj):
            return _enrich_citation(obj)
        elif isinstance(obj, dict):
            return {key: _traverse_and_enrich(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_traverse_and_enrich(item) for item in obj]
        else:
            return obj

    return _traverse_and_enrich(response)
