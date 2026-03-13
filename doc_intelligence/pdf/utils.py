from typing import Any

from doc_intelligence.pdf.schemas import PDF
from doc_intelligence.schemas.core import Document


def enrich_citations_with_bboxes(
    response: dict[str, Any], document: Document
) -> dict[str, Any]:
    """
    Enriches citation fields in the response dict with bounding boxes from the document.

    This function traverses the response dictionary recursively to find all citation
    dictionaries (identified by having both 'page' and 'lines' keys), then replaces
    'lines' with 'bboxes' looked up from the corresponding lines in the parsed document.

    Args:
        response: The response dictionary (e.g. from LLM structured output)
        document: The Document instance (e.g. PDFDocument) with parsed content

    Returns:
        A dictionary with bboxes added to all citation fields and 'lines' removed.
        Each citation will have 'page' and 'bboxes' (a list of normalized BoundingBox dicts).

    Raises:
        ValueError: If document.content is None (document not parsed yet)

    Note:
        - The 'lines' key is removed from citations in the output since bboxes
          replace them for downstream use
        - Bounding boxes are normalized (0-1 scale)
        - Handles out-of-bounds page/line indices gracefully by skipping them
    """
    if document.content is None:
        raise ValueError(
            "Document content is None. Parse the document before enriching citations."
        )

    parsed_pdf: PDF = document.content  # type: ignore[assignment]

    def _is_citation_dict(obj: Any) -> bool:
        """Check if an object is a citation dictionary."""
        return (
            isinstance(obj, dict)
            and "page" in obj
            and "lines" in obj
            and isinstance(obj.get("page"), int)
            and isinstance(obj.get("lines"), list)
        )

    def _enrich_citation(citation: dict[str, Any]) -> dict[str, Any]:
        """Add bboxes to a single citation dictionary."""
        page_idx = citation["page"]
        line_indices = citation["lines"]

        # Bounds check for page
        if page_idx < 0 or page_idx >= len(parsed_pdf.pages):
            # Page index out of bounds, return citation as-is
            return citation

        page = parsed_pdf.pages[page_idx]
        bboxes = []

        for line_idx in line_indices:
            # Bounds check for line
            if line_idx >= 0 and line_idx < len(page.lines):
                bbox = page.lines[line_idx].bounding_box
                # Convert BoundingBox to dict
                bboxes.append(bbox.model_dump())

        # Create enriched citation: add bboxes, remove lines
        enriched = {k: v for k, v in citation.items() if k != "lines"}
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
