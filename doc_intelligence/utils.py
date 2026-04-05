"""General-purpose utilities for citation handling and bounding-box transforms."""

from typing import Any

from doc_intelligence.schemas.core import BoundingBox


def normalize_bounding_box(
    bounding_box: BoundingBox, page_width: int | float, page_height: int | float
) -> BoundingBox:
    """Scale absolute bounding-box coordinates to the 0-1 range.

    Args:
        bounding_box: The bounding box with absolute pixel coordinates.
        page_width: The width of the page in pixels/points.
        page_height: The height of the page in pixels/points.

    Returns:
        A new BoundingBox with coordinates normalized to [0, 1].

    Raises:
        ValueError: If page_width or page_height is not positive.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(
            f"Page dimensions must be positive, got width={page_width}, height={page_height}"
        )
    return BoundingBox(
        x0=bounding_box.x0 / page_width,
        top=bounding_box.top / page_height,
        x1=bounding_box.x1 / page_width,
        bottom=bounding_box.bottom / page_height,
    )


def denormalize_bounding_box(
    bounding_box: BoundingBox, page_width: int | float, page_height: int | float
) -> BoundingBox:
    """Scale normalized bounding-box coordinates back to absolute values.

    Args:
        bounding_box: The bounding box with normalized [0, 1] coordinates.
        page_width: The width of the page in pixels/points.
        page_height: The height of the page in pixels/points.

    Returns:
        A new BoundingBox with absolute pixel/point coordinates.

    Raises:
        ValueError: If page_width or page_height is not positive.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(
            f"Page dimensions must be positive, got width={page_width}, height={page_height}"
        )
    return BoundingBox(
        x0=bounding_box.x0 * page_width,
        top=bounding_box.top * page_height,
        x1=bounding_box.x1 * page_width,
        bottom=bounding_box.bottom * page_height,
    )


def strip_citations(response: dict[str, Any]) -> dict[str, Any]:
    """
    Strips citation wrappers from a response dict, returning only the plain values.

    Recursively traverses the dict and unwraps any ``{'value': ..., 'citations': [...]}``
    structure into just the value.

    Args:
        response: The response dictionary with citation-wrapped values.

    Returns:
        A plain dictionary with citations removed and values unwrapped.

    Example::

        >>> strip_citations({
        ...     'name': {'value': 'Zeel', 'citations': [{'page': 1, 'blocks': [0]}]},
        ...     'ids': [
        ...         {'value': 101, 'citations': [{'page': 1, 'blocks': [0]}]},
        ...         {'value': 205, 'citations': [{'page': 1, 'blocks': [1]}]},
        ...     ],
        ... })
        {'name': 'Zeel', 'ids': [101, 205]}
    """

    def _is_value_citation_dict(obj: Any) -> bool:
        """Check if an object is exactly a {'value': ..., 'citations': [...]} wrapper."""
        return isinstance(obj, dict) and obj.keys() == {"value", "citations"}

    def _strip(obj: Any) -> Any:
        if _is_value_citation_dict(obj):
            return obj["value"]
        elif isinstance(obj, dict):
            return {key: _strip(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_strip(item) for item in obj]
        else:
            return obj

    return _strip(response)
