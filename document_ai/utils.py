from typing import Any, get_args, get_origin

from pydantic import BaseModel, create_model

from .schemas.core import PDF, BoundingBox, PydanticModel

CITATION_DESCRIPTION = """This is used to cite the page number and line number where the information is mentioned in the document.
For example:
[{"page": 1, "lines": [10, 11]}, {"page": 2, "lines": [20]}]"""


def normalize_bounding_box(
    bounding_box: BoundingBox, page_width: int | float, page_height: int | float
) -> BoundingBox:
    return BoundingBox(
        x0=bounding_box.x0 / page_width,
        top=bounding_box.top / page_height,
        x1=bounding_box.x1 / page_width,
        bottom=bounding_box.bottom / page_height,
    )


def denormalize_bounding_box(
    bounding_box: BoundingBox, page_width: int | float, page_height: int | float
) -> BoundingBox:
    return BoundingBox(
        x0=bounding_box.x0 * page_width,
        top=bounding_box.top * page_height,
        x1=bounding_box.x1 * page_width,
        bottom=bounding_box.bottom * page_height,
    )


def enrich_citations_with_bboxes(
    response: type[PydanticModel], parsed_pdf: PDF
) -> dict[str, Any]:
    """
    Enriches citation fields in the response with bounding boxes from the parsed PDF.

    This function traverses the response dictionary recursively to find all citation
    dictionaries (identified by having both 'page' and 'lines' keys), then adds
    bounding box information from the corresponding lines in the parsed PDF.

    Args:
        response: The Pydantic model response from LLM extraction
        parsed_pdf: The parsed PDF document containing pages and line bounding boxes

    Returns:
        A dictionary with bboxes added to all citation fields. Each citation will
        have a 'bboxes' key containing a list of normalized BoundingBox dicts.

    Note:
        - The Citation schema is NOT modified - bboxes are only added in the returned
          dict, keeping the LLM-facing schema clean
        - Bounding boxes are normalized (0-1 scale)
        - Handles out-of-bounds page/line indices gracefully by skipping them
    """

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

        # Create enriched citation with bboxes
        enriched = citation.copy()
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

    # Convert response to dict and enrich
    response_dict = response.model_dump()  # type:ignore
    return _traverse_and_enrich(response_dict)


def is_citation_type(field_type: Any, citation_type: Any) -> bool:
    """
    Check if a field type matches the provided citation type.

    Args:
        field_type: The field type to check
        citation_type: The citation type to match against (e.g., processor.citation_type)

    Returns:
        True if the field type matches the citation type
    """
    # Direct comparison
    if field_type == citation_type:
        return True

    # Check if both are generic list types with same args
    field_origin = get_origin(field_type)
    citation_origin = get_origin(citation_type)

    if field_origin is list and citation_origin is list:
        field_args = get_args(field_type)
        citation_args = get_args(citation_type)

        if field_args and citation_args:
            # Compare the inner types
            if field_args[0] == citation_args[0]:
                return True
            # Also check by name for TypedDict comparisons
            if hasattr(field_args[0], "__name__") and hasattr(
                citation_args[0], "__name__"
            ):
                if field_args[0].__name__ == citation_args[0].__name__:
                    return True

    return False


def find_citation_fields(
    model: type[BaseModel], citation_type: Any, prefix: str = ""
) -> list[str]:
    """
    Recursively find all fields with citation type in a Pydantic model.

    Args:
        model: The Pydantic model to inspect
        citation_type: The citation type to search for (e.g., processor.citation_type)
        prefix: Current field path prefix for nested models

    Returns:
        List of field paths (e.g., ['my_data_citation', 'nested.citation_field'])
    """
    citation_fields = []

    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        current_path = f"{prefix}.{field_name}" if prefix else field_name

        # Check if this field is a citation type
        if is_citation_type(field_type, citation_type):
            citation_fields.append(current_path)

        # Check if this field is a nested BaseModel
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_fields = find_citation_fields(
                field_type, citation_type, current_path
            )
            citation_fields.extend(nested_fields)

        # Check if it's a list of BaseModel
        elif get_origin(field_type) is list:
            args = get_args(field_type)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                nested_fields = find_citation_fields(
                    args[0], citation_type, current_path
                )
                citation_fields.extend(nested_fields)

    return citation_fields


def add_bboxes_to_citation_model(
    model: type[BaseModel], original_citation_type: Any, new_citation_type: Any
) -> type[BaseModel]:
    """
    Recursively modify a Pydantic model to replace citation type with new citation type.

    Args:
        model: The original Pydantic model
        original_citation_type: The citation type to replace (e.g., processor.citation_type)
        new_citation_type: The new citation type (e.g., processor.citation_type_with_bbox)

    Returns:
        A new model class with updated citation types
    """
    new_fields = {}

    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        default = field_info.default if field_info.default is not None else ...

        # If it's a citation field, update the type
        if is_citation_type(field_type, original_citation_type):
            new_fields[field_name] = (new_citation_type, default)

        # If it's a nested BaseModel, recursively modify it
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            modified_nested = add_bboxes_to_citation_model(
                field_type, original_citation_type, new_citation_type
            )
            new_fields[field_name] = (modified_nested, default)

        # If it's a list of BaseModel, modify the inner model
        elif get_origin(field_type) is list:
            args = get_args(field_type)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                modified_nested = add_bboxes_to_citation_model(
                    args[0], original_citation_type, new_citation_type
                )
                new_fields[field_name] = (list[modified_nested], default)  # type: ignore
            else:
                new_fields[field_name] = (field_type, default)

        # Otherwise keep the original field
        else:
            new_fields[field_name] = (field_type, default)

    # Create new model with modified fields
    return create_model(f"{model.__name__}WithBBox", __base__=BaseModel, **new_fields)
