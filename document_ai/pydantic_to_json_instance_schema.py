import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined


def get_type_string(field_type: Any) -> str:
    """
    Convert Python type to string representation for schema.

    Args:
        field_type: The type annotation from Pydantic field

    Returns:
        String representation like "string", "integer", "number", "boolean", "object"
    """
    origin = get_origin(field_type)

    # Handle Optional/Union types
    if origin is Union:
        args = get_args(field_type)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return get_type_string(non_none_args[0])

    # Handle list types
    if origin is list:
        args = get_args(field_type)
        if args:
            return get_type_string(args[0])
        return "string"

    # Handle dict types
    if origin is dict:
        return "object"

    # Check if it's an Enum
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return "string"

    # Map basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        datetime: "string",
        date: "string",
    }

    return type_map.get(field_type, "string")


def is_nested_model(field_type: Any) -> bool:
    """
    Check if a type is a nested BaseModel (unwrapping Optional/Union/list first).

    Args:
        field_type: The type annotation to check

    Returns:
        True if the type is a BaseModel or contains a BaseModel
    """
    origin = get_origin(field_type)

    # Unwrap Union/Optional
    if origin is Union:
        args = get_args(field_type)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            field_type = non_none_args[0]
            origin = get_origin(field_type)

    # Unwrap list
    if origin is list:
        args = get_args(field_type)
        if args:
            field_type = args[0]

    # Check if BaseModel
    return isinstance(field_type, type) and issubclass(field_type, BaseModel)


def get_nested_model_type(field_type: Any) -> type[BaseModel]:
    """
    Extract the BaseModel type from Optional/Union/list wrappers.

    Args:
        field_type: The wrapped type annotation

    Returns:
        The unwrapped BaseModel type
    """
    origin = get_origin(field_type)

    # Unwrap Union/Optional
    if origin is Union:
        args = get_args(field_type)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            field_type = non_none_args[0]
            origin = get_origin(field_type)

    # Unwrap list
    if origin is list:
        args = get_args(field_type)
        if args:
            field_type = args[0]

    return field_type


def is_list_type(field_type: Any) -> bool:
    """
    Check if a type is a list type (unwrapping Optional/Union first).

    Args:
        field_type: The type annotation to check

    Returns:
        True if the (unwrapped) type is a list
    """
    origin = get_origin(field_type)

    # Unwrap Union/Optional
    if origin is Union:
        args = get_args(field_type)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            field_type = non_none_args[0]
            origin = get_origin(field_type)

    return origin is list


def _citation_item_schema(citation_level: Literal["page", "line"]) -> dict:
    """Build a single citation item schema based on citation_level."""
    item: dict[str, Any] = {"page": "<integer>"}
    if citation_level == "line":
        item["lines"] = ["<integer>"]
    return item


def pydantic_to_json_instance_schema(
    model: type[BaseModel],
    citation_level: Literal["page", "line"] = "line",
    citation: bool = True,
) -> dict:
    """
    Transform a Pydantic model into a JSON instance schema.

    When ``citation=True`` (default), the output wraps every leaf field in a
    ``{"value": "<type>", "citations": [...]}`` structure so the LLM is
    prompted to return supporting citations alongside each value.

    When ``citation=False``, the output is a plain instance schema that
    mirrors the user's Pydantic model without adding ``value`` /
    ``citations`` wrappers (leaf fields are simply ``"<type>"``).

    Args:
        model: Pydantic BaseModel class to transform
        citation_level: If "page", citations include only page; if "line", citations
            include both page and lines. Default "line". Ignored when
            ``citation=False``.
        citation: If True (default), wrap leaf fields with value/citations.
            If False, return plain type placeholders without citations.

    Returns:
        Dictionary representing the schema

    Example:
        >>> class Address(BaseModel):
        ...     street: str = Field(..., description="street name")
        >>> schema = pydantic_to_json_instance_schema(Address, citation_level="line")
        >>> schema
        {
            "street": {
                "value": "<string>",
                "comment": "desc: street name",
                "citations": [{"page": "<integer>", "lines": ["<integer>"]}]
            }
        }
        >>> schema_no_cite = pydantic_to_json_instance_schema(Address, citation=False)
        >>> schema_no_cite
        {"street": "<string>"}
    """
    schema = {}

    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        description = field_info.description or ""
        examples = field_info.examples or []

        # Check for default value (exclude PydanticUndefined)
        default = None
        if (
            field_info.default is not PydanticUndefined
            and field_info.default is not None
        ):
            default = field_info.default
        elif field_info.default_factory is not None:
            default = "factory"

        is_list = is_list_type(field_type)

        # Check if the field is a nested BaseModel
        if is_nested_model(field_type):
            # Recursively process nested model
            nested_type = get_nested_model_type(field_type)
            nested_schema = pydantic_to_json_instance_schema(
                nested_type, citation_level=citation_level, citation=citation
            )
            schema[field_name] = [nested_schema] if is_list else nested_schema
        else:
            # Create the value/citations structure
            type_str = get_type_string(field_type)

            # Build comment parts
            comment_parts = []
            if description:
                comment_parts.append(f"desc: {description}")
            if examples:
                comment_parts.append(f"examples: {examples}")
            if default is not None:
                if default == "factory":
                    comment_parts.append("default: []")
                else:
                    # Handle Enum defaults
                    if isinstance(default, Enum):
                        comment_parts.append(f"default: {default.value}")
                    else:
                        comment_parts.append(f"default: {default}")

            comment = " | ".join(comment_parts) if comment_parts else ""

            if citation:
                leaf: dict | str = {
                    "value": f"<{type_str}>",
                    "comment": comment,
                    "citations": [_citation_item_schema(citation_level)],
                }
            else:
                leaf = f"<{type_str}>"

            schema[field_name] = [leaf] if is_list else leaf

    return schema


def stringify_schema(schema: dict, indent: int = 4) -> str:
    """
    Stringify the schema with comments preserved as inline comments.

    Args:
        schema: The citation schema dictionary
        indent: Number of spaces for indentation (default: 4)

    Returns:
        Formatted string representation of the schema
    """

    def format_dict(d: dict, level: int = 0) -> str:
        if not d:
            return "{}"

        lines = ["{"]
        items = list(d.items())

        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            indent_str = " " * (indent * (level + 1))
            comma = "," if not is_last else ""

            if (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], dict)
            ):
                # List-wrapped item (list[int], list[Model], etc.)
                item = value[0]
                inner_indent = " " * (indent * (level + 2))
                lines.append(f'{indent_str}"{key}": [')
                if "value" in item and "citations" in item:
                    # List of leaf nodes
                    comment = item.get("comment", "")
                    comment_str = f"  # {comment}" if comment else ""
                    lines.append(f"{inner_indent}{{")
                    lines.append(
                        f'{inner_indent}    "value": {item["value"]},{comment_str}'
                    )
                    citation_str = json.dumps(item["citations"]).replace(
                        '"<integer>"', "<integer>"
                    )
                    lines.append(f'{inner_indent}    "citations": {citation_str}')
                    lines.append(f"{inner_indent}}}")
                else:
                    # List of nested objects
                    nested = format_dict(item, level + 2)
                    nested_lines = nested.split("\n")
                    nested_lines[0] = inner_indent + nested_lines[0]
                    lines.extend(nested_lines)
                lines.append(f"{indent_str}]{comma}")
            elif isinstance(value, dict):
                if "value" in value and "citations" in value:
                    # This is a leaf node with value/citations
                    comment = value.get("comment", "")
                    comment_str = f"  # {comment}" if comment else ""

                    lines.append(f'{indent_str}"{key}": {{')
                    lines.append(
                        f'{indent_str}    "value": {value["value"]},{comment_str}'
                    )
                    citation_str = json.dumps(value["citations"]).replace(
                        '"<integer>"', "<integer>"
                    )
                    lines.append(f'{indent_str}    "citations": {citation_str}')
                    lines.append(f"{indent_str}}}{comma}")
                else:
                    # This is a nested object
                    nested = format_dict(value, level + 1)
                    lines.append(f'{indent_str}"{key}": {nested}{comma}')
            elif isinstance(value, list):
                # List of plain type placeholders (citation=False path)
                formatted = ", ".join(
                    item
                    if isinstance(item, str)
                    and item.startswith("<")
                    and item.endswith(">")
                    else json.dumps(item)
                    for item in value
                )
                lines.append(f'{indent_str}"{key}": [{formatted}]{comma}')
            elif (
                isinstance(value, str) and value.startswith("<") and value.endswith(">")
            ):
                # Plain type placeholder (citation=False path)
                lines.append(f'{indent_str}"{key}": {value}{comma}')
            else:
                lines.append(f'{indent_str}"{key}": {json.dumps(value)}{comma}')

        lines.append(" " * (indent * level) + "}")
        return "\n".join(lines)

    return format_dict(schema)


def schema_to_json(schema: dict, indent: int = 2) -> str:
    """
    Convert schema to clean JSON (without internal comment fields).

    Args:
        schema: The citation schema dictionary
        indent: Number of spaces for JSON indentation (default: 2)

    Returns:
        JSON string representation
    """

    def clean_for_json(obj):
        if isinstance(obj, dict):
            if "comment" in obj:
                # Remove comment key for clean JSON
                return {k: clean_for_json(v) for k, v in obj.items() if k != "comment"}
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        return obj

    return json.dumps(clean_for_json(schema), indent=indent)


# Example usage and testing
if __name__ == "__main__":
    # Example models
    class Address(BaseModel):
        street: str = Field(..., description="street name")
        city: str = Field(..., description="city name")
        zipcode: str = Field(..., description="zipcode")

    class User(BaseModel):
        ids: list[int] = Field(..., description="ids")
        name: str = Field(..., description="name", examples=["John Doe"])
        address: Address = Field(..., description="address")

    # Transform and print (with citations)
    print("Example: User Model (citation=True)")
    print("=" * 80)
    schema = pydantic_to_json_instance_schema(User, citation_level="line")
    print(stringify_schema(schema))
    print("\n" + "=" * 80)
    print("As JSON:")
    print("=" * 80)
    print(schema_to_json(schema))

    # Transform and print (without citations)
    print("\n" + "=" * 80)
    print("Example: User Model (citation=False)")
    print("=" * 80)
    schema_no_cite = pydantic_to_json_instance_schema(User, citation=False)
    print(stringify_schema(schema_no_cite))
    print("\n" + "=" * 80)
    print("As JSON (no citations):")
    print("=" * 80)
    print(schema_to_json(schema_no_cite))
