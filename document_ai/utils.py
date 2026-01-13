from typing import Any, Type, get_args, get_origin

from pydantic import BaseModel, Field, create_model

from document_ai.schemas import BoundingBox, PydanticModel

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


def add_appropriate_citation_type(
    original_model: PydanticModel, CitationType: Type[Any]
) -> PydanticModel:
    """
    Creates a new Pydantic model where existing citation fields are updated
    with proper Citation types. Handles nested models and lists recursively.

    Users should define citation fields with names ending in '_citation'
    or exactly 'citation' with type Any in their original models.

    Args:
        original_model: The original Pydantic model class

    Returns:
        A new Pydantic model class with citation fields properly typed
    """
    new_fields = {}

    for field_name, field_info in original_model.model_fields.items():
        field_type = field_info.annotation
        origin = get_origin(field_type)

        # Preserve the original field's metadata
        original_default = (
            field_info.default
            if field_info.default is not field_info.default_factory
            else ...
        )
        original_description = field_info.description

        # Check if this is a citation field that needs type modification
        is_citation_field = field_name.endswith("_citation") or field_name == "citation"

        if is_citation_field:
            # Use the CitationType and create a new Field with proper defaults
            new_type = CitationType
            default_value = Field(
                default_factory=list, description=CITATION_DESCRIPTION
            )

            new_fields[field_name] = (new_type, default_value)

        # Handle lists
        elif origin is list:
            inner_type = get_args(field_type)[0]

            # If it's a list of BaseModel, recursively process
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                new_inner_type = add_appropriate_citation_type(inner_type, CitationType)
                new_fields[field_name] = (
                    list[new_inner_type],  # type: ignore
                    Field(default_factory=list, description=original_description),
                )
            else:
                # For primitive types in lists, keep as is
                if field_info.default_factory:
                    new_fields[field_name] = (
                        field_type,
                        Field(
                            default_factory=field_info.default_factory,
                            description=original_description,
                        ),
                    )
                else:
                    new_fields[field_name] = (
                        field_type,
                        Field(
                            default=original_default, description=original_description
                        ),
                    )

        # Handle nested models
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            new_inner_type = add_appropriate_citation_type(field_type, CitationType)
            new_fields[field_name] = (
                new_inner_type | None,
                Field(default=None, description=original_description),
            )

        # Handle Union types and primitive types
        else:
            if field_info.default_factory:
                new_fields[field_name] = (
                    field_type,
                    Field(
                        default_factory=field_info.default_factory,
                        description=original_description,
                    ),
                )
            else:
                new_fields[field_name] = (
                    field_type,
                    Field(default=original_default, description=original_description),
                )

    # Create new model with the same name as original
    new_model = create_model(original_model.__name__, **new_fields)  # type: ignore

    # Preserve the original model's docstring
    if original_model.__doc__:
        new_model.__doc__ = original_model.__doc__

    return new_model


# def transform_model(
#     model: Any,
#     *,
#     add: dict[str, tuple[type, object]] | None = None,  # {"f": (type, default or ...)}
#     drop: set[str] | None = None,
#     retype: dict[str, type] | None = None,
#     name: str | None = None,
# ):
#     add = add or {}
#     drop = drop or set()
#     retype = retype or {}

#     new_fields: dict[str, tuple[type, object]] = {}

#     for fname, finfo in model.model_fields.items():  # pydantic v2
#         if fname in drop:
#             continue
#         ann = retype.get(fname, finfo.annotation)
#         default = ... if finfo.is_required() else finfo.default
#         new_fields[fname] = (ann, default)

#     # add new fields
#     for fname, (ann, default) in add.items():
#         new_fields[fname] = (ann, default)

#     return create_model(name or f"{model.__name__}Transformed", **new_fields)  # type: ignore


# # Example: drop zipcode, add country
# Address2 = transform_model(
#     Address, drop={"zipcode"}, add={"country": (str, "US")}, name="Address2"
# )

# # Rebuild employee to use new Address
# Employee2 = transform_model(Employee, retype={"address": Address2}, name="Employee2")
