"""Hard-limit validators called before any expensive pipeline work."""

import os
from typing import Any

import pdfplumber
from pydantic import BaseModel


def check_pdf_size(uri: str, max_mb: float) -> None:
    """Raise ValueError if the PDF file exceeds max_mb megabytes.

    Args:
        uri: Local file path to the PDF.
        max_mb: Maximum allowed file size in megabytes.

    Raises:
        ValueError: If the file size exceeds max_mb.
    """
    if not os.path.isfile(uri):
        return  # non-local URIs (e.g. URLs) are not size-checked here
    size_mb = os.path.getsize(uri) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"PDF exceeds max size ({size_mb:.1f} MB > {max_mb} MB): {uri}"
        )


def check_page_count(uri: str, max_pages: int) -> None:
    """Raise ValueError if the PDF at uri has more pages than max_pages.

    Opens the PDF to read its page count without full parsing.

    Args:
        uri: Local file path to the PDF.
        max_pages: Maximum allowed page count.

    Raises:
        ValueError: If page count exceeds max_pages.
    """
    if not os.path.isfile(uri):
        return  # non-local URIs are not page-checked here
    with pdfplumber.open(uri) as pdf:
        page_count = len(pdf.pages)
    if page_count > max_pages:
        raise ValueError(f"PDF has {page_count} pages, limit is {max_pages}")


def check_schema_depth(
    model: type[BaseModel], max_depth: int, _current: int = 0
) -> None:
    """Raise ValueError if the Pydantic model nesting depth exceeds max_depth.

    Recursively walks model_fields to detect nested BaseModel subclasses.
    The root model is at depth 0; each level of nesting increments the counter.

    Args:
        model: The Pydantic model class to inspect.
        max_depth: Maximum allowed nesting depth (inclusive).
        _current: Internal recursion counter — do not set manually.

    Raises:
        ValueError: If schema depth exceeds max_depth.
    """
    if _current > max_depth:
        raise ValueError(f"Schema depth {_current} exceeds limit {max_depth}")
    for field in model.model_fields.values():
        _walk_annotation(field.annotation, max_depth, _current)


def _walk_annotation(annotation: Any, max_depth: int, current: int) -> None:
    """Recursively walk a type annotation, descending into nested BaseModels."""
    args = getattr(annotation, "__args__", None) or ()
    if args:
        # Generic or union type: list[T], T | None, Union[T, U], etc.
        # Handles both typing.Union (__origin__ set) and X | Y (__args__ only).
        for arg in args:
            if arg is not type(None):
                _walk_annotation(arg, max_depth, current)
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        check_schema_depth(annotation, max_depth, current + 1)
