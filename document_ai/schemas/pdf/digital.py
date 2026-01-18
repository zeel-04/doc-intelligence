from typing import Any

from document_ai.schemas.core import BaseCitation


# -------------------------------------
# Citation schemas for Digital PDF with line numbers(Norm)
# -------------------------------------
class PageLineCitation(BaseCitation):
    page: int
    lines: list[int]


class PageLineBboxCitation(BaseCitation):
    page: int
    lines: list[int]
    bboxes: list[dict[str, Any]]


# -------------------------------------
# Citation schemas for Digital PDF without line numbers
# -------------------------------------
class PageCitation(BaseCitation):
    page: int
