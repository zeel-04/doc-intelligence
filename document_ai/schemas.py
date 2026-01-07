from pydantic import BaseModel


class BoundingBox(BaseModel):
    x0: float
    top: float
    x1: float
    bottom: float


class Line(BaseModel):
    text: str
    bounding_box: BoundingBox


class Page(BaseModel):
    lines: list[Line]
    width: int
    height: int


class PDFDocument(BaseModel):
    pages: list[Page]
