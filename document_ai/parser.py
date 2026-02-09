from abc import abstractmethod

import pdfplumber

from .base import BaseParser
from .schemas.core import Document
from .schemas.pdf import PDF, BoundingBox, Line, Page, PDFDocument
from .utils import normalize_bounding_box


class PDFParser(BaseParser):
    @abstractmethod
    def parse(self, document: Document) -> PDFDocument:
        pass


class DigitalPDFParser(PDFParser):
    def parse(self, document: Document) -> PDFDocument:
        pages = []
        with pdfplumber.open(document.uri) as pdf:
            for page in pdf.pages:
                lines = []
                for line in page.extract_text_lines(return_chars=False):
                    bbox = normalize_bounding_box(
                        BoundingBox(
                            x0=line["x0"],
                            top=line["top"],
                            x1=line["x1"],
                            bottom=line["bottom"],
                        ),
                        page.width,
                        page.height,
                    )
                    lines.append(Line(text=line["text"], bounding_box=bbox))
                pages.append(Page(lines=lines, width=page.width, height=page.height))
        return PDFDocument(uri=document.uri, content=PDF(pages=pages))
