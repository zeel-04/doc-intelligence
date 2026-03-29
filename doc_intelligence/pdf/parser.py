from abc import abstractmethod
from io import BytesIO
from urllib.parse import urlparse

import pdfplumber
import requests

from doc_intelligence.base import BaseParser
from doc_intelligence.pdf.schemas import PDF, Line, Page, PDFDocument, TextBlock
from doc_intelligence.schemas.core import BoundingBox
from doc_intelligence.utils import normalize_bounding_box


class PDFParser(BaseParser[PDFDocument]):
    @abstractmethod
    def parse(self, document: PDFDocument) -> PDFDocument:
        pass


class DigitalPDFParser(PDFParser):
    def parse(self, document: PDFDocument) -> PDFDocument:
        pages = []

        # Check if URI is a URL or local path
        parsed = urlparse(document.uri)
        if parsed.scheme in ("http", "https"):
            # Download the PDF from URL
            response = requests.get(document.uri)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
        else:
            # Use local file path
            pdf_file = document.uri

        with pdfplumber.open(pdf_file) as pdf:
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
                blocks = [TextBlock(lines=lines)] if lines else []
                pages.append(Page(blocks=blocks, width=page.width, height=page.height))
        return PDFDocument(uri=document.uri, content=PDF(pages=pages))
