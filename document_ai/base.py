from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .schemas import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> BaseModel:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_for_llm(self, content: BaseModel) -> list[str]:
        pass


class DocumentProcessor:
    def __init__(self, parser: BaseParser, formatter: BaseFormatter):
        self.parser = parser
        self.formatter = formatter

    def process(self, document: Document) -> Document:
        content = self.parser.parse(document)
        document.content = content  # type: ignore
        llm_input = self.formatter.format_for_llm(content)
        document.llm_input = llm_input
        print(document.content)
        return document
