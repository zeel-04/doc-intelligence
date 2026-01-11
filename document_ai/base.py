from abc import ABC, abstractmethod
from typing import Any

from .schemas import Document, Mode, PydanticModel


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> PydanticModel:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_for_llm(self, content: PydanticModel, mode: Mode) -> list[str] | str:
        pass


class DocumentProcessor:
    def __init__(
        self,
        parser: BaseParser,
        formatter: BaseFormatter,
        document: Document,
        mode: Mode,
    ):
        self.parser = parser
        self.formatter = formatter
        self.document = document
        self.mode = mode

    def parse(self) -> Document:
        self.document.content = self.parser.parse(self.document)
        return self.document

    def formatted_input_for_llm(self) -> list[str] | str:
        if not self.document.content:
            raise ValueError("Please parse the document first")
        self.document.llm_input = self.formatter.format_for_llm(
            self.document.content,  # type: ignore
            self.mode,  
        ) 
        return self.document.llm_input
