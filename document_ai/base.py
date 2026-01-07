from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseDocument(ABC):
    def __init__(self, document_type: str, uri: str):
        self.document_type = document_type
        self.uri = uri
        self.content: type[BaseModel] | None = None
        self.llm_input: str | None = None

    def get_content(self) -> Any:
        """Returns parsed content"""
        return self._content

    def set_content(self, content: Any) -> Any:
        """Sets the parsed content"""
        self._content = content


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: BaseDocument) -> Any:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_for_llm(self, document: BaseDocument) -> Any:
        pass


class DocumentProcessor:
    def __init__(self, parser: BaseParser, formatter: BaseFormatter):
        self.parser = parser
        self.formatter = formatter

    def process(self, document: BaseDocument) -> Any:
        content = self.parser.parse(document)
        return self.formatter.format_for_llm(content)
