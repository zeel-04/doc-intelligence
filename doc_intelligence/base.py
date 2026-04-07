from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.output_parsers import JsonOutputParser

from doc_intelligence.schemas.core import (
    Document,
    ExtractionRequest,
    ExtractionResult,
)

TDocument = TypeVar("TDocument", bound=Document)


class BaseParser(ABC, Generic[TDocument]):
    @abstractmethod
    def parse(self, uri: str) -> TDocument:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_document_for_llm(
        self,
        document: Document,
        **kwargs,
    ) -> str:
        pass


class BaseLLM(ABC):
    def __init__(self, model: str = ""):
        self.model = model

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt, optionally including images.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            images: Optional base64-encoded data URLs (``data:image/png;base64,...``).
            **kwargs: Additional provider-specific arguments.

        Returns:
            The text content of the model's reply.
        """
        pass


class BaseExtractor(ABC):
    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm
        self.json_parser = JsonOutputParser()

    @abstractmethod
    def extract(
        self,
        document: Document,
        request: ExtractionRequest,
        formatter: BaseFormatter,
    ) -> ExtractionResult:
        pass
