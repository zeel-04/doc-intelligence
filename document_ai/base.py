from abc import ABC, abstractmethod
from typing import Any

from .llm import BaseLLM
from .schemas import Document, Mode, PydanticModel


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> PydanticModel:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_document_for_llm(self, document: Document, mode: Mode) -> list[str]:
        pass


class BaseExtractor(ABC):
    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    @abstractmethod
    def extract(
        self,
        document: Document,
        model: str,
        reasoning: Any,
        response_format: PydanticModel,
        mode: Mode,
        llm_input: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel:
        pass


class DocumentProcessor:
    def __init__(
        self,
        parser: BaseParser,
        formatter: BaseFormatter,
        extractor: BaseExtractor,
        document: Document,
        mode: Mode = Mode(),
    ):
        self.parser = parser
        self.formatter = formatter
        self.extractor = extractor
        self.document = document
        self.mode = mode

    def parse(self) -> Document:
        self.document.content = self.parser.parse(self.document)
        return self.document

    def format_document_for_llm(self) -> list[str]:
        if not self.document.content:
            raise ValueError("Please parse the document first")
        self.document.llm_input = self.formatter.format_document_for_llm(
            self.document,
            self.mode,
        )
        return self.document.llm_input

    def extract(
        self,
        model: str,
        reasoning: Any,
        response_format: PydanticModel,
        llm_input: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        openai_text: dict[str, Any] | None = None,
    ) -> Any:
        if not self.document.content and not self.document.llm_input:
            raise ValueError("Either document content or llm input is missing")
        return self.extractor.extract(
            document=self.document,
            model=model,
            reasoning=reasoning,
            response_format=response_format,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=self.mode,
            llm_input=llm_input,
            openai_text=openai_text,
        )
