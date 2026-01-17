from abc import ABC, abstractmethod
from typing import Any

from .schemas import Document, PydanticModel


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> PydanticModel:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_document_for_llm(
        self,
        document: Document,
        include_line_numbers: bool,
        page_numbers: list[int] | None = None,
    ) -> str:
        pass


class BaseLLM(ABC):
    @abstractmethod
    def generate_structured_output(
        self,
        model: str,
        messages: list[dict[str, str]],
        reasoning: Any,
        output_format: type[PydanticModel],
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel | None:
        pass


class BaseExtractor(ABC):
    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm
        self.system_prompt = ""
        self.user_prompt = ""

    @abstractmethod
    def extract(
        self,
        document: Document,
        model: str,
        reasoning: Any,
        response_format: type[PydanticModel],
        include_line_numbers: bool,
        llm_input: str,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel:
        pass
