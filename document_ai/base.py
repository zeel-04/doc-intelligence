from abc import ABC, abstractmethod
from typing import Any

from .schemas import Document, Mode, PydanticModel


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> PydanticModel:
        pass


class BaseFormatter(ABC):
    @abstractmethod
    def format_document_for_llm(self, document: Document, mode: Mode) -> str:
        pass


class BaseLLM(ABC):
    @abstractmethod
    def generate_structured_output(
        self,
        model: str,
        messages: list[dict[str, str]],
        reasoning: Any,
        output_format: PydanticModel,
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel:
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