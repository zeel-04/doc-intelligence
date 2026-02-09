from abc import ABC, abstractmethod
from typing import Any

from langchain_core.output_parsers import JsonOutputParser

from .schemas.core import Document, PydanticModel


class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> PydanticModel:
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

    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
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
        llm_config: dict[str, Any],
        extraction_config: dict[str, Any],
        formatter: BaseFormatter,
        response_format: type[PydanticModel],
    ) -> tuple[PydanticModel, dict[str, Any] | None]:
        pass
