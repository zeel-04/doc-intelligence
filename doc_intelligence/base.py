from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.output_parsers import JsonOutputParser

from doc_intelligence.schemas.core import Document, ExtractionResult, PydanticModel

TDocument = TypeVar("TDocument", bound=Document)


class BaseParser(ABC, Generic[TDocument]):
    @abstractmethod
    def parse(self, document: TDocument) -> TDocument:
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
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        pass

    def generate_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[PydanticModel],
        **kwargs,
    ) -> PydanticModel | None:
        """Generate a structured Pydantic model response.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            response_format: The Pydantic model class to parse the response into.
            **kwargs: Additional provider-specific arguments.

        Raises:
            NotImplementedError: If the provider does not support structured output natively.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output. "
            "Use generate_text with a JSON schema prompt instead."
        )


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
    ) -> ExtractionResult:
        pass
