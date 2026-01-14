from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt

from .base import BaseLLM
from .schemas import PydanticModel


class OpenAILLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI()

    @retry(stop=stop_after_attempt(3))
    def generate_structured_output(
        self,
        model: str,
        messages: list[dict[str, str]],
        reasoning: Any,
        output_format: PydanticModel,
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel:
        response = self.client.responses.parse(
            model=model,
            input=messages,  # type:ignore[arg-type]
            reasoning=reasoning,
            text=openai_text if openai_text else None,  # type:ignore[arg-type]
            text_format=output_format,  # type:ignore[arg-type]
        ).output_parsed
        return response  # ty:ignore[invalid-return-type]
