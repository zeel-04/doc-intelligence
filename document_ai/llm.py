from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt

from .base import BaseLLM
from .config import config
from .schemas.core import PydanticModel


class OpenAILLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI()

    @retry(stop=stop_after_attempt(3))
    def generate_structured_output(
        self,
        model: str,
        messages: list[dict[str, str]],
        reasoning: Any,
        output_format: type[PydanticModel],
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel | None:
        response = self.client.responses.parse(
            model=model,
            input=messages,  # type:ignore[arg-type]
            reasoning=reasoning,
            text=openai_text if openai_text else None,  # type:ignore[arg-type]
            text_format=output_format,
        ).output_parsed
        return response

    @retry(stop=stop_after_attempt(3))
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        model = kwargs.pop("model", config["digital_pdf"]["llm"]["model"])
        logger.debug(f"OpenAILLM: generate_text: Generating text with model: {model}")
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            **kwargs,
        )
        return response.output_text
