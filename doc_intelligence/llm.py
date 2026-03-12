from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt

from .base import BaseLLM
from .config import config


class OpenAILLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI()

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
