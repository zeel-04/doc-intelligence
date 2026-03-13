from typing import Any

from google import genai
from google.genai import types as genai_types
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt

from .base import BaseLLM
from .config import settings


class OpenAILLM(BaseLLM):
    def __init__(self, model: str | None = None):
        super().__init__(model=model or settings.openai_default_model)
        self.client = OpenAI()

    @retry(stop=stop_after_attempt(3))
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        model = kwargs.pop("model", self.model)
        logger.debug(f"OpenAILLM: generate_text: Generating text with model: {model}")
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            **kwargs,
        )
        return response.output_text


class OllamaLLM(BaseLLM):
    """LLM backed by a local Ollama server via the native Ollama Python SDK.

    Uses ``ollama.Client.chat()`` which correctly respects first-class
    parameters such as ``think=False`` — unlike the OpenAI-compatible
    ``/v1/chat/completions`` endpoint which silently ignores that field.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str | None = None):
        super().__init__(model=model or settings.ollama_default_model)
        import ollama  # optional dependency  # type: ignore[missing-import]

        self.client = ollama.Client(host=host)

    @retry(stop=stop_after_attempt(3))
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text using the native Ollama chat API.

        Args:
            system_prompt: The system instruction for the LLM.
            user_prompt: The user message to send.
            **kwargs: Additional arguments forwarded to ``client.chat()``
                (e.g. ``think=False``, ``options={"temperature": 0.7}``).

        Returns:
            The text content of the assistant's reply.

        Raises:
            ValueError: If the model returns an empty response.
        """
        model = kwargs.pop("model", self.model)
        kwargs.pop("stream", None)
        logger.debug(f"OllamaLLM: generate_text: Generating text with model: {model}")
        response = self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            **kwargs,
        )
        content = response.message.content
        if content is None:
            raise ValueError("OllamaLLM: received empty response content")
        return content


class AnthropicLLM(BaseLLM):
    """LLM backed by the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(model=model or settings.anthropic_default_model)
        import anthropic  # optional dependency  # type: ignore[missing-import]

        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3))
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", 4096)
        logger.debug(
            f"AnthropicLLM: generate_text: Generating text with model: {model}"
        )
        response = self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text


class GeminiLLM(BaseLLM):
    """LLM backed by the Google Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(model=model or settings.gemini_default_model)
        self.client = genai.Client(api_key=api_key)

    @retry(stop=stop_after_attempt(3))
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        model = kwargs.pop("model", self.model)
        logger.debug(f"GeminiLLM: generate_text: Generating text with model: {model}")
        response = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                **kwargs,
            ),
        )
        if response.text is None:
            raise ValueError("GeminiLLM: received empty response text")
        return response.text


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------
_LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "openai": OpenAILLM,
    "ollama": OllamaLLM,
    "anthropic": AnthropicLLM,
    "gemini": GeminiLLM,
}


def create_llm(provider: str, model: str | None = None, **kwargs) -> BaseLLM:
    """Create an LLM instance by provider name.

    Args:
        provider: The LLM provider name (e.g., "openai", "anthropic",
            "gemini", "ollama"). Case-insensitive.
        model: Optional model name. If *None*, the provider's default is used.
        **kwargs: Provider-specific arguments (e.g., ``api_key``, ``host``).

    Returns:
        A configured :class:`BaseLLM` instance.

    Raises:
        ValueError: If *provider* is not recognised.
    """
    cls = _LLM_REGISTRY.get(provider.lower())
    if cls is None:
        available = sorted(_LLM_REGISTRY)
        raise ValueError(f"Unknown LLM provider: {provider!r}. Available: {available}")
    if model is not None:
        kwargs["model"] = model
    return cls(**kwargs)
