"""Integration tests for the Ollama LLM provider.

Automatically discovers every locally installed Ollama model and runs a
simple smoke-test against each one.
"""

import pytest
from loguru import logger

from doc_intelligence.llm import OllamaLLM

from .conftest import assert_valid_response, get_ollama_model_ids

_THINKING_PREFIXES = ("qwen3", "deepseek-r1")


def _is_thinking_model(model_name: str) -> bool:
    """Return True if the model defaults to thinking mode."""
    lower = model_name.lower()
    return any(lower.startswith(prefix) for prefix in _THINKING_PREFIXES)


# ---------------------------------------------------------------------------
# OllamaLLM — all available models
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.ollama
class TestOllamaLLMIntegration:
    """Smoke-test OllamaLLM against every locally available model."""

    @pytest.mark.parametrize("model_name", get_ollama_model_ids())
    def test_generate_text_returns_valid_response(
        self,
        require_ollama_server: list[str],
        standard_prompts: tuple[str, str],
        model_name: str,
    ) -> None:
        if model_name == "__no_models_available__":
            pytest.skip("No Ollama models available")

        llm = OllamaLLM()
        system_prompt, user_prompt = standard_prompts

        kwargs: dict = {"model": model_name}
        if _is_thinking_model(model_name):
            kwargs["think"] = False

        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )
        assert_valid_response(response)

    @pytest.mark.parametrize("model_name", get_ollama_model_ids())
    def test_response_is_nonempty_string(
        self,
        require_ollama_server: list[str],
        standard_prompts: tuple[str, str],
        model_name: str,
    ) -> None:
        if model_name == "__no_models_available__":
            pytest.skip("No Ollama models available")

        llm = OllamaLLM()
        system_prompt, user_prompt = standard_prompts

        kwargs: dict = {"model": model_name}
        if _is_thinking_model(model_name):
            kwargs["think"] = False

        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )
        logger.info(f"LLM response: {response!r}")
        assert isinstance(response, str)
        assert len(response.strip()) > 0
