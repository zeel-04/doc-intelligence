"""LLM Playground — interactive experimentation with any provider.

Edit the configuration constants below, then run:

    uv run pytest tests_integration/test_llm_playground.py -v -s
"""

import os

import pytest
from loguru import logger

from doc_intelligence.base import BaseLLM

# ---------------------------------------------------------------------------
# ✏️  Edit these to experiment
# ---------------------------------------------------------------------------

# ── Pick your provider ────────────────────────────────────────────────
PROVIDER = "ollama"  # "ollama" | "openai" | "anthropic" | "gemini"

# ── Model ─────────────────────────────────────────────────────────────
MODEL = "qwen3.5:9b"  # any model available for the chosen provider

# ── Prompts ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "What is 2+2"

# ── Provider-specific kwargs (forwarded to generate_text) ─────────────
#    Ollama:    {"think": False, "options": {"temperature": 0.7}}
#    OpenAI:    {} (most config is on the model side)
#    Anthropic: {"max_tokens": 1024}
#    Gemini:    {"temperature": 0.7}
EXTRA_KWARGS: dict = {"think": False}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_llm(provider: str, available_ollama_models: list[str]) -> BaseLLM:
    """Instantiate the correct LLM class for *provider*.

    Skips the test gracefully when required credentials or services are
    unavailable.

    Args:
        provider: One of ``"ollama"``, ``"openai"``, ``"anthropic"``,
            ``"gemini"``.
        available_ollama_models: Model IDs returned by the
            ``require_ollama_server`` fixture (used only when
            *provider* is ``"ollama"``).
    """
    if provider == "ollama":
        if not available_ollama_models:
            pytest.skip("Ollama server not reachable or has no models")
        from doc_intelligence.llm import OllamaLLM

        return OllamaLLM()

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        from doc_intelligence.llm import OpenAILLM

        return OpenAILLM()

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        from doc_intelligence.llm import AnthropicLLM

        return AnthropicLLM()

    if provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        from doc_intelligence.llm import GeminiLLM

        return GeminiLLM()

    pytest.fail(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Playground test
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestLLMPlayground:
    """Ad-hoc playground — tweak the constants at the top of this file and run."""

    def test_playground(self) -> None:
        """Send a single prompt and log the full response."""
        from tests_integration.conftest import _get_ollama_models

        ollama_models = _get_ollama_models() if PROVIDER == "ollama" else []
        llm = _make_llm(PROVIDER, ollama_models)

        logger.info(f"Provider : {PROVIDER}")
        logger.info(f"Model    : {MODEL}")
        logger.info(f"System   : {SYSTEM_PROMPT!r}")
        logger.info(f"User     : {USER_PROMPT!r}")
        if EXTRA_KWARGS:
            logger.info(f"Kwargs   : {EXTRA_KWARGS}")

        response = llm.generate_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            model=MODEL,
            **EXTRA_KWARGS,
        )

        logger.info(f"Response :\n{response}")

        assert isinstance(response, str)
        assert len(response.strip()) > 0, "Response is empty"
