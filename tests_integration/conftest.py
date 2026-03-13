"""Shared fixtures and helpers for LLM provider integration tests."""

import json
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from loguru import logger


# ---------------------------------------------------------------------------
# Loguru → pytest stdout bridge (shows DEBUG logs with -s or --capture=no)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def propagate_loguru():
    """Re-route loguru to stdout at DEBUG level for every integration test."""
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True)
    yield
    logger.remove()


# ---------------------------------------------------------------------------
# Standard test prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
USER_PROMPT = "What is 2 + 2? Reply with just the number."


@pytest.fixture
def standard_prompts() -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a trivial arithmetic question."""
    return SYSTEM_PROMPT, USER_PROMPT


def assert_valid_response(response: str) -> None:
    """Assert that the LLM response is a non-empty string containing '4'.

    Args:
        response: The raw text returned by the LLM.
    """
    logger.info(f"LLM response: {response!r}")
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    assert len(response.strip()) > 0, "Response is empty"
    assert "4" in response, f"Expected '4' in response, got: {response!r}"


# ---------------------------------------------------------------------------
# Environment-gated skip helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def require_openai_key() -> None:
    """Skip if OPENAI_API_KEY is not set."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def require_anthropic_key() -> None:
    """Skip if ANTHROPIC_API_KEY is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture
def require_gemini_key() -> None:
    """Skip if GOOGLE_API_KEY is not set."""
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


# ---------------------------------------------------------------------------
# Ollama dynamic model discovery
# ---------------------------------------------------------------------------
def _get_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Query the Ollama ``/api/tags`` endpoint for locally available models.

    Args:
        base_url: The Ollama server URL.

    Returns:
        A list of model name strings, or an empty list if the server is
        unreachable.
    """
    try:
        with urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except (URLError, OSError, json.JSONDecodeError):
        return []


def get_ollama_model_ids() -> list[str]:
    """Return model IDs for ``@pytest.mark.parametrize``.

    Called at collection time.  If the Ollama server is unreachable the
    function returns a single sentinel value that causes the test to be
    skipped at runtime.
    """
    models = _get_ollama_models()
    return models if models else ["__no_models_available__"]


@pytest.fixture
def require_ollama_server() -> list[str]:
    """Skip if the Ollama server is not reachable or has no models."""
    models = _get_ollama_models()
    if not models:
        pytest.skip("Ollama server not reachable or has no models")
    return models
