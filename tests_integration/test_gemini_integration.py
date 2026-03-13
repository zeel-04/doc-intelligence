"""Integration tests for the Gemini LLM provider."""

import pytest

from doc_intelligence.llm import GeminiLLM
from tests_integration.conftest import assert_valid_response


# ---------------------------------------------------------------------------
# GeminiLLM
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.gemini
class TestGeminiLLMIntegration:
    """Smoke-test GeminiLLM against the live Google Gemini API."""

    def test_generate_text_returns_valid_response(
        self,
        require_gemini_key: None,
        standard_prompts: tuple[str, str],
    ) -> None:
        llm = GeminiLLM()
        system_prompt, user_prompt = standard_prompts
        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gemini-2.0-flash",
        )
        assert_valid_response(response)

    def test_response_is_nonempty_string(
        self,
        require_gemini_key: None,
        standard_prompts: tuple[str, str],
    ) -> None:
        llm = GeminiLLM()
        system_prompt, user_prompt = standard_prompts
        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gemini-2.0-flash",
        )
        assert isinstance(response, str)
        assert len(response.strip()) > 0
