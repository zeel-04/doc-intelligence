"""Integration tests for the Anthropic LLM provider."""

import pytest

from doc_intelligence.llm import AnthropicLLM
from tests_integration.conftest import assert_valid_response


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.anthropic
class TestAnthropicLLMIntegration:
    """Smoke-test AnthropicLLM against the live Anthropic Messages API."""

    def test_generate_text_returns_valid_response(
        self,
        require_anthropic_key: None,
        standard_prompts: tuple[str, str],
    ) -> None:
        llm = AnthropicLLM()
        system_prompt, user_prompt = standard_prompts
        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="claude-sonnet-4-20250514",
        )
        assert_valid_response(response)

    def test_response_is_nonempty_string(
        self,
        require_anthropic_key: None,
        standard_prompts: tuple[str, str],
    ) -> None:
        llm = AnthropicLLM()
        system_prompt, user_prompt = standard_prompts
        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="claude-sonnet-4-20250514",
        )
        assert isinstance(response, str)
        assert len(response.strip()) > 0
