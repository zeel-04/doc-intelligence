"""Integration tests for the OpenAI LLM provider."""

import pytest

from doc_intelligence.llm import OpenAILLM
from tests_integration.conftest import assert_valid_response


# ---------------------------------------------------------------------------
# OpenAILLM
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.openai
class TestOpenAILLMIntegration:
    """Smoke-test OpenAILLM against the live OpenAI API."""

    def test_generate_text_returns_valid_response(
        self,
        require_openai_key: None,
        standard_prompts: tuple[str, str],
    ) -> None:
        llm = OpenAILLM()
        system_prompt, user_prompt = standard_prompts
        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4o-mini",
        )
        assert_valid_response(response)

    def test_response_is_nonempty_string(
        self,
        require_openai_key: None,
        standard_prompts: tuple[str, str],
    ) -> None:
        llm = OpenAILLM()
        system_prompt, user_prompt = standard_prompts
        response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4o-mini",
        )
        assert isinstance(response, str)
        assert len(response.strip()) > 0
