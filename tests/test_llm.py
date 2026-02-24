"""Tests for llm module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from doc_intelligence.llm import OpenAILLM


@pytest.fixture
def mock_openai_client():
    """Patch OpenAI and return the mock client instance."""
    with patch("doc_intelligence.llm.OpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


@pytest.fixture
def llm(mock_openai_client) -> OpenAILLM:
    """An OpenAILLM instance with a mocked client."""
    return OpenAILLM()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestOpenAILLMInit:
    def test_creates_client(self, llm: OpenAILLM, mock_openai_client):
        assert llm.client is mock_openai_client


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------
class TestGenerateText:
    def test_returns_output_text(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(
            output_text="Hello from LLM"
        )
        result = llm.generate_text(
            system_prompt="You are helpful.", user_prompt="Say hello."
        )
        assert result == "Hello from LLM"

    def test_calls_create_with_correct_args(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate_text(system_prompt="sys", user_prompt="usr")
        mock_openai_client.responses.create.assert_called_once_with(
            model="gpt-5-mini",
            instructions="sys",
            input="usr",
        )

    def test_default_model_from_config(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate_text(system_prompt="s", user_prompt="u")
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5-mini"

    def test_custom_model_override(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate_text(system_prompt="s", user_prompt="u", model="gpt-4o")
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    def test_passes_extra_kwargs(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate_text(system_prompt="s", user_prompt="u", temperature=0.5)
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5

    def test_retry_on_failure(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            MagicMock(output_text="third time lucky"),
        ]
        result = llm.generate_text(system_prompt="s", user_prompt="u")
        assert result == "third time lucky"
        assert mock_openai_client.responses.create.call_count == 3

    def test_retry_exhausted_raises(self, llm: OpenAILLM, mock_openai_client):
        from tenacity import RetryError

        mock_openai_client.responses.create.side_effect = Exception("always fail")
        with pytest.raises(RetryError):
            llm.generate_text(system_prompt="s", user_prompt="u")
        assert mock_openai_client.responses.create.call_count == 3


# ---------------------------------------------------------------------------
# generate_structured_output
# ---------------------------------------------------------------------------
class TestGenerateStructuredOutput:
    def test_returns_output_parsed(self, llm: OpenAILLM, mock_openai_client):
        class MyModel(BaseModel):
            name: str

        expected = MyModel(name="Alice")
        mock_openai_client.responses.parse.return_value = MagicMock(
            output_parsed=expected
        )

        result = llm.generate_structured_output(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            reasoning=None,
            output_format=MyModel,
        )
        assert result == expected

    def test_calls_parse_with_correct_args(self, llm: OpenAILLM, mock_openai_client):
        class MyModel(BaseModel):
            name: str

        mock_openai_client.responses.parse.return_value = MagicMock(output_parsed=None)
        messages = [{"role": "user", "content": "hi"}]
        llm.generate_structured_output(
            model="gpt-5-mini",
            messages=messages,
            reasoning={"effort": "high"},
            output_format=MyModel,
        )
        mock_openai_client.responses.parse.assert_called_once_with(
            model="gpt-5-mini",
            input=messages,
            reasoning={"effort": "high"},
            text=None,
            text_format=MyModel,
        )

    def test_openai_text_forwarded(self, llm: OpenAILLM, mock_openai_client):
        class MyModel(BaseModel):
            name: str

        mock_openai_client.responses.parse.return_value = MagicMock(output_parsed=None)
        text_config = {"format": {"type": "json_schema"}}
        llm.generate_structured_output(
            model="gpt-5-mini",
            messages=[],
            reasoning=None,
            output_format=MyModel,
            openai_text=text_config,
        )
        call_kwargs = mock_openai_client.responses.parse.call_args
        assert call_kwargs.kwargs["text"] == text_config

    def test_openai_text_none_sends_none(self, llm: OpenAILLM, mock_openai_client):
        class MyModel(BaseModel):
            name: str

        mock_openai_client.responses.parse.return_value = MagicMock(output_parsed=None)
        llm.generate_structured_output(
            model="gpt-5-mini",
            messages=[],
            reasoning=None,
            output_format=MyModel,
            openai_text=None,
        )
        call_kwargs = mock_openai_client.responses.parse.call_args
        assert call_kwargs.kwargs["text"] is None
