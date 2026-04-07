"""Tests for llm module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from doc_intelligence.config import settings
from doc_intelligence.llm import (
    AnthropicLLM,
    GeminiLLM,
    OllamaLLM,
    OpenAILLM,
    create_llm,
)


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
# OpenAILLM
# ---------------------------------------------------------------------------
class TestOpenAILLMInit:
    def test_creates_client(self, llm: OpenAILLM, mock_openai_client):
        assert llm.client is mock_openai_client

    def test_default_model(self, llm: OpenAILLM):
        assert llm.model == settings.openai_default_model

    def test_custom_model_stored(self, mock_openai_client):
        llm = OpenAILLM(model="gpt-4o")
        assert llm.model == "gpt-4o"

    def test_settings_default_model_respected(self, mock_openai_client):
        with patch("doc_intelligence.llm.settings") as mock_settings:
            mock_settings.openai_default_model = "gpt-4o-turbo"
            llm = OpenAILLM()
            assert llm.model == "gpt-4o-turbo"


class TestOpenAILLMGenerate:
    def test_returns_output_text(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(
            output_text="Hello from LLM"
        )
        result = llm.generate(
            system_prompt="You are helpful.", user_prompt="Say hello."
        )
        assert result == "Hello from LLM"

    def test_calls_create_with_correct_args(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate(system_prompt="sys", user_prompt="usr")
        mock_openai_client.responses.create.assert_called_once_with(
            model=settings.openai_default_model,
            instructions="sys",
            input="usr",
        )

    def test_default_model_from_instance(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate(system_prompt="s", user_prompt="u")
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == settings.openai_default_model

    def test_custom_model_override(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate(system_prompt="s", user_prompt="u", model="gpt-4o")
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    def test_per_call_override_does_not_mutate_instance(
        self, llm: OpenAILLM, mock_openai_client
    ):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate(system_prompt="s", user_prompt="u", model="gpt-4o")
        assert llm.model == settings.openai_default_model

    def test_passes_extra_kwargs(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate(system_prompt="s", user_prompt="u", temperature=0.5)
        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5

    def test_retry_on_failure(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            MagicMock(output_text="third time lucky"),
        ]
        result = llm.generate(system_prompt="s", user_prompt="u")
        assert result == "third time lucky"
        assert mock_openai_client.responses.create.call_count == 3

    def test_retry_exhausted_raises(self, llm: OpenAILLM, mock_openai_client):
        from tenacity import RetryError

        mock_openai_client.responses.create.side_effect = Exception("always fail")
        with pytest.raises(RetryError):
            llm.generate(system_prompt="s", user_prompt="u")
        assert mock_openai_client.responses.create.call_count == 3


# ---------------------------------------------------------------------------
# OpenAILLM.generate (vision)
# ---------------------------------------------------------------------------
class TestOpenAILLMGenerateVision:
    def test_returns_output_text(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(
            output_text="image response"
        )
        result = llm.generate(
            system_prompt="sys",
            user_prompt="describe",
            images=["data:image/png;base64,abc123"],
        )
        assert result == "image response"

    def test_builds_multipart_input(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate(
            system_prompt="sys",
            user_prompt="describe",
            images=["data:image/png;base64,img1", "data:image/png;base64,img2"],
        )
        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == "sys"
        input_content = call_kwargs["input"][0]["content"]
        assert input_content[0] == {"type": "input_text", "text": "describe"}
        assert input_content[1]["type"] == "input_image"
        assert input_content[1]["image_url"] == "data:image/png;base64,img1"
        assert input_content[1]["detail"] == "high"
        assert input_content[2]["type"] == "input_image"
        assert input_content[2]["image_url"] == "data:image/png;base64,img2"

    def test_custom_model_override(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate("sys", "usr", ["data:image/png;base64,x"], model="gpt-4o")
        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_passes_extra_kwargs(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.return_value = MagicMock(output_text="ok")
        llm.generate("sys", "usr", ["data:image/png;base64,x"], temperature=0.5)
        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_retry_on_failure(self, llm: OpenAILLM, mock_openai_client):
        mock_openai_client.responses.create.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            MagicMock(output_text="third time"),
        ]
        result = llm.generate("sys", "usr", ["data:image/png;base64,x"])
        assert result == "third time"
        assert mock_openai_client.responses.create.call_count == 3


# ---------------------------------------------------------------------------
# OllamaLLM
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ollama_module():
    """Inject a mock ollama module into sys.modules."""
    mock_mod = MagicMock()
    mock_client_instance = MagicMock()
    mock_mod.Client.return_value = mock_client_instance
    with patch.dict(sys.modules, {"ollama": mock_mod}):
        yield mock_mod, mock_client_instance


class TestOllamaLLMInit:
    def test_creates_ollama_client(self, mock_ollama_module):
        mock_mod, mock_client = mock_ollama_module
        llm = OllamaLLM(host="http://myhost:11434")
        mock_mod.Client.assert_called_once_with(host="http://myhost:11434")
        assert llm.client is mock_client

    def test_default_host(self, mock_ollama_module):
        mock_mod, _ = mock_ollama_module
        OllamaLLM()
        mock_mod.Client.assert_called_once_with(host="http://localhost:11434")

    def test_default_model(self, mock_ollama_module):
        llm = OllamaLLM()
        assert llm.model == settings.ollama_default_model

    def test_custom_model_stored(self, mock_ollama_module):
        llm = OllamaLLM(model="qwen3")
        assert llm.model == "qwen3"

    def test_settings_default_model_respected(self, mock_ollama_module):
        with patch("doc_intelligence.llm.settings") as mock_settings:
            mock_settings.ollama_default_model = "mistral"
            llm = OllamaLLM()
            assert llm.model == "mistral"

    def test_is_subclass_of_base_llm(self):
        from doc_intelligence.base import BaseLLM

        assert issubclass(OllamaLLM, BaseLLM)


class TestOllamaLLMGenerate:
    def _make_llm(self, mock_ollama_module) -> tuple[OllamaLLM, MagicMock]:
        _, mock_client = mock_ollama_module
        llm = OllamaLLM()
        return llm, mock_client

    def test_returns_message_content(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ollama reply"))
        result = llm.generate(system_prompt="sys", user_prompt="usr")
        assert result == "ollama reply"

    def test_calls_chat_with_correct_args(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate(system_prompt="sys", user_prompt="usr")
        client.chat.assert_called_once_with(
            model=settings.ollama_default_model,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ],
            stream=False,
        )

    def test_custom_model_override(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr", model="qwen3")
        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs["model"] == "qwen3"

    def test_per_call_override_does_not_mutate_instance(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr", model="qwen3")
        assert llm.model == settings.ollama_default_model

    def test_think_kwarg_forwarded_as_top_level(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr", think=False)
        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs["think"] is False

    def test_stream_false_always_sent(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr")
        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs["stream"] is False

    def test_stream_kwarg_stripped_from_caller(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr", stream=True)
        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs["stream"] is False

    def test_raises_on_none_content(self, mock_ollama_module):
        from tenacity import RetryError

        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content=None))
        with pytest.raises(RetryError) as exc_info:
            llm.generate("sys", "usr")
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "empty response content" in str(exc_info.value.__cause__)

    def test_retry_on_failure(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            MagicMock(message=MagicMock(content="success")),
        ]
        result = llm.generate("sys", "usr")
        assert result == "success"
        assert client.chat.call_count == 3

    def test_retry_exhausted_raises(self, mock_ollama_module):
        from tenacity import RetryError

        llm, client = self._make_llm(mock_ollama_module)
        client.chat.side_effect = Exception("always fail")
        with pytest.raises(RetryError):
            llm.generate("sys", "usr")
        assert client.chat.call_count == 3


# ---------------------------------------------------------------------------
# OllamaLLM.generate (vision)
# ---------------------------------------------------------------------------
class TestOllamaLLMGenerateVision:
    def _make_llm(self, mock_ollama_module) -> tuple[OllamaLLM, MagicMock]:
        _, mock_client = mock_ollama_module
        llm = OllamaLLM()
        return llm, mock_client

    def test_returns_message_content(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="vision reply"))
        result = llm.generate("sys", "usr", ["data:image/png;base64,abc"])
        assert result == "vision reply"

    def test_strips_data_url_prefix(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr", ["data:image/png;base64,abc123"])
        call_kwargs = client.chat.call_args.kwargs
        user_msg = call_kwargs["messages"][1]
        assert user_msg["images"] == ["abc123"]

    def test_raw_base64_passthrough(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate("sys", "usr", ["rawbase64data"])
        call_kwargs = client.chat.call_args.kwargs
        user_msg = call_kwargs["messages"][1]
        assert user_msg["images"] == ["rawbase64data"]

    def test_multiple_images(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
        llm.generate(
            "sys",
            "usr",
            ["data:image/png;base64,img1", "data:image/png;base64,img2"],
        )
        call_kwargs = client.chat.call_args.kwargs
        user_msg = call_kwargs["messages"][1]
        assert user_msg["images"] == ["img1", "img2"]

    def test_raises_on_none_content(self, mock_ollama_module):
        from tenacity import RetryError

        llm, client = self._make_llm(mock_ollama_module)
        client.chat.return_value = MagicMock(message=MagicMock(content=None))
        with pytest.raises(RetryError) as exc_info:
            llm.generate("sys", "usr", ["data:image/png;base64,x"])
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_retry_on_failure(self, mock_ollama_module):
        llm, client = self._make_llm(mock_ollama_module)
        client.chat.side_effect = [
            Exception("fail"),
            Exception("fail"),
            MagicMock(message=MagicMock(content="success")),
        ]
        result = llm.generate("sys", "usr", ["data:image/png;base64,x"])
        assert result == "success"
        assert client.chat.call_count == 3


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_anthropic_module():
    """Inject a mock anthropic module into sys.modules."""
    mock_mod = MagicMock()
    mock_client_instance = MagicMock()
    mock_mod.Anthropic.return_value = mock_client_instance
    with patch.dict(sys.modules, {"anthropic": mock_mod}):
        yield mock_mod, mock_client_instance


class TestAnthropicLLMInit:
    def test_creates_anthropic_client(self, mock_anthropic_module):
        mock_mod, mock_client = mock_anthropic_module
        llm = AnthropicLLM(api_key="sk-test")
        mock_mod.Anthropic.assert_called_once_with(api_key="sk-test")
        assert llm.client is mock_client

    def test_api_key_none_by_default(self, mock_anthropic_module):
        mock_mod, _ = mock_anthropic_module
        AnthropicLLM()
        mock_mod.Anthropic.assert_called_once_with(api_key=None)

    def test_default_model(self, mock_anthropic_module):
        llm = AnthropicLLM()
        assert llm.model == settings.anthropic_default_model

    def test_custom_model_stored(self, mock_anthropic_module):
        llm = AnthropicLLM(model="claude-opus-4-20250514")
        assert llm.model == "claude-opus-4-20250514"

    def test_settings_default_model_respected(self, mock_anthropic_module):
        with patch("doc_intelligence.llm.settings") as mock_settings:
            mock_settings.anthropic_default_model = "claude-haiku-4-20250514"
            llm = AnthropicLLM()
            assert llm.model == "claude-haiku-4-20250514"


class TestAnthropicLLMGenerate:
    def _make_llm(self, mock_anthropic_module) -> tuple[AnthropicLLM, MagicMock]:
        _, mock_client = mock_anthropic_module
        llm = AnthropicLLM()
        return llm, mock_client

    def test_returns_first_content_text(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="anthropic reply")]
        )
        result = llm.generate("sys", "usr")
        assert result == "anthropic reply"

    def test_calls_messages_create_correctly(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        llm.generate("sys", "usr")
        mock_client.messages.create.assert_called_once_with(
            model=settings.anthropic_default_model,
            system="sys",
            messages=[{"role": "user", "content": "usr"}],
            max_tokens=4096,
        )

    def test_custom_model_override(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        llm.generate("sys", "usr", model="claude-opus-4-6")
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

    def test_per_call_override_does_not_mutate_instance(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        llm.generate("sys", "usr", model="claude-opus-4-6")
        assert llm.model == settings.anthropic_default_model

    def test_custom_max_tokens(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        llm.generate("sys", "usr", max_tokens=1024)
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 1024

    def test_retry_on_failure(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            MagicMock(content=[MagicMock(text="success")]),
        ]
        result = llm.generate("sys", "usr")
        assert result == "success"
        assert mock_client.messages.create.call_count == 3

    def test_retry_exhausted_raises(self, mock_anthropic_module):
        from tenacity import RetryError

        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.side_effect = Exception("always fail")
        with pytest.raises(RetryError):
            llm.generate("sys", "usr")
        assert mock_client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# AnthropicLLM.generate (vision)
# ---------------------------------------------------------------------------
class TestAnthropicLLMGenerateVision:
    def _make_llm(self, mock_anthropic_module) -> tuple[AnthropicLLM, MagicMock]:
        _, mock_client = mock_anthropic_module
        llm = AnthropicLLM()
        return llm, mock_client

    def test_returns_first_content_text(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="vision reply")]
        )
        result = llm.generate("sys", "usr", ["data:image/png;base64,abc123"])
        assert result == "vision reply"

    def test_builds_multipart_content(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        llm.generate("sys", "describe", ["data:image/png;base64,abc123"])
        call_kwargs = mock_client.messages.create.call_args.kwargs
        msg_content = call_kwargs["messages"][0]["content"]
        assert msg_content[0] == {"type": "text", "text": "describe"}
        assert msg_content[1]["type"] == "image"
        assert msg_content[1]["source"]["type"] == "base64"
        assert msg_content[1]["source"]["media_type"] == "image/png"
        assert msg_content[1]["source"]["data"] == "abc123"

    def test_custom_model_override(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        llm.generate("sys", "usr", ["data:image/png;base64,x"], model="claude-opus-4-6")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_retry_on_failure(self, mock_anthropic_module):
        llm, mock_client = self._make_llm(mock_anthropic_module)
        mock_client.messages.create.side_effect = [
            Exception("fail"),
            Exception("fail"),
            MagicMock(content=[MagicMock(text="success")]),
        ]
        result = llm.generate("sys", "usr", ["data:image/png;base64,x"])
        assert result == "success"
        assert mock_client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# GeminiLLM
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_gemini(monkeypatch):
    """Patch genai.Client and genai_types.GenerateContentConfig in llm module."""
    mock_client_instance = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_instance)
    mock_config_cls = MagicMock()

    monkeypatch.setattr("doc_intelligence.llm.genai.Client", mock_client_cls)
    monkeypatch.setattr(
        "doc_intelligence.llm.genai_types.GenerateContentConfig", mock_config_cls
    )
    return mock_client_cls, mock_client_instance, mock_config_cls


class TestGeminiLLMInit:
    def test_creates_genai_client(self, mock_gemini):
        mock_client_cls, mock_client_instance, _ = mock_gemini
        llm = GeminiLLM(api_key="gemini-key")
        mock_client_cls.assert_called_once_with(api_key="gemini-key")
        assert llm.client is mock_client_instance

    def test_api_key_none_by_default(self, mock_gemini):
        mock_client_cls, _, _ = mock_gemini
        GeminiLLM()
        mock_client_cls.assert_called_once_with(api_key=None)

    def test_default_model(self, mock_gemini):
        llm = GeminiLLM()
        assert llm.model == settings.gemini_default_model

    def test_custom_model_stored(self, mock_gemini):
        llm = GeminiLLM(model="gemini-2.0-pro")
        assert llm.model == "gemini-2.0-pro"

    def test_settings_default_model_respected(self, mock_gemini):
        with patch("doc_intelligence.llm.settings") as mock_settings:
            mock_settings.gemini_default_model = "gemini-2.0-pro"
            llm = GeminiLLM()
            assert llm.model == "gemini-2.0-pro"


class TestGeminiLLMGenerate:
    def _make_llm(self, mock_gemini) -> tuple[GeminiLLM, MagicMock, MagicMock]:
        _, mock_client, mock_config_cls = mock_gemini
        llm = GeminiLLM()
        return llm, mock_client, mock_config_cls

    def test_returns_response_text(self, mock_gemini):
        llm, mock_client, _ = self._make_llm(mock_gemini)
        mock_client.models.generate_content.return_value = MagicMock(
            text="gemini reply"
        )
        result = llm.generate("sys", "usr")
        assert result == "gemini reply"

    def test_calls_generate_content_with_correct_args(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_client.models.generate_content.return_value = MagicMock(text="ok")
        fake_config = MagicMock()
        mock_config_cls.return_value = fake_config
        llm.generate("sys", "usr")
        mock_client.models.generate_content.assert_called_once_with(
            model=settings.gemini_default_model,
            contents="usr",
            config=fake_config,
        )
        mock_config_cls.assert_called_once_with(system_instruction="sys")

    def test_custom_model_override(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_client.models.generate_content.return_value = MagicMock(text="ok")
        mock_config_cls.return_value = MagicMock()
        llm.generate("sys", "usr", model="gemini-2.0-pro")
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.0-pro"

    def test_per_call_override_does_not_mutate_instance(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_client.models.generate_content.return_value = MagicMock(text="ok")
        mock_config_cls.return_value = MagicMock()
        llm.generate("sys", "usr", model="gemini-2.0-pro")
        assert llm.model == settings.gemini_default_model

    def test_retry_on_failure(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            MagicMock(text="success"),
        ]
        result = llm.generate("sys", "usr")
        assert result == "success"
        assert mock_client.models.generate_content.call_count == 3

    def test_retry_exhausted_raises(self, mock_gemini):
        from tenacity import RetryError

        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("always fail")
        with pytest.raises(RetryError):
            llm.generate("sys", "usr")
        assert mock_client.models.generate_content.call_count == 3


# ---------------------------------------------------------------------------
# GeminiLLM.generate (vision)
# ---------------------------------------------------------------------------
class TestGeminiLLMGenerateVision:
    """Tests for GeminiLLM vision support.

    Gemini decodes base64 data in-method, so test data URLs must use valid
    base64 (``dGVzdA==`` encodes ``b"test"``).
    """

    _VALID_DATA_URL = "data:image/png;base64,dGVzdA=="

    def _make_llm(self, mock_gemini) -> tuple[GeminiLLM, MagicMock, MagicMock]:
        _, mock_client, mock_config_cls = mock_gemini
        llm = GeminiLLM()
        return llm, mock_client, mock_config_cls

    def test_returns_response_text(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="vision reply"
        )
        result = llm.generate("sys", "usr", [self._VALID_DATA_URL])
        assert result == "vision reply"

    def test_calls_generate_content_with_image(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="ok")
        llm.generate("sys", "describe", [self._VALID_DATA_URL])
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        assert len(contents) == 1
        parts = contents[0].parts
        assert len(parts) == 2
        assert parts[0].text == "describe"
        assert parts[1].inline_data.mime_type == "image/png"
        assert parts[1].inline_data.data == b"test"

    def test_custom_model_override(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="ok")
        llm.generate("sys", "usr", [self._VALID_DATA_URL], model="gemini-2.0-pro")
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.0-pro"

    def test_raises_on_none_text(self, mock_gemini):
        from tenacity import RetryError

        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text=None)
        with pytest.raises(RetryError) as exc_info:
            llm.generate("sys", "usr", [self._VALID_DATA_URL])
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_retry_on_failure(self, mock_gemini):
        llm, mock_client, mock_config_cls = self._make_llm(mock_gemini)
        mock_config_cls.return_value = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("fail"),
            Exception("fail"),
            MagicMock(text="success"),
        ]
        result = llm.generate("sys", "usr", [self._VALID_DATA_URL])
        assert result == "success"
        assert mock_client.models.generate_content.call_count == 3


# ---------------------------------------------------------------------------
# create_llm factory
# ---------------------------------------------------------------------------
class TestCreateLLM:
    def test_creates_openai(self, mock_openai_client):
        llm = create_llm("openai")
        assert isinstance(llm, OpenAILLM)

    def test_creates_ollama(self, mock_ollama_module):
        llm = create_llm("ollama")
        assert isinstance(llm, OllamaLLM)

    def test_creates_anthropic(self, mock_anthropic_module):
        llm = create_llm("anthropic")
        assert isinstance(llm, AnthropicLLM)

    def test_creates_gemini(self, mock_gemini):
        llm = create_llm("gemini")
        assert isinstance(llm, GeminiLLM)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm("unknown_provider")

    def test_model_forwarded_to_constructor(self, mock_openai_client):
        llm = create_llm("openai", model="gpt-4o")
        assert llm.model == "gpt-4o"

    def test_kwargs_forwarded_to_constructor(self, mock_gemini):
        llm = create_llm("gemini", api_key="my-key")
        assert isinstance(llm, GeminiLLM)

    def test_case_insensitive_provider(self, mock_openai_client):
        llm = create_llm("OpenAI")
        assert isinstance(llm, OpenAILLM)

    def test_provider_default_model_when_none(self, mock_openai_client):
        llm = create_llm("openai")
        assert llm.model == settings.openai_default_model
