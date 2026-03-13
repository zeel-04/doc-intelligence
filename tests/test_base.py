"""Tests for base module."""

import pytest
from langchain_core.output_parsers import JsonOutputParser

from doc_intelligence.base import BaseExtractor, BaseFormatter, BaseLLM, BaseParser
from tests.conftest import FakeExtractor, FakeLLM


# ---------------------------------------------------------------------------
# ABC instantiation enforcement
# ---------------------------------------------------------------------------
class TestBaseClassesNotInstantiable:
    def test_base_parser(self):
        with pytest.raises(TypeError):
            BaseParser()  # type: ignore[abstract]

    def test_base_formatter(self):
        with pytest.raises(TypeError):
            BaseFormatter()  # type: ignore[abstract]

    def test_base_llm(self):
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_base_extractor(self, fake_llm: FakeLLM):
        with pytest.raises(TypeError):
            BaseExtractor(fake_llm)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Incomplete subclass enforcement
# ---------------------------------------------------------------------------
class TestIncompleteSubclassRaises:
    def test_parser_missing_parse(self):
        class BadParser(BaseParser):
            pass

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_formatter_missing_format(self):
        class BadFormatter(BaseFormatter):
            pass

        with pytest.raises(TypeError):
            BadFormatter()  # type: ignore[abstract]

    def test_llm_missing_generate_text(self):
        class BadLLM(BaseLLM):
            pass

        with pytest.raises(TypeError):
            BadLLM()  # type: ignore[abstract]

    def test_extractor_missing_extract(self, fake_llm: FakeLLM):
        class BadExtractor(BaseExtractor):
            pass

        with pytest.raises(TypeError):
            BadExtractor(fake_llm)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# BaseLLM.generate_structured_output default
# ---------------------------------------------------------------------------
class TestBaseLLMGenerateStructuredOutput:
    def test_raises_not_implemented(self, fake_llm: FakeLLM):
        from pydantic import BaseModel

        class MyModel(BaseModel):
            value: str

        with pytest.raises(
            NotImplementedError, match="does not support structured output"
        ):
            fake_llm.generate_structured_output(
                system_prompt="sys",
                user_prompt="usr",
                response_format=MyModel,
            )

    def test_error_message_includes_class_name(self, fake_llm: FakeLLM):
        from pydantic import BaseModel

        class MyModel(BaseModel):
            value: str

        with pytest.raises(NotImplementedError, match="FakeLLM"):
            fake_llm.generate_structured_output("s", "u", MyModel)


# ---------------------------------------------------------------------------
# BaseExtractor __init__ side effects
# ---------------------------------------------------------------------------
class TestBaseExtractorInit:
    def test_stores_llm(self, fake_llm: FakeLLM):
        extractor = FakeExtractor(llm=fake_llm)
        assert extractor.llm is fake_llm

    def test_creates_json_parser(self, fake_llm: FakeLLM):
        extractor = FakeExtractor(llm=fake_llm)
        assert isinstance(extractor.json_parser, JsonOutputParser)
