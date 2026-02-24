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
            BaseParser()

    def test_base_formatter(self):
        with pytest.raises(TypeError):
            BaseFormatter()

    def test_base_llm(self):
        with pytest.raises(TypeError):
            BaseLLM()

    def test_base_extractor(self, fake_llm: FakeLLM):
        with pytest.raises(TypeError):
            BaseExtractor(fake_llm)


# ---------------------------------------------------------------------------
# Incomplete subclass enforcement
# ---------------------------------------------------------------------------
class TestIncompleteSubclassRaises:
    def test_parser_missing_parse(self):
        class BadParser(BaseParser):
            pass

        with pytest.raises(TypeError):
            BadParser()

    def test_formatter_missing_format(self):
        class BadFormatter(BaseFormatter):
            pass

        with pytest.raises(TypeError):
            BadFormatter()

    def test_llm_missing_generate_structured_output(self):
        class BadLLM(BaseLLM):
            def generate_text(self, system_prompt, user_prompt, **kwargs):
                return ""

        with pytest.raises(TypeError):
            BadLLM()

    def test_llm_missing_generate_text(self):
        class BadLLM(BaseLLM):
            def generate_structured_output(
                self, model, messages, reasoning, output_format, openai_text=None
            ):
                return None

        with pytest.raises(TypeError):
            BadLLM()

    def test_extractor_missing_extract(self, fake_llm: FakeLLM):
        class BadExtractor(BaseExtractor):
            pass

        with pytest.raises(TypeError):
            BadExtractor(fake_llm)


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
