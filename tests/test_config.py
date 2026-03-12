"""Tests for config module."""

from doc_intelligence.config import config


# ---------------------------------------------------------------------------
# config dict
# ---------------------------------------------------------------------------
class TestConfig:
    def test_top_level_keys(self):
        assert "digital_pdf" in config

    def test_digital_pdf_has_llm(self):
        assert "llm" in config["digital_pdf"]

    def test_llm_has_model(self):
        assert "model" in config["digital_pdf"]["llm"]

    def test_default_model_value(self):
        assert config["digital_pdf"]["llm"]["model"] == "gpt-5-mini"
