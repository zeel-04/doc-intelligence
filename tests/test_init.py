"""Tests for the top-level public API surface."""

import doc_intelligence as di


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
class TestPublicAPI:
    _expected = [
        "PDFProcessor",
        "DocumentProcessor",
        "create_llm",
        "BaseLLM",
        "OpenAILLM",
        "OllamaLLM",
        "AnthropicLLM",
        "GeminiLLM",
        "PDFDocument",
        "PDFExtractionMode",
        "PDFExtractionRequest",
        "PDFParser",
        "ParseStrategy",
        "ExtractionResult",
        "BoundingBox",
        "BaseCitation",
    ]

    def test_all_names_importable(self) -> None:
        for name in self._expected:
            assert hasattr(di, name), f"Missing from public API: {name}"

    def test_dunder_all_matches_importable_names(self) -> None:
        for name in di.__all__:
            assert hasattr(di, name), f"Listed in __all__ but not importable: {name}"

    def test_no_unexpected_names_in_all(self) -> None:
        missing = [n for n in self._expected if n not in di.__all__]
        assert missing == [], f"Expected names missing from __all__: {missing}"
