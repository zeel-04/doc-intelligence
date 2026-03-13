"""doc_intelligence — AI-powered document processing pipelines."""

from doc_intelligence.base import BaseLLM
from doc_intelligence.extract import extract
from doc_intelligence.llm import (
    AnthropicLLM,
    GeminiLLM,
    OllamaLLM,
    OpenAILLM,
    create_llm,
)
from doc_intelligence.pdf.processor import DocumentProcessor, PDFProcessor
from doc_intelligence.pdf.schemas import PDFDocument, PDFExtractionConfig
from doc_intelligence.pdf.types import PDFExtractionMode
from doc_intelligence.schemas.core import BaseCitation, BoundingBox, ExtractionResult

__all__ = [
    # Processors
    "PDFProcessor",
    "DocumentProcessor",
    # One-liner
    "extract",
    # LLMs
    "create_llm",
    "BaseLLM",
    "OpenAILLM",
    "OllamaLLM",
    "AnthropicLLM",
    "GeminiLLM",
    # PDF types
    "PDFDocument",
    "PDFExtractionMode",
    "PDFExtractionConfig",
    # Result & schema primitives
    "ExtractionResult",
    "BoundingBox",
    "BaseCitation",
]
