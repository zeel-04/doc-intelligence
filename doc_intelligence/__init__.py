"""doc_intelligence — AI-powered document processing pipelines."""

from doc_intelligence.base import BaseLLM
from doc_intelligence.llm import (
    AnthropicLLM,
    GeminiLLM,
    OllamaLLM,
    OpenAILLM,
    create_llm,
)
from doc_intelligence.ocr.base import BaseLayoutDetector, BaseOCREngine, LayoutRegion
from doc_intelligence.pdf.parser import PDFParser
from doc_intelligence.pdf.processor import DocumentProcessor, PDFProcessor
from doc_intelligence.pdf.schemas import PDFDocument, PDFExtractionRequest
from doc_intelligence.pdf.types import (
    ParseStrategy,
    PDFExtractionMode,
    ScannedPipelineType,
)
from doc_intelligence.schemas.core import BaseCitation, BoundingBox, ExtractionResult

__all__ = [
    # Processors
    "PDFProcessor",
    "DocumentProcessor",
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
    "PDFExtractionRequest",
    "PDFParser",
    "ParseStrategy",
    "ScannedPipelineType",
    # Result & schema primitives
    "ExtractionResult",
    "BoundingBox",
    "BaseCitation",
    # OCR — custom implementation hooks
    "BaseLayoutDetector",
    "BaseOCREngine",
    "LayoutRegion",
]
