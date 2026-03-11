"""Tests for pdf.utils module."""

import pytest

from doc_intelligence.pdf.schemas import PDFDocument
from doc_intelligence.pdf.utils import enrich_citations_with_bboxes


# ---------------------------------------------------------------------------
# enrich_citations_with_bboxes
# ---------------------------------------------------------------------------
class TestEnrichCitationsWithBboxes:
    def test_single_citation(self, sample_pdf_document: PDFDocument, sample_bbox):
        response = {"name": {"page": 0, "lines": [0]}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert "bboxes" in result["name"]
        assert "lines" not in result["name"]
        assert result["name"]["page"] == 0
        assert result["name"]["bboxes"] == [sample_bbox.model_dump()]

    def test_multiple_lines_in_citation(
        self, sample_pdf_document: PDFDocument, sample_bbox
    ):
        response = {"name": {"page": 0, "lines": [0, 1, 2]}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert len(result["name"]["bboxes"]) == 3

    def test_nested_citations(self, sample_pdf_document: PDFDocument):
        response = {
            "person": {
                "name": {"page": 0, "lines": [0]},
                "age": {"page": 1, "lines": [1]},
            }
        }
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert "bboxes" in result["person"]["name"]
        assert "bboxes" in result["person"]["age"]

    def test_list_of_citations(self, sample_pdf_document: PDFDocument):
        response = {
            "items": [
                {"page": 0, "lines": [0]},
                {"page": 0, "lines": [1]},
            ]
        }
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert all("bboxes" in item for item in result["items"])
        assert all("lines" not in item for item in result["items"])

    def test_page_out_of_bounds_returns_as_is(self, sample_pdf_document: PDFDocument):
        response = {"name": {"page": 99, "lines": [0]}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert result["name"] == {"page": 99, "lines": [0]}

    def test_negative_page_returns_as_is(self, sample_pdf_document: PDFDocument):
        response = {"name": {"page": -1, "lines": [0]}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert result["name"] == {"page": -1, "lines": [0]}

    def test_line_out_of_bounds_skipped(self, sample_pdf_document: PDFDocument):
        response = {"name": {"page": 0, "lines": [0, 99]}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert len(result["name"]["bboxes"]) == 1

    def test_negative_line_index_skipped(self, sample_pdf_document: PDFDocument):
        response = {"name": {"page": 0, "lines": [-1, 0]}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert len(result["name"]["bboxes"]) == 1

    def test_none_content_raises(self, sample_pdf_document_unparsed: PDFDocument):
        with pytest.raises(ValueError, match="Document content is None"):
            enrich_citations_with_bboxes({}, sample_pdf_document_unparsed)

    def test_non_citation_dict_unchanged(self, sample_pdf_document: PDFDocument):
        response = {"name": "Alice", "count": 42}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert result == {"name": "Alice", "count": 42}

    def test_deeply_nested_mixed(self, sample_pdf_document: PDFDocument):
        response = {
            "data": {
                "items": [
                    {"value": "test", "ref": {"page": 0, "lines": [0]}},
                ],
                "plain": "untouched",
            }
        }
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert result["data"]["plain"] == "untouched"
        assert result["data"]["items"][0]["value"] == "test"
        assert "bboxes" in result["data"]["items"][0]["ref"]

    def test_empty_lines_list(self, sample_pdf_document: PDFDocument):
        response = {"name": {"page": 0, "lines": []}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert result["name"]["bboxes"] == []

    def test_preserves_extra_keys_in_citation(
        self,
        sample_pdf_document: PDFDocument,
    ):
        response = {"name": {"page": 0, "lines": [0], "confidence": 0.95}}
        result = enrich_citations_with_bboxes(response, sample_pdf_document)
        assert result["name"]["confidence"] == 0.95
        assert result["name"]["page"] == 0
        assert "bboxes" in result["name"]
        assert "lines" not in result["name"]
