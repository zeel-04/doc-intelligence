"""Tests for utils module."""

from typing import Any

import pytest
from pydantic import BaseModel

from doc_intelligence.schemas.core import BoundingBox
from doc_intelligence.schemas.pdf import PDF, PDFDocument
from doc_intelligence.utils import (
    add_bboxes_to_citation_model,
    denormalize_bounding_box,
    enrich_citations_with_bboxes,
    find_citation_fields,
    is_citation_type,
    normalize_bounding_box,
    strip_citations,
)


# ---------------------------------------------------------------------------
# normalize_bounding_box
# ---------------------------------------------------------------------------
class TestNormalizeBoundingBox:
    def test_basic_normalization(self):
        bbox = BoundingBox(x0=100.0, top=200.0, x1=300.0, bottom=400.0)
        result = normalize_bounding_box(bbox, page_width=1000, page_height=1000)
        assert result.x0 == pytest.approx(0.1)
        assert result.top == pytest.approx(0.2)
        assert result.x1 == pytest.approx(0.3)
        assert result.bottom == pytest.approx(0.4)

    def test_identity_when_dimensions_are_one(self):
        bbox = BoundingBox(x0=0.5, top=0.6, x1=0.7, bottom=0.8)
        result = normalize_bounding_box(bbox, page_width=1, page_height=1)
        assert result == bbox

    def test_float_dimensions(self):
        bbox = BoundingBox(x0=50.0, top=100.0, x1=250.0, bottom=125.0)
        result = normalize_bounding_box(bbox, page_width=500.0, page_height=1000.0)
        assert result.x0 == pytest.approx(0.1)
        assert result.top == pytest.approx(0.1)
        assert result.x1 == pytest.approx(0.5)
        assert result.bottom == pytest.approx(0.125)

    @pytest.mark.parametrize(
        "bbox, width, height, expected_x0",
        [
            (BoundingBox(x0=100, top=0, x1=0, bottom=0), 1000, 1, 0.1),
            (BoundingBox(x0=0, top=0, x1=0, bottom=0), 500, 800, 0.0),
            (BoundingBox(x0=250, top=400, x1=250, bottom=400), 500, 800, 0.5),
        ],
    )
    def test_various_inputs(self, bbox, width, height, expected_x0):
        result = normalize_bounding_box(bbox, width, height)
        assert result.x0 == pytest.approx(expected_x0)


# ---------------------------------------------------------------------------
# denormalize_bounding_box
# ---------------------------------------------------------------------------
class TestDenormalizeBoundingBox:
    def test_basic_denormalization(self):
        bbox = BoundingBox(x0=0.1, top=0.2, x1=0.3, bottom=0.4)
        result = denormalize_bounding_box(bbox, page_width=1000, page_height=1000)
        assert result.x0 == pytest.approx(100.0)
        assert result.top == pytest.approx(200.0)
        assert result.x1 == pytest.approx(300.0)
        assert result.bottom == pytest.approx(400.0)

    def test_identity_when_dimensions_are_one(self):
        bbox = BoundingBox(x0=0.5, top=0.6, x1=0.7, bottom=0.8)
        result = denormalize_bounding_box(bbox, page_width=1, page_height=1)
        assert result == bbox

    def test_roundtrip(self):
        original = BoundingBox(x0=150.0, top=300.0, x1=450.0, bottom=600.0)
        width, height = 612, 792
        normalized = normalize_bounding_box(original, width, height)
        restored = denormalize_bounding_box(normalized, width, height)
        assert restored.x0 == pytest.approx(original.x0)
        assert restored.top == pytest.approx(original.top)
        assert restored.x1 == pytest.approx(original.x1)
        assert restored.bottom == pytest.approx(original.bottom)


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


# ---------------------------------------------------------------------------
# strip_citations
# ---------------------------------------------------------------------------
class TestStripCitations:
    def test_simple_unwrap(self, citation_response_simple: dict[str, Any]):
        result = strip_citations(citation_response_simple)
        assert result == {"name": "Alice"}

    def test_nested_unwrap(self, citation_response_nested: dict[str, Any]):
        result = strip_citations(citation_response_nested)
        assert result == {"person": {"name": "Bob", "age": 30}}

    def test_list_of_wrapped(self, citation_response_with_list: dict[str, Any]):
        result = strip_citations(citation_response_with_list)
        assert result == {"ids": [101, 205]}

    def test_non_citation_dict_unchanged(self):
        response = {"name": "Alice", "age": 30}
        result = strip_citations(response)
        assert result == {"name": "Alice", "age": 30}

    def test_deeply_nested(self):
        response = {
            "level1": {
                "level2": {
                    "field": {"value": "deep", "citations": [{"page": 0}]},
                },
            },
        }
        result = strip_citations(response)
        assert result == {"level1": {"level2": {"field": "deep"}}}

    def test_dict_with_extra_keys_not_stripped(self):
        response = {"name": {"value": "Alice", "citations": [], "extra": True}}
        result = strip_citations(response)
        assert result == response

    def test_empty_dict(self):
        assert strip_citations({}) == {}

    def test_scalar_values_in_list(self):
        response = {"tags": ["a", "b", "c"]}
        result = strip_citations(response)
        assert result == {"tags": ["a", "b", "c"]}

    def test_none_value_preserved(self):
        response = {"field": {"value": None, "citations": []}}
        result = strip_citations(response)
        assert result == {"field": None}

    def test_nested_value_preserved(self):
        response = {
            "field": {
                "value": {"nested_key": 42},
                "citations": [{"page": 0}],
            },
        }
        result = strip_citations(response)
        assert result == {"field": {"nested_key": 42}}


# ---------------------------------------------------------------------------
# is_citation_type
# ---------------------------------------------------------------------------
class TestIsCitationType:
    def test_direct_match(self):
        assert is_citation_type(str, str) is True

    def test_direct_no_match(self):
        assert is_citation_type(str, int) is False

    def test_list_type_match(self):
        assert is_citation_type(list[str], list[str]) is True

    def test_list_type_no_match(self):
        assert is_citation_type(list[str], list[int]) is False

    def test_non_list_vs_list(self):
        assert is_citation_type(str, list[str]) is False

    def test_list_vs_non_list(self):
        assert is_citation_type(list[str], str) is False

    def test_pydantic_model_match(self):
        class Citation(BaseModel):
            page: int

        assert is_citation_type(Citation, Citation) is True

    def test_list_of_model_match(self):
        class Citation(BaseModel):
            page: int

        assert is_citation_type(list[Citation], list[Citation]) is True

    def test_name_based_match(self):
        """Two distinct classes with the same __name__ match via name comparison."""

        class Citation(BaseModel):
            page: int

        Citation1 = type("Citation", (BaseModel,), {"__annotations__": {"page": int}})
        Citation2 = type("Citation", (BaseModel,), {"__annotations__": {"page": int}})
        assert is_citation_type(list[Citation1], list[Citation2]) is True


# ---------------------------------------------------------------------------
# find_citation_fields
# ---------------------------------------------------------------------------
class TestFindCitationFields:
    def test_flat_model(self):
        class CitationType(BaseModel):
            page: int

        class MyModel(BaseModel):
            name: str
            ref: CitationType

        result = find_citation_fields(MyModel, CitationType)
        assert result == ["ref"]

    def test_nested_model(self):
        class CitationType(BaseModel):
            page: int

        class Inner(BaseModel):
            ref: CitationType

        class Outer(BaseModel):
            name: str
            inner: Inner

        result = find_citation_fields(Outer, CitationType)
        assert result == ["inner.ref"]

    def test_list_of_models(self):
        class CitationType(BaseModel):
            page: int

        class Item(BaseModel):
            ref: CitationType

        class MyModel(BaseModel):
            items: list[Item]

        result = find_citation_fields(MyModel, CitationType)
        assert result == ["items.ref"]

    def test_no_citations(self):
        class CitationType(BaseModel):
            page: int

        class MyModel(BaseModel):
            name: str
            age: int

        result = find_citation_fields(MyModel, CitationType)
        assert result == []

    def test_multiple_citation_fields(self):
        class CitationType(BaseModel):
            page: int

        class MyModel(BaseModel):
            name_ref: CitationType
            age_ref: CitationType
            plain: str

        result = find_citation_fields(MyModel, CitationType)
        assert "name_ref" in result
        assert "age_ref" in result
        assert len(result) == 2

    def test_deep_nesting(self):
        class CitationType(BaseModel):
            page: int

        class Level2(BaseModel):
            ref: CitationType

        class Level1(BaseModel):
            nested: Level2

        class Root(BaseModel):
            level1: Level1

        result = find_citation_fields(Root, CitationType)
        assert result == ["level1.nested.ref"]

    def test_with_prefix(self):
        class CitationType(BaseModel):
            page: int

        class MyModel(BaseModel):
            ref: CitationType

        result = find_citation_fields(MyModel, CitationType, prefix="parent")
        assert result == ["parent.ref"]


# ---------------------------------------------------------------------------
# add_bboxes_to_citation_model
# ---------------------------------------------------------------------------
class TestAddBboxesToCitationModel:
    def test_replaces_citation_type(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class MyModel(BaseModel):
            name: str
            ref: OldCitation

        result_model = add_bboxes_to_citation_model(MyModel, OldCitation, NewCitation)
        assert result_model.model_fields["ref"].annotation is NewCitation
        assert result_model.model_fields["name"].annotation is str

    def test_nested_replacement(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class Inner(BaseModel):
            ref: OldCitation

        class Outer(BaseModel):
            inner: Inner

        result_model = add_bboxes_to_citation_model(Outer, OldCitation, NewCitation)
        inner_type = result_model.model_fields["inner"].annotation
        assert inner_type.model_fields["ref"].annotation is NewCitation

    def test_list_of_models(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class Item(BaseModel):
            ref: OldCitation

        class MyModel(BaseModel):
            items: list[Item]

        result_model = add_bboxes_to_citation_model(MyModel, OldCitation, NewCitation)
        from typing import get_args, get_origin

        items_type = result_model.model_fields["items"].annotation
        assert get_origin(items_type) is list
        inner_model = get_args(items_type)[0]
        assert inner_model.model_fields["ref"].annotation is NewCitation

    def test_non_citation_fields_preserved(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class MyModel(BaseModel):
            name: str
            age: int
            ref: OldCitation

        result_model = add_bboxes_to_citation_model(MyModel, OldCitation, NewCitation)
        assert result_model.model_fields["name"].annotation is str
        assert result_model.model_fields["age"].annotation is int

    def test_model_name_suffix(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class Invoice(BaseModel):
            ref: OldCitation

        result_model = add_bboxes_to_citation_model(Invoice, OldCitation, NewCitation)
        assert result_model.__name__ == "InvoiceWithBBox"

    def test_new_model_is_valid_pydantic(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class MyModel(BaseModel):
            name: str
            ref: OldCitation

        result_model = add_bboxes_to_citation_model(MyModel, OldCitation, NewCitation)
        instance = result_model(
            name="test", ref=NewCitation(page=1, bboxes=[{"x0": 0}])
        )
        assert instance.name == "test"
        assert instance.ref.page == 1

    def test_no_citation_fields_unchanged(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class MyModel(BaseModel):
            name: str
            age: int

        result_model = add_bboxes_to_citation_model(MyModel, OldCitation, NewCitation)
        assert result_model.model_fields["name"].annotation is str
        assert result_model.model_fields["age"].annotation is int

    def test_list_of_primitives_preserved(self):
        class OldCitation(BaseModel):
            page: int

        class NewCitation(BaseModel):
            page: int
            bboxes: list[dict]

        class MyModel(BaseModel):
            tags: list[str]
            ref: OldCitation

        result_model = add_bboxes_to_citation_model(MyModel, OldCitation, NewCitation)
        assert result_model.model_fields["tags"].annotation == list[str]
        assert result_model.model_fields["ref"].annotation is NewCitation
