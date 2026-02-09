"""Tests for pydantic_to_json_instance_schema module."""

from datetime import date, datetime
from enum import Enum
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from document_ai.pydantic_to_json_instance_schema import (
    get_nested_model_type,
    get_type_string,
    is_list_type,
    is_nested_model,
    pydantic_to_json_instance_schema,
    schema_to_json,
    stringify_schema,
)


# ---------------------------------------------------------------------------
# Shared test models
# ---------------------------------------------------------------------------
class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Coord(BaseModel):
    lat: float = Field(..., description="latitude")
    lon: float = Field(..., description="longitude")


class Address(BaseModel):
    street: str = Field(..., description="street name")
    city: str = Field(..., description="city name")


class Person(BaseModel):
    name: str = Field(..., description="full name")
    address: Address = Field(..., description="home address")


class LineItem(BaseModel):
    product: str = Field(..., description="product name")
    qty: int = Field(..., description="quantity")


class Invoice(BaseModel):
    invoice_no: str = Field(..., description="invoice number")
    items: list[LineItem] = Field(..., description="line items")


class Location(BaseModel):
    label: str = Field(..., description="location label")
    coord: Coord = Field(..., description="coordinates")


class Trip(BaseModel):
    trip_id: str = Field(..., description="trip identifier")
    origin: Location = Field(..., description="start location")
    destination: Location = Field(..., description="end location")


# ---------------------------------------------------------------------------
# get_type_string
# ---------------------------------------------------------------------------
class TestGetTypeString:
    @pytest.mark.parametrize(
        "field_type, expected",
        [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
            (datetime, "string"),
            (date, "string"),
        ],
    )
    def test_basic_types(self, field_type, expected):
        assert get_type_string(field_type) == expected

    def test_optional_unwraps(self):
        assert get_type_string(Optional[int]) == "integer"

    def test_list_unwraps(self):
        assert get_type_string(list[int]) == "integer"

    def test_dict_returns_object(self):
        assert get_type_string(dict[str, int]) == "object"

    def test_enum_returns_string(self):
        assert get_type_string(Color) == "string"


# ---------------------------------------------------------------------------
# is_nested_model / get_nested_model_type / is_list_type
# ---------------------------------------------------------------------------
class TestTypeHelpers:
    def test_is_nested_model_plain(self):
        assert is_nested_model(Address) is True
        assert is_nested_model(str) is False

    def test_is_nested_model_optional(self):
        assert is_nested_model(Optional[Address]) is True
        assert is_nested_model(Optional[str]) is False

    def test_is_nested_model_list(self):
        assert is_nested_model(list[LineItem]) is True
        assert is_nested_model(list[int]) is False

    def test_get_nested_model_type(self):
        assert get_nested_model_type(Address) is Address
        assert get_nested_model_type(Optional[Address]) is Address
        assert get_nested_model_type(list[LineItem]) is LineItem

    def test_is_list_type(self):
        assert is_list_type(list[int]) is True
        assert is_list_type(Optional[list[int]]) is True
        assert is_list_type(str) is False
        assert is_list_type(int) is False


# ---------------------------------------------------------------------------
# pydantic_to_json_instance_schema — citation=True (default)
# ---------------------------------------------------------------------------
class TestSchemaWithCitation:
    """Tests for the default citation=True path."""

    def test_primitive_fields(self):
        class M(BaseModel):
            a_str: str = Field(..., description="a string")
            a_int: int = Field(...)
            a_float: float = Field(...)
            a_bool: bool = Field(...)

        schema = pydantic_to_json_instance_schema(M)
        for key in ("a_str", "a_int", "a_float", "a_bool"):
            assert "value" in schema[key]
            assert "citations" in schema[key]

        assert schema["a_str"]["value"] == "<string>"
        assert schema["a_int"]["value"] == "<integer>"
        assert schema["a_float"]["value"] == "<number>"
        assert schema["a_bool"]["value"] == "<boolean>"

    def test_comment_includes_description(self):
        class M(BaseModel):
            name: str = Field(..., description="the name")

        schema = pydantic_to_json_instance_schema(M)
        assert "desc: the name" in schema["name"]["comment"]

    def test_comment_includes_examples(self):
        class M(BaseModel):
            name: str = Field(..., description="name", examples=["Alice", "Bob"])

        schema = pydantic_to_json_instance_schema(M)
        assert "examples:" in schema["name"]["comment"]

    def test_comment_includes_default(self):
        class M(BaseModel):
            status: str = Field("active", description="status")

        schema = pydantic_to_json_instance_schema(M)
        assert "default: active" in schema["status"]["comment"]

    def test_enum_default_in_comment(self):
        class M(BaseModel):
            color: Color = Field(Color.RED, description="color")

        schema = pydantic_to_json_instance_schema(M)
        assert "default: red" in schema["color"]["comment"]

    def test_factory_default_in_comment(self):
        class M(BaseModel):
            tags: list[str] = Field(default_factory=list, description="tags")

        schema = pydantic_to_json_instance_schema(M)
        # list of primitives → list-wrapped leaf
        leaf = schema["tags"][0]
        assert "default: []" in leaf["comment"]

    def test_optional_field(self):
        class M(BaseModel):
            maybe: Optional[str] = Field(None, description="optional")

        schema = pydantic_to_json_instance_schema(M)
        assert schema["maybe"]["value"] == "<string>"
        assert "citations" in schema["maybe"]

    def test_nested_model(self):
        schema = pydantic_to_json_instance_schema(Person)
        assert isinstance(schema["address"], dict)
        assert "value" in schema["address"]["street"]
        assert "citations" in schema["address"]["street"]

    def test_optional_nested_model(self):
        class M(BaseModel):
            addr: Optional[Address] = Field(None, description="optional addr")

        schema = pydantic_to_json_instance_schema(M)
        assert isinstance(schema["addr"], dict)
        assert "value" in schema["addr"]["street"]

    def test_list_of_primitives(self):
        class M(BaseModel):
            ids: list[int] = Field(..., description="id list")

        schema = pydantic_to_json_instance_schema(M)
        assert isinstance(schema["ids"], list)
        assert len(schema["ids"]) == 1
        assert schema["ids"][0]["value"] == "<integer>"
        assert "citations" in schema["ids"][0]

    def test_list_of_nested_models(self):
        schema = pydantic_to_json_instance_schema(Invoice)
        assert isinstance(schema["items"], list)
        assert len(schema["items"]) == 1
        nested = schema["items"][0]
        assert "product" in nested
        assert "value" in nested["product"]

    def test_deep_nesting(self):
        schema = pydantic_to_json_instance_schema(Trip)
        assert "value" in schema["origin"]["coord"]["lat"]
        assert schema["origin"]["coord"]["lat"]["value"] == "<number>"

    def test_datetime_and_date(self):
        class M(BaseModel):
            ts: datetime = Field(...)
            d: date = Field(...)

        schema = pydantic_to_json_instance_schema(M)
        assert schema["ts"]["value"] == "<string>"
        assert schema["d"]["value"] == "<string>"

    def test_citation_level_line(self):
        class M(BaseModel):
            v: str = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation_level="line")
        cit = schema["v"]["citations"][0]
        assert "page" in cit
        assert "lines" in cit

    def test_citation_level_page(self):
        class M(BaseModel):
            v: str = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation_level="page")
        cit = schema["v"]["citations"][0]
        assert "page" in cit
        assert "lines" not in cit

    def test_bare_model_no_metadata(self):
        class M(BaseModel):
            x: str
            y: int

        schema = pydantic_to_json_instance_schema(M)
        assert schema["x"]["value"] == "<string>"
        assert schema["x"]["comment"] == ""


# ---------------------------------------------------------------------------
# pydantic_to_json_instance_schema — citation=False
# ---------------------------------------------------------------------------
class TestSchemaWithoutCitation:
    """Tests for the citation=False path."""

    def test_primitive_fields_are_plain_strings(self):
        class M(BaseModel):
            a_str: str = Field(...)
            a_int: int = Field(...)
            a_float: float = Field(...)
            a_bool: bool = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema["a_str"] == "<string>"
        assert schema["a_int"] == "<integer>"
        assert schema["a_float"] == "<number>"
        assert schema["a_bool"] == "<boolean>"

    def test_no_value_or_citations_keys(self):
        class M(BaseModel):
            name: str = Field(..., description="name")

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert isinstance(schema["name"], str)
        assert "value" not in schema
        assert "citations" not in schema

    def test_optional_field(self):
        class M(BaseModel):
            maybe: Optional[str] = Field(None)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema["maybe"] == "<string>"

    def test_nested_model(self):
        schema = pydantic_to_json_instance_schema(Person, citation=False)
        assert isinstance(schema["address"], dict)
        assert schema["address"]["street"] == "<string>"
        assert schema["address"]["city"] == "<string>"

    def test_optional_nested_model(self):
        class M(BaseModel):
            addr: Optional[Address] = Field(None)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema["addr"]["street"] == "<string>"

    def test_list_of_primitives(self):
        class M(BaseModel):
            ids: list[int] = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema["ids"] == ["<integer>"]

    def test_list_of_nested_models(self):
        schema = pydantic_to_json_instance_schema(Invoice, citation=False)
        assert isinstance(schema["items"], list)
        assert len(schema["items"]) == 1
        assert schema["items"][0]["product"] == "<string>"
        assert schema["items"][0]["qty"] == "<integer>"

    def test_deep_nesting(self):
        schema = pydantic_to_json_instance_schema(Trip, citation=False)
        assert schema["origin"]["coord"]["lat"] == "<number>"
        assert schema["destination"]["coord"]["lon"] == "<number>"

    def test_datetime_and_date(self):
        class M(BaseModel):
            ts: datetime = Field(...)
            d: date = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema["ts"] == "<string>"
        assert schema["d"] == "<string>"

    def test_enum_field(self):
        class M(BaseModel):
            color: Color = Field(Color.RED)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema["color"] == "<string>"

    def test_citation_level_ignored(self):
        class M(BaseModel):
            v: str = Field(...)

        s1 = pydantic_to_json_instance_schema(M, citation=False, citation_level="page")
        s2 = pydantic_to_json_instance_schema(M, citation=False, citation_level="line")
        assert s1 == s2
        assert s1["v"] == "<string>"

    def test_bare_model(self):
        class M(BaseModel):
            x: str
            y: int

        schema = pydantic_to_json_instance_schema(M, citation=False)
        assert schema == {"x": "<string>", "y": "<integer>"}


# ---------------------------------------------------------------------------
# stringify_schema
# ---------------------------------------------------------------------------
class TestStringifySchema:
    """Tests that stringify_schema produces correct text for both modes."""

    def test_citation_true_has_value_and_citations(self):
        class M(BaseModel):
            name: str = Field(..., description="the name")

        schema = pydantic_to_json_instance_schema(M)
        text = stringify_schema(schema)
        assert '"value": <string>' in text
        assert '"citations":' in text
        assert "# desc: the name" in text

    def test_citation_false_no_value_no_citations(self):
        class M(BaseModel):
            name: str = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        text = stringify_schema(schema)
        assert '"name": <string>' in text
        assert '"value"' not in text
        assert '"citations"' not in text

    def test_citation_false_no_quotes_on_placeholders(self):
        class M(BaseModel):
            a: str = Field(...)
            b: int = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        text = stringify_schema(schema)
        # Placeholders must NOT be quoted
        assert '"<string>"' not in text
        assert '"<integer>"' not in text
        # But they must be present unquoted
        assert "<string>" in text
        assert "<integer>" in text

    def test_citation_true_no_quotes_on_placeholders(self):
        class M(BaseModel):
            a: str = Field(...)

        schema = pydantic_to_json_instance_schema(M)
        text = stringify_schema(schema)
        # "value": <string> — not "value": "<string>"
        assert '"value": <string>' in text

    def test_list_of_primitives_citation_false(self):
        class M(BaseModel):
            ids: list[int] = Field(...)

        schema = pydantic_to_json_instance_schema(M, citation=False)
        text = stringify_schema(schema)
        assert '"ids": [<integer>]' in text

    def test_list_of_nested_models_citation_false(self):
        schema = pydantic_to_json_instance_schema(Invoice, citation=False)
        text = stringify_schema(schema)
        assert '"items": [' in text
        assert '"product": <string>' in text
        assert '"qty": <integer>' in text

    def test_nested_model_citation_false(self):
        schema = pydantic_to_json_instance_schema(Person, citation=False)
        text = stringify_schema(schema)
        assert '"address": {' in text
        assert '"street": <string>' in text

    def test_deep_nesting_citation_false(self):
        schema = pydantic_to_json_instance_schema(Trip, citation=False)
        text = stringify_schema(schema)
        assert '"origin": {' in text
        assert '"coord": {' in text
        assert '"lat": <number>' in text

    def test_empty_schema(self):
        class M(BaseModel):
            pass

        schema = pydantic_to_json_instance_schema(M, citation=False)
        text = stringify_schema(schema)
        assert text == "{}"


# ---------------------------------------------------------------------------
# schema_to_json
# ---------------------------------------------------------------------------
class TestSchemaToJson:
    def test_removes_comment_key(self):
        class M(BaseModel):
            name: str = Field(..., description="name")

        schema = pydantic_to_json_instance_schema(M)
        json_str = schema_to_json(schema)
        assert '"comment"' not in json_str
        assert '"value"' in json_str
        assert '"citations"' in json_str

    def test_citation_false_produces_valid_json(self):
        import json

        schema = pydantic_to_json_instance_schema(Person, citation=False)
        json_str = schema_to_json(schema)
        parsed = json.loads(json_str)
        assert parsed["name"] == "<string>"
        assert parsed["address"]["street"] == "<string>"

    def test_list_of_nested_models_valid_json(self):
        import json

        schema = pydantic_to_json_instance_schema(Invoice, citation=False)
        json_str = schema_to_json(schema)
        parsed = json.loads(json_str)
        assert parsed["items"][0]["product"] == "<string>"
