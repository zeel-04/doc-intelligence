"""Tests for restrictions module."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from doc_intelligence.restrictions import (
    check_page_count,
    check_pdf_size,
    check_schema_depth,
)


# ---------------------------------------------------------------------------
# check_pdf_size
# ---------------------------------------------------------------------------
class TestCheckPdfSize:
    def test_passes_when_under_limit(self, tmp_path):
        pdf = tmp_path / "small.pdf"
        pdf.write_bytes(b"x" * 1024)  # 1 KB
        check_pdf_size(str(pdf), max_mb=10.0)  # should not raise

    def test_raises_when_over_limit(self, tmp_path):
        pdf = tmp_path / "large.pdf"
        pdf.write_bytes(b"x" * (11 * 1024 * 1024))  # 11 MB
        with pytest.raises(ValueError, match="exceeds max size"):
            check_pdf_size(str(pdf), max_mb=10.0)

    def test_error_message_contains_sizes(self, tmp_path):
        pdf = tmp_path / "big.pdf"
        pdf.write_bytes(b"x" * (12 * 1024 * 1024))  # 12 MB
        with pytest.raises(ValueError, match="12.0 MB > 10.0 MB"):
            check_pdf_size(str(pdf), max_mb=10.0)

    def test_passes_exactly_at_limit(self, tmp_path):
        pdf = tmp_path / "exact.pdf"
        pdf.write_bytes(b"x" * (10 * 1024 * 1024))  # exactly 10 MB
        check_pdf_size(str(pdf), max_mb=10.0)  # should not raise

    def test_non_local_uri_skipped(self):
        # URLs are not size-checked — should not raise even with 0 MB limit
        check_pdf_size("https://example.com/doc.pdf", max_mb=0.0)

    def test_missing_file_skipped(self, tmp_path):
        # Non-existent path is not a file, so it's skipped (parse will fail later)
        check_pdf_size(str(tmp_path / "missing.pdf"), max_mb=0.0)


# ---------------------------------------------------------------------------
# check_page_count
# ---------------------------------------------------------------------------
class TestCheckPageCount:
    def test_passes_when_under_limit(self):
        check_page_count(50, max_pages=100)  # should not raise

    def test_passes_exactly_at_limit(self):
        check_page_count(100, max_pages=100)  # should not raise

    def test_raises_when_over_limit(self):
        with pytest.raises(ValueError, match="101 pages, limit is 100"):
            check_page_count(101, max_pages=100)

    def test_raises_for_zero_limit(self):
        with pytest.raises(ValueError, match="1 pages, limit is 0"):
            check_page_count(1, max_pages=0)

    @pytest.mark.parametrize("count,limit", [(5, 10), (99, 100), (0, 1)])
    def test_parametrized_passes(self, count, limit):
        check_page_count(count, max_pages=limit)  # should not raise

    @pytest.mark.parametrize("count,limit", [(11, 10), (101, 100), (1, 0)])
    def test_parametrized_raises(self, count, limit):
        with pytest.raises(ValueError):
            check_page_count(count, max_pages=limit)


# ---------------------------------------------------------------------------
# check_schema_depth
# ---------------------------------------------------------------------------
class Flat(BaseModel):
    name: str
    age: int


class OneLevel(BaseModel):
    class Inner(BaseModel):
        value: str

    inner: Inner


class TwoLevel(BaseModel):
    class Mid(BaseModel):
        class Deep(BaseModel):
            x: int

        deep: Deep

    mid: Mid


class WithOptional(BaseModel):
    class Child(BaseModel):
        val: int

    child: Child | None = None


class WithList(BaseModel):
    class Item(BaseModel):
        id: int

    items: list[Item]


class TestCheckSchemaDepth:
    def test_flat_model_passes(self):
        check_schema_depth(Flat, max_depth=0)  # no nesting

    def test_one_level_passes_with_depth_1(self):
        check_schema_depth(OneLevel, max_depth=1)

    def test_one_level_raises_with_depth_0(self):
        with pytest.raises(ValueError, match="depth 1 exceeds limit 0"):
            check_schema_depth(OneLevel, max_depth=0)

    def test_two_levels_passes_with_depth_2(self):
        check_schema_depth(TwoLevel, max_depth=2)

    def test_two_levels_raises_with_depth_1(self):
        with pytest.raises(ValueError, match="exceeds limit 1"):
            check_schema_depth(TwoLevel, max_depth=1)

    def test_optional_nested_model_checked(self):
        with pytest.raises(ValueError):
            check_schema_depth(WithOptional, max_depth=0)

    def test_optional_nested_passes_with_depth_1(self):
        check_schema_depth(WithOptional, max_depth=1)

    def test_list_of_models_checked(self):
        with pytest.raises(ValueError):
            check_schema_depth(WithList, max_depth=0)

    def test_list_of_models_passes_with_depth_1(self):
        check_schema_depth(WithList, max_depth=1)

    def test_generous_depth_always_passes(self):
        check_schema_depth(TwoLevel, max_depth=10)
