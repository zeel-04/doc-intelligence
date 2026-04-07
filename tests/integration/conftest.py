"""Fixtures and configuration for data-driven integration tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import NestedExtraction, SimpleExtraction

# ---------------------------------------------------------------------------
# Schema registry — maps string names in test cases to Pydantic models
# ---------------------------------------------------------------------------
SCHEMA_REGISTRY: dict[str, Any] = {
    "SimpleExtraction": SimpleExtraction,
    "NestedExtraction": NestedExtraction,
}


def resolve_schema(name: str) -> type:
    """Look up a Pydantic model by name from the schema registry.

    Args:
        name: The string name of the schema (must be a key in SCHEMA_REGISTRY).

    Returns:
        The Pydantic model class.

    Raises:
        KeyError: If the name is not found in the registry.
    """
    if name not in SCHEMA_REGISTRY:
        msg = (
            f"Schema '{name}' not found in SCHEMA_REGISTRY. "
            f"Available: {list(SCHEMA_REGISTRY.keys())}"
        )
        raise KeyError(msg)
    return SCHEMA_REGISTRY[name]


# ---------------------------------------------------------------------------
# Pytest CLI flag and marker setup
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-live CLI flag for live LLM tests."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run integration tests that make real LLM API calls.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the 'live' marker."""
    config.addinivalue_line(
        "markers",
        "live: marks tests that require real LLM API calls (deselect with '-m \"not live\"')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip @pytest.mark.live tests when --run-live is not set."""
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="Need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PDFS_DIR = Path(__file__).parent / "pdfs"


@pytest.fixture
def pdf_path() -> Callable[[str], str]:
    """Return a callable that resolves a PDF filename to its absolute path.

    Usage in tests::

        def test_parse(pdf_path):
            path = pdf_path("simple_one_page.pdf")
    """

    def _resolve(filename: str) -> str:
        full_path = PDFS_DIR / filename
        if not full_path.exists():
            msg = f"Test PDF not found: {full_path}"
            raise FileNotFoundError(msg)
        return str(full_path)

    return _resolve


@pytest.fixture
def is_live(request: pytest.FixtureRequest) -> bool:
    """Return True if --run-live was passed."""
    return request.config.getoption("--run-live")
