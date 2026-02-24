import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, create_model

from doc_intelligence.llm import OpenAILLM
from doc_intelligence.processer import DocumentProcessor

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="centered",
)

# ── Schema presets ───────────────────────────────────────────────────────────
PRESETS: dict[str, dict[str, str]] = {
    "License": {
        "license_name": "str",
    },
    "Invoice": {
        "invoice_number": "str",
        "invoice_date": "str",
        "due_date": "str",
        "vendor_name": "str",
        "vendor_address": "str",
        "total_amount": "str",
        "tax_amount": "str",
        "line_items": "list[str]",
    },
    "Resume": {
        "full_name": "str",
        "email": "str",
        "phone": "str",
        "summary": "str",
        "skills": "list[str]",
        "experience": "list[str]",
        "education": "list[str]",
    },
    "Custom": {},
}

TYPE_MAP: dict[str, tuple] = {
    "str": (str, ...),
    "int": (int, ...),
    "float": (float, ...),
    "bool": (bool, ...),
    "list[str]": (list[str], ...),
    "list[int]": (list[int], ...),
    "list[float]": (list[float], ...),
}


def build_pydantic_model(
    schema_dict: dict[str, str],
    model_name: str = "ExtractedData",
) -> type[BaseModel]:
    """Create a Pydantic model dynamically from a {field_name: type_str} dict."""
    fields: dict = {}
    for name, type_str in schema_dict.items():
        fields[name] = TYPE_MAP.get(type_str, (str, ...))
    return create_model(model_name, **fields)


def serialize_result(obj: object) -> object:
    """Recursively convert Pydantic models / custom objects to JSON-safe dicts."""
    if isinstance(obj, BaseModel):
        return {k: serialize_result(v) for k, v in obj.model_dump().items()}
    if isinstance(obj, dict):
        return {k: serialize_result(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_result(v) for v in obj]
    return obj


# ── Sidebar – input & config ────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Input")

    input_mode = st.radio(
        "Input method",
        ["Upload PDF", "Enter URL"],
        horizontal=True,
    )

    pdf_uri: str | None = None
    tmp_path: str | None = None

    if input_mode == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="doc_ai_",
            )
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            tmp_path = tmp.name
            pdf_uri = tmp_path
            st.success(f"Uploaded: **{uploaded_file.name}**")
    else:
        url_input = st.text_input(
            "PDF URL",
            placeholder="https://example.com/document.pdf",
        )
        if url_input:
            pdf_uri = url_input

    st.divider()
    st.header("Extraction Config")

    model_name = st.selectbox(
        "LLM Model",
        ["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        index=0,
    )

    include_citations = st.toggle("Include citations", value=False)


# ── Main area – schema definition & results ──────────────────────────────────
st.title("Document Intelligence")
st.caption("Extract structured JSON from any PDF using LLMs")

# ── Schema editor ────────────────────────────────────────────────────────────
st.subheader("1 · Define extraction schema")

preset_name = st.selectbox(
    "Start from a preset",
    list(PRESETS.keys()),
    index=0,
)

default_schema = (
    json.dumps(PRESETS[preset_name], indent=2) if PRESETS[preset_name] else "{}"
)

schema_json = st.text_area(
    "Schema (field_name → type)",
    value=default_schema,
    height=220,
    help="Supported types: str, int, float, bool, list[str], list[int], list[float]",
)

# ── Extract button ───────────────────────────────────────────────────────────
st.subheader("2 · Extract")

extract_disabled = pdf_uri is None
extract_btn = st.button(
    "Extract data",
    type="primary",
    use_container_width=True,
    disabled=extract_disabled,
)

if extract_disabled:
    st.info("Upload a PDF or enter a URL to get started.")

# ── Run extraction ───────────────────────────────────────────────────────────
if extract_btn and pdf_uri:
    # Validate schema JSON
    try:
        schema_dict: dict[str, str] = json.loads(schema_json)
        if not schema_dict:
            st.error("Schema cannot be empty. Add at least one field.")
            st.stop()
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON in schema editor: {exc}")
        st.stop()

    response_model = build_pydantic_model(schema_dict)

    config = {
        "response_format": response_model,
        "llm_config": {
            "model": model_name,
        },
        "extraction_config": {
            "include_citations": include_citations,
            "extraction_mode": "single_pass",
        },
    }

    try:
        with st.spinner("Parsing PDF and extracting data…"):
            llm = OpenAILLM()
            processor = DocumentProcessor.from_digital_pdf(uri=pdf_uri, llm=llm)
            result = processor.extract(config)

        st.subheader("3 · Results")

        extracted = result.get("extracted_data")
        metadata = result.get("metadata")

        if include_citations and metadata:
            tab_data, tab_meta = st.tabs(["Extracted Data", "Citations / Metadata"])
            with tab_data:
                st.json(serialize_result(extracted), expanded=True)
            with tab_meta:
                st.json(serialize_result(metadata), expanded=True)
        else:
            st.json(serialize_result(extracted), expanded=True)

    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        st.exception(exc)

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
