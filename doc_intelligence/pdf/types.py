from enum import Enum


class PDFExtractionMode(Enum):
    SINGLE_PASS = "single_pass"
    MULTI_PASS = "multi_pass"


class ParseStrategy(Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"


class ScannedPipelineType(Enum):
    """Sub-pipeline type for the SCANNED parse strategy.

    - ``TWO_STAGE``: Separate layout detection + OCR engine (Type 1).
      *Not yet implemented.*
    - ``VLM``: Single-stage vision LLM for both layout and OCR (Type 2).
      This is the default and currently the only implemented pipeline.
    """

    TWO_STAGE = "two_stage"
    VLM = "vlm"
