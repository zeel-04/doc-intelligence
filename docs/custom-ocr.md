# Custom OCR Components (planned)

!!! warning
    The two-stage scanned pipeline (`ScannedPipelineType.TWO_STAGE`) is **not
    yet implemented**. The contracts below are defined and ready for use, but
    selecting `TWO_STAGE` will raise `NotImplementedError`. Use the VLM
    pipeline for scanned PDF extraction today.

The scanned PDF pipeline will be fully pluggable. Implement the
`BaseLayoutDetector` and `BaseOCREngine` abstract base class contracts
to provide your own layout detection and OCR capabilities.

## Contracts

### BaseLayoutDetector

```python
from doc_intelligence import BaseLayoutDetector, LayoutRegion
import numpy as np

class MyLayoutDetector(BaseLayoutDetector):
    def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
        """Segment a page image into typed regions.

        Args:
            page_image: HxWxC uint8 numpy array of the full page.

        Returns:
            List of LayoutRegion with pixel-coordinate bounding boxes,
            region type labels, and confidence scores.
        """
        ...
```

`LayoutRegion` fields:

| Field | Type | Description |
|---|---|---|
| `bounding_box` | `BoundingBox` | Pixel coordinates within the page image |
| `region_type` | `str` | Any label, e.g. `"text"`, `"table"`, `"figure"` |
| `confidence` | `float` | Detection confidence in `[0, 1]` |

Regions with `region_type == "table"` become `TableBlock`s in the output;
all other types become `TextBlock`s.

### BaseOCREngine

```python
from doc_intelligence import BaseOCREngine
from doc_intelligence.pdf.schemas import Line
import numpy as np

class MyOCREngine(BaseOCREngine):
    def ocr(self, region_image: np.ndarray) -> list[Line]:
        """Read text from a single cropped region image.

        Args:
            region_image: HxWxC uint8 numpy array of a cropped page region.

        Returns:
            List of Line with text and bounding boxes normalized to [0, 1]
            relative to the region image dimensions.
        """
        ...
```

`Line` fields:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The recognized text |
| `bounding_box` | `BoundingBox \| None` | Normalized coordinates in `[0, 1]` |

## Example: Custom Detector + Engine

```python
import numpy as np
from doc_intelligence import (
    BaseLayoutDetector,
    BaseOCREngine,
    BoundingBox,
    PDFProcessor,
    ParseStrategy,
    ScannedPipelineType,
    LayoutRegion,
)
from doc_intelligence.schemas.core import Line


class MyLayoutDetector(BaseLayoutDetector):
    def detect(self, page_image: np.ndarray) -> list[LayoutRegion]:
        # Treat the entire page as a single text region
        h, w = page_image.shape[:2]
        return [
            LayoutRegion(
                bounding_box=BoundingBox(x0=0, top=0, x1=w, bottom=h),
                region_type="text",
                confidence=1.0,
            )
        ]


class MyOCREngine(BaseOCREngine):
    def ocr(self, region_image: np.ndarray) -> list[Line]:
        # Call your OCR service here
        return [Line(text="Hello, world!", bounding_box=BoundingBox(x0=0, top=0, x1=1, bottom=1))]


processor = PDFProcessor(
    provider="openai",
    strategy=ParseStrategy.SCANNED,
    scanned_pipeline=ScannedPipelineType.TWO_STAGE,
    layout_detector=MyLayoutDetector(),
    ocr_engine=MyOCREngine(),
    dpi=150,
)

result = processor.extract("scanned.pdf", MySchema)
```

## Using PDFParser Directly

You can also instantiate `PDFParser` directly and pass it to
`DocumentProcessor` alongside any formatter and extractor:

```python
from doc_intelligence import DocumentProcessor
from doc_intelligence.pdf.parser import PDFParser
from doc_intelligence.pdf.formatter import PDFFormatter
from doc_intelligence.pdf.extractor import PDFExtractor
from doc_intelligence.pdf.types import ParseStrategy, ScannedPipelineType
from doc_intelligence.llm import OpenAILLM

llm = OpenAILLM()
parser = PDFParser(
    strategy=ParseStrategy.SCANNED,
    scanned_pipeline=ScannedPipelineType.TWO_STAGE,
    layout_detector=MyLayoutDetector(),
    ocr_engine=MyOCREngine(),
    dpi=200,
)
processor = DocumentProcessor(
    parser=parser,
    formatter=PDFFormatter(),
    extractor=PDFExtractor(llm),
)
```
