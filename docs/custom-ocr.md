# Custom OCR Components

The scanned PDF pipeline is fully pluggable. You can replace the default
`PaddleLayoutDetector` and `PaddleOCREngine` with any implementation that
satisfies the abstract base class contracts.

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
    DocumentProcessor,
    LayoutRegion,
)
from doc_intelligence.pdf.schemas import Line
from doc_intelligence.llm import OpenAILLM


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
        return [Line(text="Hello, world!")]


llm = OpenAILLM(model="gpt-4o")
processor = DocumentProcessor.from_scanned_pdf(
    llm=llm,
    layout_detector=MyLayoutDetector(),
    ocr_engine=MyOCREngine(),
    dpi=150,
)

result = processor.extract("scanned.pdf", MySchema)
```

## Using ScannedPDFParser Directly

You can also instantiate `ScannedPDFParser` directly and pass it to
`DocumentProcessor` alongside any formatter and extractor:

```python
from doc_intelligence import DocumentProcessor, ScannedPDFParser
from doc_intelligence.pdf.formatter import DigitalPDFFormatter
from doc_intelligence.pdf.extractor import DigitalPDFExtractor
from doc_intelligence.llm import OpenAILLM

llm = OpenAILLM()
parser = ScannedPDFParser(
    layout_detector=MyLayoutDetector(),
    ocr_engine=MyOCREngine(),
    dpi=200,
)
processor = DocumentProcessor(
    parser=parser,
    formatter=DigitalPDFFormatter(),
    extractor=DigitalPDFExtractor(llm),
)
```
