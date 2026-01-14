from typing import Any

from pydantic import BaseModel
from typing_extensions import TypedDict

from .base import BaseExtractor
from .llm import BaseLLM
from .schemas import Document, Mode, PydanticModel
from .types import CitationType, CitationWithBboxesType
from .utils import add_appropriate_citation_type, enrich_citations_with_bboxes


class PDFExtractor(BaseExtractor):
    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        super().__init__(llm, system_prompt, user_prompt)
        if not system_prompt:
            self.system_prompt = """Act as an expert in the field of document extraction and information extraction from documents.
Note:
- If user defines citations, use the page number and line number where the information is mentioned in the document.
Example:
[{"page": 1, "lines": [10, 11]}, {"page": 2, "lines": [20]}]
"""
        if not user_prompt:
            self.user_prompt = """Please extract the information from the below document.
Document:
{document}
"""

    def extract(
        self,
        document: Document,
        model: str,
        reasoning: Any,
        response_format: PydanticModel,
        mode: Mode,
        llm_input: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {
                "role": "user",
                "content": user_prompt or self.user_prompt.format(document=llm_input),  # type:ignore[reportFormatStrCall]
            },
        ]
        # Modify the response format to add mode based citations
        if mode.include_line_numbers:
            response_format = add_appropriate_citation_type(
                response_format, CitationType
            )

        # else:
        #     response_format = set_page_citations(response_format, "Citations")
        response = self.llm.generate_structured_output(
            model=model,
            messages=messages,  # type:ignore[arg-type]
            reasoning=reasoning,
            output_format=response_format,
            openai_text=openai_text,
        )

        # enrich the response with bboxes
        if mode.include_line_numbers:
            response_with_bboxes = enrich_citations_with_bboxes(
                response,
                document.content,  # type: ignore
            )

            final_cited_response_model = add_appropriate_citation_type(
                response_format,
                CitationWithBboxesType,
            )
            return final_cited_response_model(**response_with_bboxes)  # ty:ignore[call-non-callable]
        return response
