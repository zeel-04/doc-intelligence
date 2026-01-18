from typing import Any

from loguru import logger

from .base import BaseExtractor
from .llm import BaseLLM
from .schemas.core import Document, PydanticModel
from .utils import add_bboxes_to_citation_model, enrich_citations_with_bboxes


class DigitalPDFExtractor(BaseExtractor):
    def __init__(
        self,
        llm: BaseLLM,
        include_line_numbers: bool = True,
        citation_type: Any = None,
        citation_type_with_bboxes: Any = None,
    ):
        super().__init__(
            llm, include_line_numbers, citation_type, citation_type_with_bboxes
        )

        self.system_prompt = """Act as an expert in the field of document extraction and information extraction from documents.
Note:
- If user defines citations, use the page number and line number where the information is mentioned in the document.
Example: [{"page": 1, "lines": [10, 11]}, {"page": 2, "lines": [20]}]
"""
        self.user_prompt = """Please extract the information from the below document.
Document: {document}
"""

    def extract(
        self,
        document: Document,
        model: str,
        reasoning: Any,
        response_format: type[PydanticModel],
        llm_input: str,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        openai_text: dict[str, Any] | None = None,
    ) -> PydanticModel:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {
                "role": "user",
                "content": user_prompt or self.user_prompt.format(document=llm_input),
            },
        ]

        response = self.llm.generate_structured_output(
            model=model,
            messages=messages,
            reasoning=reasoning,
            output_format=response_format,
            openai_text=openai_text,
        )

        # enrich the response with bboxes
        if self.include_line_numbers:
            response_with_bboxes = enrich_citations_with_bboxes(
                response,  # type:ignore
                document.content,  # type: ignore
            )

            final_cited_response_model = add_bboxes_to_citation_model(
                model=response_format,
                original_citation_type=self.citation_type,
                new_citation_type=self.citation_type_with_bboxes,
            )
            return final_cited_response_model(**response_with_bboxes)  # type:ignore

        return response  # type:ignore
