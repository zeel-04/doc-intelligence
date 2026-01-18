from typing import Any

from .base import BaseExtractor
from .llm import BaseLLM
from .schemas import Document, PydanticModel
from .utils import add_bboxes_to_citation_model, enrich_citations_with_bboxes


class DigitalPDFExtractor(BaseExtractor):
    def __init__(
        self,
        llm: BaseLLM,
    ):
        super().__init__(llm)

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
        include_line_numbers: bool,
        llm_input: str,
        citation_type: Any,
        citation_type_with_bboxes: Any,
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
        # Modify the response format to add mode based citations
        # if include_line_numbers:
        #     response_format = add_appropriate_citation_type(
        #         response_format, CitationType
        #     )

        response = self.llm.generate_structured_output(
            model=model,
            messages=messages,
            reasoning=reasoning,
            output_format=response_format,
            openai_text=openai_text,
        )

        # enrich the response with bboxes
        if include_line_numbers:
            response_with_bboxes = enrich_citations_with_bboxes(
                response,  # type:ignore
                document.content,  # type: ignore
            )

            final_cited_response_model = add_bboxes_to_citation_model(
                model=response_format,
                original_citation_type=citation_type,
                new_citation_type=citation_type_with_bboxes,
            )
            return final_cited_response_model(**response_with_bboxes)  # type:ignore

        return response  # type:ignore
