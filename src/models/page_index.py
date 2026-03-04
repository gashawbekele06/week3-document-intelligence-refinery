from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """A node in the hierarchical PageIndex tree."""

    model_config = ConfigDict()

    section_id: str
    title: str
    page_start: int
    page_end: int
    level: int = 1  # 1=top-level chapter, 2=section, 3=subsection

    child_sections: List["Section"] = Field(default_factory=list)
    key_entities: List[str] = Field(
        default_factory=list, description="Named entities extracted from this section"
    )
    summary: Optional[str] = Field(
        default=None, description="LLM-generated 2-3 sentence summary"
    )
    data_types_present: List[Literal["text", "table", "figure", "equation", "list"]] = (
        Field(default_factory=list)
    )
    ldu_ids: List[str] = Field(
        default_factory=list, description="LDU IDs belonging to this section"
    )


class PageIndex(BaseModel):
    """
    Hierarchical navigation index over a document.
    Equivalent to a 'smart table of contents' enabling LLM traversal
    without reading the full document.
    """

    model_config = ConfigDict()

    doc_id: str
    document_name: str
    page_count: int
    sections: List[Section] = Field(default_factory=list)
    created_at: str = ""

    def find_sections_for_query(self, query: str, top_k: int = 3) -> List[Section]:
        """
        Traverse tree to find most relevant sections for a topic query.
        Returns top_k sections by keyword overlap.
        """
        query_terms = set(query.lower().split())
        scored: List[tuple[float, Section]] = []

        def _score_section(section: Section) -> None:
            text = (section.title + " " + (section.summary or "")).lower()
            text_terms = set(text.split())
            overlap = len(query_terms & text_terms)
            entity_overlap = sum(
                1 for e in section.key_entities if any(t in e.lower() for t in query_terms)
            )
            score = overlap + 2 * entity_overlap
            scored.append((score, section))
            for child in section.child_sections:
                _score_section(child)

        for section in self.sections:
            _score_section(section)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k] if _ > 0]
