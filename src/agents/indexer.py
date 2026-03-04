"""
Stage 4: PageIndex Tree Builder
Builds a hierarchical navigation index over a document —
the 'smart table of contents' for LLM traversal.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.models import ExtractedDocument, LDU, PageIndex, Section

_PAGEINDEX_DIR = Path(".refinery") / "pageindex"


def _section_id() -> str:
    return "sec_" + uuid.uuid4().hex[:10]


class PageIndexBuilder:
    """
    Traverses an ExtractedDocument's section hierarchy and builds a PageIndex.
    Generates LLM summaries for each section when Gemini API is available.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(os.getenv("GEMINI_API_KEY"))
        self.pageindex_dir = _PAGEINDEX_DIR
        self.pageindex_dir.mkdir(parents=True, exist_ok=True)
        self._llm_client = None

    def _get_llm(self):
        if self._llm_client is None and self.use_llm:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._llm_client = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                self.use_llm = False
        return self._llm_client

    def _generate_summary(self, section_text: str, title: str) -> str:
        """Generate a 2-3 sentence summary using Gemini Flash."""
        if not self.use_llm:
            return self._extract_key_sentences(section_text)

        try:
            client = self._get_llm()
            if not client:
                return self._extract_key_sentences(section_text)

            prompt = (
                f"Section: {title}\n\n"
                f"Content (first 1500 chars):\n{section_text[:1500]}\n\n"
                "Write a 2-3 sentence factual summary of this section. "
                "Focus on key facts, numbers, and findings. Be concise."
            )
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 200},
            )
            return response.text.strip()
        except Exception:
            return self._extract_key_sentences(section_text)

    def _extract_key_sentences(self, text: str) -> str:
        """Fallback: extract first 2 meaningful sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        meaningful = [s for s in sentences if len(s) > 30]
        return " ".join(meaningful[:2]) if meaningful else text[:200]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities: numbers, organizations, currencies."""
        entities: List[str] = []
        # Money amounts
        entities += re.findall(r"\b(?:ETB|USD|\$|Birr)[\s]?[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?\b", text)
        # Percentages
        entities += re.findall(r"\b\d+(?:\.\d+)?%", text)
        # Years
        entities += re.findall(r"\b(20\d\d|19\d\d)\b", text)
        # Q1/Q2/Q3/Q4
        entities += re.findall(r"\bQ[1-4]\s*\d{4}\b", text)
        return list(dict.fromkeys(entities))[:10]  # deduplicate, cap at 10

    def build(self, doc: ExtractedDocument, ldus: List[LDU]) -> PageIndex:
        """Build a hierarchical PageIndex from ExtractedDocument and its LDUs."""
        sections: List[Section] = []
        current_section: Optional[Section] = None
        root_sections: List[Section] = []

        # Build LDU lookup by section
        ldu_by_section: dict[str, List[LDU]] = {}
        for ldu in ldus:
            key = ldu.parent_section or "_root"
            ldu_by_section.setdefault(key, []).append(ldu)

        # Build section tree from heading LDUs
        heading_ldus = [l for l in ldus if l.chunk_type == "heading"]

        if not heading_ldus:
            # No headings: create one root section for entire document
            root_text = " ".join(b.content for b in ldus if b.chunk_type == "text")[:3000]
            root_section = Section(
                section_id=_section_id(),
                title=doc.filename,
                page_start=1,
                page_end=doc.page_count,
                level=1,
                summary=self._generate_summary(root_text, doc.filename),
                key_entities=self._extract_entities(root_text),
                data_types_present=self._detect_data_types(ldus),
                ldu_ids=[l.ldu_id for l in ldus],
            )
            root_sections.append(root_section)
        else:
            for h_ldu in heading_ldus:
                section_title = h_ldu.content.strip()
                page_start = h_ldu.page_refs[0] if h_ldu.page_refs else 1

                # Gather child LDUs for this section
                child_ldus = ldu_by_section.get(section_title, [])
                section_text = " ".join(l.content for l in child_ldus)[:3000]

                # Detect depth level from heading style
                level = self._detect_heading_level(section_title)

                section = Section(
                    section_id=_section_id(),
                    title=section_title,
                    page_start=page_start,
                    page_end=page_start,  # updated below
                    level=level,
                    summary=self._generate_summary(section_text, section_title),
                    key_entities=self._extract_entities(section_text),
                    data_types_present=self._detect_data_types(child_ldus),
                    ldu_ids=[l.ldu_id for l in child_ldus] + [h_ldu.ldu_id],
                )

                if level == 1 or not root_sections:
                    root_sections.append(section)
                    current_section = section
                else:
                    parent = root_sections[-1] if root_sections else None
                    if parent:
                        parent.child_sections.append(section)
                    else:
                        root_sections.append(section)

            # Fix page_end values
            for i, sec in enumerate(root_sections):
                if i + 1 < len(root_sections):
                    sec.page_end = root_sections[i + 1].page_start - 1
                else:
                    sec.page_end = doc.page_count

        page_index = PageIndex(
            doc_id=doc.doc_id,
            document_name=doc.filename,
            page_count=doc.page_count,
            sections=root_sections,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save to disk
        out_path = self.pageindex_dir / f"{doc.doc_id}.json"
        out_path.write_text(page_index.model_dump_json(indent=2))
        return page_index

    def _detect_heading_level(self, title: str) -> int:
        """Detect heading level from font size markers or numbering patterns."""
        # Numbered like 1. or 1.1 or 1.1.1
        if re.match(r"^\d+\.\d+\.\d+", title):
            return 3
        if re.match(r"^\d+\.\d+", title):
            return 2
        if re.match(r"^\d+\.", title) or len(title) < 30:
            return 1
        return 2

    def _detect_data_types(self, ldus: List[LDU]) -> List[str]:
        types_present = set()
        for ldu in ldus:
            if ldu.chunk_type in ("text", "table", "figure", "equation", "list"):
                types_present.add(ldu.chunk_type)
        return list(types_present)
