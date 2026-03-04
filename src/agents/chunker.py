"""
Stage 3: Semantic Chunking Engine
Converts ExtractedDocument into RAG-ready Logical Document Units (LDUs).
Enforces 5 chunking rules via ChunkValidator to guarantee data quality.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List, Optional
import yaml

from src.models import (
    BBoxRef,
    ExtractedDocument,
    ExtractedTable,
    LDU,
    TextBlock,
)

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"


def _load_rules() -> dict:
    with open(_RULES_PATH) as f:
        return yaml.safe_load(f)


def _estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for English)."""
    return max(1, len(text) // 4)


def _to_bbox_ref(bbox, page: int = -1) -> Optional[BBoxRef]:
    if bbox is None:
        return None
    if hasattr(bbox, "x0"):
        return BBoxRef(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1, page=bbox.page)
    return None


# ─── Chunk Validator ─────────────────────────────────────────────────────────

class ChunkValidationError(Exception):
    """Raised when a proposed LDU violates a chunking rule."""


class ChunkValidator:
    """
    Enforces the chunking constitution from extraction_rules.yaml.
    All 5 rules are checked before an LDU is emitted.
    """

    def __init__(self, rules: dict):
        self.rules = rules.get("chunking", {})

    def validate_table(self, ldu: LDU, table: ExtractedTable) -> None:
        """Rule 1: A table LDU must include all cells (header + rows in same LDU)."""
        if self.rules.get("table_always_together", True):
            if ldu.chunk_type == "table":
                # Check that content includes header markers
                if table.headers and not any(h in ldu.content for h in table.headers[:1]):
                    raise ChunkValidationError(
                        f"Table LDU {ldu.ldu_id} is missing header row — Rule 1 violated."
                    )

    def validate_figure_caption(self, ldu: LDU) -> None:
        """Rule 2: Figure LDU must have caption stored as metadata."""
        if self.rules.get("figure_caption_as_metadata", True):
            if ldu.chunk_type == "figure":
                # Caption must be in metadata, not content body
                if "caption" not in ldu.metadata:
                    ldu.metadata["caption"] = ""  # ensure key exists

    def validate_list_integrity(self, ldu: LDU) -> None:
        """Rule 3: Numbered lists stay as a single LDU unless > max_tokens."""
        max_tokens = self.rules.get("max_tokens_per_ldu", 512)
        if ldu.chunk_type == "list" and ldu.token_count > max_tokens * 2:
            raise ChunkValidationError(
                f"List LDU {ldu.ldu_id} exceeds 2x max token limit — consider splitting."
            )

    def validate_section_header(self, ldu: LDU) -> None:
        """Rule 4: parent_section must be set for all non-heading LDUs."""
        if self.rules.get("header_as_parent_metadata", True):
            if ldu.chunk_type not in ("heading",) and ldu.parent_section is None:
                ldu.parent_section = "_root"  # assign root section as fallback

    def validate_all(self, ldu: LDU, source_table: Optional[ExtractedTable] = None) -> LDU:
        """Run all validation rules and fix issues in-place where possible."""
        if source_table:
            self.validate_table(ldu, source_table)
        self.validate_figure_caption(ldu)
        self.validate_list_integrity(ldu)
        self.validate_section_header(ldu)
        return ldu


# ─── Chunking Engine ─────────────────────────────────────────────────────────

class ChunkingEngine:
    """
    Converts ExtractedDocument into List[LDU].
    
    Processing order:
    1. Section headings → heading LDUs (set current_section context)
    2. Text blocks → text/list LDUs (grouped by section)
    3. Tables → table LDUs (always atomic)
    4. Figures → figure LDUs (caption in metadata)
    """

    def __init__(self):
        self.rules = _load_rules()
        self.validator = ChunkValidator(self.rules)
        self.max_tokens = self.rules.get("chunking", {}).get("max_tokens_per_ldu", 512)

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        """Main entry point: emit all LDUs from an ExtractedDocument."""
        ldus: List[LDU] = []
        current_section: Optional[str] = None

        # ── Build ordered item list (heading, text, table, figure) sorted by page/order ──
        heading_pages = {b.page for b in doc.section_headings}

        # Pass 1: headings
        for block in doc.section_headings:
            ldu = self._make_text_ldu(block, "heading", current_section, doc)
            ldus.append(ldu)
            current_section = block.text[:80]  # use heading text as section key

        # Pass 2: text blocks (non-heading, reading-order sorted)
        text_only = [b for b in doc.text_blocks if not b.is_header]
        # Group consecutive text into paragraph chunks
        ldus.extend(self._chunk_text_blocks(text_only, current_section, doc))

        # Pass 3: tables (atomic)
        for table in doc.tables:
            ldu = self._make_table_ldu(table, current_section, doc)
            ldus.append(ldu)

        # Pass 4: figures
        for fig in doc.figures:
            ldu = self._make_figure_ldu(fig, current_section, doc)
            ldus.append(ldu)

        # Rule 5: resolve cross-references
        if self.rules.get("chunking", {}).get("resolve_cross_references", True):
            ldus = self._resolve_cross_references(ldus)

        return ldus

    def _make_text_ldu(
        self, block: TextBlock, chunk_type: str, section: Optional[str], doc: ExtractedDocument
    ) -> LDU:
        token_count = _estimate_tokens(block.text)
        ldu = LDU(
            ldu_id=f"ldu_{uuid.uuid4().hex[:12]}",
            content=block.text,
            chunk_type=chunk_type,
            page_refs=[block.page] if block.page >= 0 else [],
            bounding_box=_to_bbox_ref(block.bbox),
            parent_section=section,
            doc_id=doc.doc_id,
            document_name=doc.filename,
            token_count=token_count,
        )
        ldu.content_hash = LDU.compute_hash(block.text)
        return self.validator.validate_all(ldu)

    def _chunk_text_blocks(
        self, blocks: List[TextBlock], section: Optional[str], doc: ExtractedDocument
    ) -> List[LDU]:
        """
        Group consecutive text blocks into paragraph-sized LDUs.
        Detect numbered lists and keep them together.
        """
        ldus: List[LDU] = []
        buffer_texts: List[str] = []
        buffer_pages: List[int] = []
        buffer_bbox = None
        is_list = False

        def flush_buffer(chunk_type: str) -> None:
            if not buffer_texts:
                return
            content = " ".join(buffer_texts)
            token_count = _estimate_tokens(content)

            # Split if over max_tokens (except lists -- Rule 3)
            if chunk_type != "list" and token_count > self.max_tokens:
                # Split into sub-chunks
                words = content.split()
                chunk_words: List[str] = []
                for word in words:
                    chunk_words.append(word)
                    if _estimate_tokens(" ".join(chunk_words)) >= self.max_tokens:
                        sub_content = " ".join(chunk_words)
                        sub_ldu = LDU(
                            ldu_id=f"ldu_{uuid.uuid4().hex[:12]}",
                            content=sub_content,
                            chunk_type="text",
                            page_refs=list(set(buffer_pages)),
                            bounding_box=buffer_bbox,
                            parent_section=section,
                            doc_id=doc.doc_id,
                            document_name=doc.filename,
                            token_count=_estimate_tokens(sub_content),
                        )
                        sub_ldu.content_hash = LDU.compute_hash(sub_content)
                        ldus.append(self.validator.validate_all(sub_ldu))
                        chunk_words = []

                if chunk_words:
                    sub_content = " ".join(chunk_words)
                    sub_ldu = LDU(
                        ldu_id=f"ldu_{uuid.uuid4().hex[:12]}",
                        content=sub_content,
                        chunk_type="text",
                        page_refs=list(set(buffer_pages)),
                        bounding_box=buffer_bbox,
                        parent_section=section,
                        doc_id=doc.doc_id,
                        document_name=doc.filename,
                        token_count=_estimate_tokens(sub_content),
                    )
                    sub_ldu.content_hash = LDU.compute_hash(sub_content)
                    ldus.append(self.validator.validate_all(sub_ldu))
            else:
                ldu = LDU(
                    ldu_id=f"ldu_{uuid.uuid4().hex[:12]}",
                    content=content,
                    chunk_type=chunk_type,
                    page_refs=list(set(buffer_pages)),
                    bounding_box=buffer_bbox,
                    parent_section=section,
                    doc_id=doc.doc_id,
                    document_name=doc.filename,
                    token_count=token_count,
                )
                ldu.content_hash = LDU.compute_hash(content)
                ldus.append(self.validator.validate_all(ldu))

            buffer_texts.clear()
            buffer_pages.clear()

        list_pattern = re.compile(r"^\s*(\d+[\.\)]\s|\u2022\s|\-\s|\*\s)")

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            block_is_list = bool(list_pattern.match(text))

            # Flush on type change
            if buffer_texts and block_is_list != is_list:
                flush_buffer("list" if is_list else "text")
                is_list = block_is_list

            buffer_texts.append(text)
            if block.page >= 0:
                buffer_pages.append(block.page)
            if buffer_bbox is None:
                buffer_bbox = _to_bbox_ref(block.bbox)

        flush_buffer("list" if is_list else "text")
        return ldus

    def _make_table_ldu(self, table: ExtractedTable, section: Optional[str], doc: ExtractedDocument) -> LDU:
        """Rule 1: entire table (headers + rows) in a single LDU."""
        content = table.to_markdown()
        token_count = _estimate_tokens(content)
        ldu = LDU(
            ldu_id=f"ldu_{uuid.uuid4().hex[:12]}",
            content=content,
            chunk_type="table",
            page_refs=[table.page] if table.page >= 0 else [],
            bounding_box=_to_bbox_ref(table.bbox, table.page),
            parent_section=section,
            doc_id=doc.doc_id,
            document_name=doc.filename,
            token_count=token_count,
            metadata={
                "table_id": table.table_id,
                "headers": table.headers,
                "rows_count": len(table.rows),
                "caption": table.caption or "",
            },
        )
        ldu.content_hash = LDU.compute_hash(content)
        return self.validator.validate_all(ldu, source_table=table)

    def _make_figure_ldu(self, fig, section: Optional[str], doc: ExtractedDocument) -> LDU:
        """Rule 2: figure caption stored as metadata."""
        content = f"[Figure {fig.figure_id}]" + (f": {fig.caption}" if fig.caption else "")
        ldu = LDU(
            ldu_id=f"ldu_{uuid.uuid4().hex[:12]}",
            content=content,
            chunk_type="figure",
            page_refs=[fig.page] if fig.page >= 0 else [],
            bounding_box=_to_bbox_ref(fig.bbox, fig.page),
            parent_section=section,
            doc_id=doc.doc_id,
            document_name=doc.filename,
            token_count=_estimate_tokens(content),
            metadata={"caption": fig.caption or "", "figure_id": fig.figure_id},
        )
        ldu.content_hash = LDU.compute_hash(content)
        return self.validator.validate_all(ldu)

    def _resolve_cross_references(self, ldus: List[LDU]) -> List[LDU]:
        """
        Rule 5: Detect cross-references like 'see Table 3', 'Figure 2' and
        link the referring LDU to the referenced LDU's ldu_id.
        """
        # Build index: table_id/figure_id → ldu_id
        ref_index: dict[str, str] = {}
        for ldu in ldus:
            if ldu.chunk_type in ("table", "figure"):
                tid = ldu.metadata.get("table_id") or ldu.metadata.get("figure_id", "")
                if tid:
                    ref_index[tid.lower()] = ldu.ldu_id

        # Pattern to detect cross-references
        xref_pattern = re.compile(r"\b(table|figure|fig\.?|see)\s+(\d+)", re.IGNORECASE)

        for ldu in ldus:
            matches = xref_pattern.findall(ldu.content)
            for ref_type, ref_num in matches:
                # Try to find matching LDU
                for key, ref_ldu_id in ref_index.items():
                    if ref_num in key and ref_ldu_id not in ldu.cross_references:
                        ldu.cross_references.append(ref_ldu_id)
        return ldus
