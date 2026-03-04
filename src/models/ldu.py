from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field
import hashlib


class BBoxRef(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class LDU(BaseModel):
    """
    Logical Document Unit — the fundamental RAG-ready chunk.
    Every LDU is a semantically coherent, self-contained unit that preserves
    structural context (section header, table integrity, figure caption).
    """

    model_config = ConfigDict()

    ldu_id: str
    content: str
    chunk_type: Literal["text", "table", "figure", "list", "heading", "equation"]

    # ── Provenance ──────────────────────────────────────────────────────────
    page_refs: List[int] = Field(default_factory=list)
    bounding_box: Optional[BBoxRef] = None
    parent_section: Optional[str] = None
    doc_id: str = ""
    document_name: str = ""

    # ── Size & Identity ─────────────────────────────────────────────────────
    token_count: int = 0
    content_hash: str = ""  # SHA-256 of normalized content

    # ── Relationships ───────────────────────────────────────────────────────
    cross_references: List[str] = Field(
        default_factory=list,
        description="LDU IDs of related chunks (e.g., 'see Table 3')",
    )

    # ── Extra Metadata ──────────────────────────────────────────────────────
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def compute_hash(cls, content: str) -> str:
        """Deterministic SHA-256 of normalized content."""
        normalized = " ".join(content.split())
        return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash and self.content:
            self.content_hash = LDU.compute_hash(self.content)
