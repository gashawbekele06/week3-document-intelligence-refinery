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
    def compute_hash(
        cls,
        content: str,
        page_refs: Optional[List[int]] = None,
        bbox: Optional[BBoxRef] = None,
    ) -> str:
        """
        Deterministic SHA-256 of normalized content + spatial anchors.
        Mirrors Week 1 spatial hashing by mixing content with page and bbox when available.
        """
        normalized = " ".join(content.split())
        spatial_parts: List[str] = []
        if page_refs:
            spatial_parts.append("pages=" + ",".join(str(p) for p in sorted(page_refs)))
        if bbox:
            spatial_parts.append(
                "bbox="
                + ",".join(
                    f"{v:.2f}" for v in [bbox.x0, bbox.y0, bbox.x1, bbox.y1, float(bbox.page)]
                )
            )
        payload = normalized + ("|" + "|".join(spatial_parts) if spatial_parts else "")
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash and self.content:
            self.content_hash = LDU.compute_hash(
                self.content,
                page_refs=self.page_refs,
                bbox=self.bounding_box,
            )
