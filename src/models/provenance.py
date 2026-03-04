from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class ProvenanceCitation(BaseModel):
    """A single source citation for one piece of evidence."""

    model_config = ConfigDict()

    document_name: str
    file_path: str
    page_number: int
    bbox: Optional[dict] = Field(
        default=None, description="Bounding box {x0,y0,x1,y1} in PDF points"
    )
    content_hash: str
    strategy_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    ldu_id: str = ""


class ProvenanceChain(BaseModel):
    """
    Full audit trail for an answer — a list of source citations.
    Every answer from the Query Agent must carry a ProvenanceChain.
    """

    model_config = ConfigDict()

    citations: List[ProvenanceCitation] = Field(default_factory=list)
    answer: str = ""
    verified: Optional[bool] = None  # Set by AuditMode
    audit_note: Optional[str] = None

    def summary(self) -> str:
        """Human-readable provenance summary."""
        if not self.citations:
            return "⚠️  No source citations available."
        lines = [f"📄 {c.document_name}, page {c.page_number}" for c in self.citations]
        return "\n".join(lines)
