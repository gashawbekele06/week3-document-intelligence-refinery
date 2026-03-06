"""
AuditMode: claim verification against the refinery's knowledge store.
Given a claim, verifies it against FactTable + VectorStore and returns citations or "unverifiable".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.storage.fact_table import FactTable
from src.storage.vector_store import VectorStore
from src.models import ProvenanceChain, ProvenanceCitation


class AuditMode:
    """
    Given a claim like "The report states revenue was $4.2B in Q3",
    verifies it against:
    1. FactTable (SQL search for the number/label)
    2. VectorStore (semantic search for supporting text)
    Returns a ProvenanceChain with verified=True/False and audit_note.
    """

    def __init__(self):
        self.fact_table = FactTable()
        self.vector_store = VectorStore()
        self.data_dir = Path("data")

    def _resolve_document_path(self, document_name: str) -> str:
        direct = self.data_dir / document_name
        if direct.exists():
            return str(direct)
        matches = list(self.data_dir.rglob(document_name))
        return str(matches[0]) if matches else ""

    def verify(self, claim: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """
        Attempt to verify a claim.
        Returns ProvenanceChain with:
        - verified=True + citations if found
        - verified=False + audit_note if not found
        """
        # Step 1: Extract key terms from claim
        numbers = re.findall(r"[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|%))?", claim)
        keywords = re.findall(r"\b[A-Za-z][\w\s]{3,30}\b", claim)

        citations = []
        found_in_facts = False

        # Step 2: Search FactTable for numerics
        for num in numbers[:3]:
            clean_num = re.sub(r"[^\d.]", "", num.split()[0]) if num else ""
            if not clean_num:
                continue
            facts = self.fact_table.query(
                f"SELECT * FROM facts WHERE value LIKE '%{clean_num}%'"
                + (f" AND doc_id = '{doc_id}'" if doc_id else "")
                + " LIMIT 5"
            )
            for f in facts:
                bbox = None
                if f.get("bbox_x0") is not None:
                    bbox = {
                        "x0": f.get("bbox_x0"),
                        "y0": f.get("bbox_y0"),
                        "x1": f.get("bbox_x1"),
                        "y1": f.get("bbox_y1"),
                    }
                citations.append(
                    ProvenanceCitation(
                        document_name=f.get("doc_name", "unknown"),
                        file_path=self._resolve_document_path(f.get("doc_name", "unknown")),
                        page_number=f.get("page_number", -1),
                        bbox=bbox,
                        content_hash=f.get("content_hash", ""),
                        strategy_used="fact_table",
                        confidence_score=0.85,
                        ldu_id=f.get("ldu_id", ""),
                    )
                )
                found_in_facts = True

        # Step 3: Semantic search for supporting context
        semantic_results = self.vector_store.search(claim, k=3, doc_id=doc_id)
        for content, meta, dist in semantic_results:
            if dist < 0.5:  # close semantic match
                pages_raw = meta.get("page_refs", "")
                pages = [int(p) for p in pages_raw.split(",") if p.strip().isdigit()]
                bbox = None
                if meta.get("bbox_x0") is not None:
                    bbox = {
                        "x0": meta.get("bbox_x0"),
                        "y0": meta.get("bbox_y0"),
                        "x1": meta.get("bbox_x1"),
                        "y1": meta.get("bbox_y1"),
                    }
                citations.append(
                    ProvenanceCitation(
                        document_name=meta.get("document_name", "unknown"),
                        file_path=self._resolve_document_path(meta.get("document_name", "unknown")),
                        page_number=pages[0] if pages else -1,
                        bbox=bbox,
                        content_hash=meta.get("content_hash", ""),
                        strategy_used="semantic_search",
                        confidence_score=round(1.0 - dist, 3),
                        ldu_id=meta.get("ldu_id", ""),
                    )
                )

        if citations:
            return ProvenanceChain(
                citations=citations,
                answer=f"Claim verified. Found {len(citations)} supporting source(s).",
                verified=True,
                audit_note="Verified via fact table and/or semantic search.",
            )
        else:
            return ProvenanceChain(
                citations=[],
                answer="",
                verified=False,
                audit_note=(
                    f"Claim '{claim[:100]}...' could not be verified — "
                    "no matching facts or semantically similar passages found."
                ),
            )
