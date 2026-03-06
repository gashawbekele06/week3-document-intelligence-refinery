"""
Vector Store: ChromaDB-based LDU ingestion and semantic search.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib
import math
import re

from src.models import LDU, ProvenanceChain, ProvenanceCitation

_CHROMA_DIR = Path(".refinery") / "chromadb"
_COLLECTION_NAME = "document_ldus_hash"
_EMBED_DIM = 384  # lightweight, deterministic hash embedding size
_DATA_DIR = Path("data")


class SimpleHashEmbeddingFunction:
    """
    Offline-friendly embedding function to avoid network downloads.
    Uses a stable hashing trick into a fixed-size vector.
    """

    def __init__(self, dim: int = _EMBED_DIM):
        self.dim = dim

    def name(self) -> str:
        return "simple_hash_v1"

    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        return [self._embed_text(t) for t in texts]

    def embed_query(self, input):
        return self.__call__(input)

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class VectorStore:
    """
    ChromaDB-backed vector store for LDU semantic search.
    Uses Google's embedding model or sentence-transformers as fallback.
    """

    def __init__(self, collection_name: str = _COLLECTION_NAME):
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embed_fn = None

    def _init_client(self):
        if self._client is None:
            import chromadb
            _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=SimpleHashEmbeddingFunction(),
            )

    def ingest(self, ldus: List[LDU]) -> int:
        """Ingest LDUs into the vector store. Returns count ingested."""
        self._init_client()
        if not ldus:
            return 0

        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_metas: List[dict] = []

        for ldu in ldus:
            if not ldu.content.strip():
                continue
            batch_ids.append(ldu.ldu_id)
            batch_docs.append(ldu.content)
            bbox = ldu.bounding_box
            batch_metas.append(
                {
                    "doc_id": ldu.doc_id,
                    "document_name": ldu.document_name,
                    "chunk_type": ldu.chunk_type,
                    "page_refs": ",".join(str(p) for p in ldu.page_refs),
                    "parent_section": ldu.parent_section or "",
                    "content_hash": ldu.content_hash,
                    "token_count": ldu.token_count,
                    "ldu_id": ldu.ldu_id,
                    "bbox_x0": float(bbox.x0) if bbox else None,
                    "bbox_y0": float(bbox.y0) if bbox else None,
                    "bbox_x1": float(bbox.x1) if bbox else None,
                    "bbox_y1": float(bbox.y1) if bbox else None,
                    "bbox_page": int(bbox.page) if bbox else None,
                }
            )

        # Upsert in batches of 100
        for i in range(0, len(batch_ids), 100):
            self._collection.upsert(
                ids=batch_ids[i : i + 100],
                documents=batch_docs[i : i + 100],
                metadatas=batch_metas[i : i + 100],
            )
        return len(batch_ids)

    def search(
        self,
        query: str,
        k: int = 5,
        doc_id: Optional[str] = None,
        ldu_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, dict, float]]:
        """
        Search for top-k similar LDUs.
        Returns list of (content, metadata, distance) tuples.
        """
        self._init_client()
        where = None
        if doc_id and ldu_ids:
            where = {"$and": [{"doc_id": doc_id}, {"ldu_id": {"$in": ldu_ids}}]}
        elif doc_id:
            where = {"doc_id": doc_id}
        elif ldu_ids:
            where = {"ldu_id": {"$in": ldu_ids}}
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, 10),
            where=where,
        )
        output = []
        if results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for doc, meta, dist in zip(docs, metas, dists):
                output.append((doc, meta, dist))
        return output

    def build_provenance(
        self, results: List[Tuple[str, dict, float]], strategy_used: str = "semantic_search"
    ) -> ProvenanceChain:
        """Convert search results into a ProvenanceChain."""
        citations = []
        for content, meta, dist in results:
            pages_raw = meta.get("page_refs", "")
            pages = [int(p) for p in pages_raw.split(",") if p.strip().isdigit()]
            document_name = meta.get("document_name", "unknown")
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
                    document_name=document_name,
                    file_path=str(self._resolve_document_path(document_name) or ""),
                    page_number=pages[0] if pages else -1,
                    bbox=bbox,
                    content_hash=meta.get("content_hash", ""),
                    strategy_used=strategy_used,
                    confidence_score=round(1.0 - dist, 3),
                    ldu_id=meta.get("ldu_id", ""),
                )
            )
        return ProvenanceChain(citations=citations)

    def _resolve_document_path(self, document_name: str) -> Optional[Path]:
        direct = _DATA_DIR / document_name
        if direct.exists():
            return direct
        matches = list(_DATA_DIR.rglob(document_name))
        return matches[0] if matches else None
