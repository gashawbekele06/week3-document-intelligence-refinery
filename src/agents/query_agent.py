"""
Stage 5: Query Interface Agent
LangGraph-based agent with 3 tools:
1. pageindex_navigate  — tree traversal for section-aware retrieval
2. semantic_search     — vector similarity search
3. structured_query    — SQL over FactTable

Every answer includes a ProvenanceChain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from src.data.audit import AuditMode
from src.data.fact_table import FactTable
from src.data.vector_store import VectorStore
from src.models import PageIndex, ProvenanceChain

_PAGEINDEX_DIR = Path(".refinery") / "pageindex"


class QueryState(TypedDict):
    query: str
    doc_id: Optional[str]
    answer: str
    provenance: Optional[ProvenanceChain]
    tool_calls: list[str]


# ─── Tool Implementations ────────────────────────────────────────────────────

def pageindex_navigate(query: str, doc_id: Optional[str] = None) -> dict:
    """
    Tool 1: Navigate the PageIndex tree to find relevant sections.
    Returns top-3 sections + their LDU IDs for focused retrieval.
    """
    results = []

    index_files = list(_PAGEINDEX_DIR.glob("*.json"))
    if doc_id:
        index_files = [f for f in index_files if f.stem == doc_id]

    for idx_file in index_files:
        try:
            page_index = PageIndex.model_validate_json(idx_file.read_text())
            matching = page_index.find_sections_for_query(query, top_k=3)
            for section in matching:
                results.append(
                    {
                        "document": page_index.document_name,
                        "doc_id": page_index.doc_id,
                        "section_title": section.title,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "summary": section.summary,
                        "key_entities": section.key_entities,
                        "ldu_ids": section.ldu_ids,
                    }
                )
        except Exception:
            continue

    return {
        "tool": "pageindex_navigate",
        "query": query,
        "results": results[:3],
        "message": f"Found {len(results)} sections matching '{query}'",
    }


def semantic_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> dict:
    """
    Tool 2: Vector similarity search over all LDUs in ChromaDB.
    Returns top-k relevant passages with provenance.
    """
    store = VectorStore()
    try:
        results = store.search(query, k=k, doc_id=doc_id)
        provenance = store.build_provenance(results, strategy_used="semantic_search")

        passages = []
        for content, meta, dist in results:
            pages_raw = meta.get("page_refs", "")
            pages = [int(p) for p in pages_raw.split(",") if p.strip().isdigit()]
            passages.append(
                {
                    "content": content[:500],
                    "document": meta.get("document_name", ""),
                    "page": pages[0] if pages else -1,
                    "section": meta.get("parent_section", ""),
                    "chunk_type": meta.get("chunk_type", ""),
                    "similarity": round(1.0 - dist, 3),
                    "content_hash": meta.get("content_hash", ""),
                }
            )

        return {
            "tool": "semantic_search",
            "query": query,
            "passages": passages,
            "provenance": provenance.model_dump(),
        }
    except Exception as e:
        return {
            "tool": "semantic_search",
            "query": query,
            "passages": [],
            "error": str(e),
        }


def structured_query(sql: str) -> dict:
    """
    Tool 3: SQL query over the SQLite FactTable.
    Returns rows with document provenance.
    """
    ft = FactTable()
    try:
        # Safety: only allow SELECT statements
        if not sql.strip().upper().startswith("SELECT"):
            return {"tool": "structured_query", "error": "Only SELECT queries allowed.", "rows": []}

        rows = ft.query(sql)
        return {
            "tool": "structured_query",
            "sql": sql,
            "rows": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"tool": "structured_query", "sql": sql, "error": str(e), "rows": []}


# ─── LangGraph Query Agent ────────────────────────────────────────────────────

class QueryAgent:
    """
    LangGraph-based query agent that combines all 3 tools.
    Falls back to simple tool orchestration if LangGraph not available.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(os.getenv("GEMINI_API_KEY"))
        self.audit = AuditMode()

    def query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """
        Answer a natural language question using multi-tool orchestration.
        Returns answer + ProvenanceChain.
        """
        if self.use_llm:
            return self._llm_query(question, doc_id)
        return self._deterministic_query(question, doc_id)

    def _llm_query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """LLM-orchestrated query using Gemini Flash."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Step 1: Navigate PageIndex for section context
            nav_result = pageindex_navigate(question, doc_id)
            section_context = ""
            if nav_result["results"]:
                top_sec = nav_result["results"][0]
                section_context = (
                    f"Most relevant section: {top_sec['section_title']} "
                    f"(pages {top_sec['page_start']}–{top_sec['page_end']})\n"
                    f"Summary: {top_sec.get('summary', '')}"
                )

            # Step 2: Semantic search for supporting passages
            search_result = semantic_search(question, k=5, doc_id=doc_id)
            passages_text = ""
            for p in search_result.get("passages", []):
                passages_text += (
                    f"\n[{p['document']}, p.{p['page']}]: {p['content'][:300]}\n"
                )

            # Step 3: Check FactTable for relevant facts
            # Extract potential number queries
            import re
            keywords = re.findall(r"\b[A-Za-z][\w\s]{4,30}\b", question)
            fact_context = ""
            for kw in keywords[:2]:
                facts = FactTable().search_facts(kw[:20], doc_id)
                if facts:
                    fact_context += f"\nFact: {facts[0]['label']} = {facts[0]['value']} {facts[0].get('unit','')}\n"

            # Step 4: Synthesize answer
            prompt = f"""You are a document intelligence assistant. Answer the following question based ONLY on the provided document excerpts. Include specific page numbers in your answer.

Question: {question}

{section_context}

Document passages:
{passages_text}

{fact_context}

Answer concisely, cite specific page numbers, and note if information is not found in the documents."""

            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 500},
            )
            answer = response.text.strip()

            # Build provenance
            provenance_data = search_result.get("provenance", {})
            provenance = ProvenanceChain.model_validate(provenance_data) if provenance_data else ProvenanceChain()
            provenance.answer = answer

            return {
                "question": question,
                "answer": answer,
                "provenance": provenance.model_dump(),
                "pageindex_result": nav_result,
                "passages_used": len(search_result.get("passages", [])),
            }
        except Exception as e:
            return self._deterministic_query(question, doc_id)

    def _deterministic_query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        nav = pageindex_navigate(question, doc_id)
        search = semantic_search(question, k=5, doc_id=doc_id)

        answer_parts = []
        if nav["results"]:
            top = nav["results"][0]
            answer_parts.append(
                f"In section '{top['section_title']}' (pages {top['page_start']}–{top['page_end']}): "
                f"{top.get('summary', 'See original document')}"
            )

        for p in search.get("passages", [])[:2]:
            if p["similarity"] > 0.4:
                answer_parts.append(
                    f"From {p['document']}, page {p['page']}: {p['content'][:200]}"
                )

        answer = "\n\n".join(answer_parts) or "No relevant information found."

        provenance_data = search.get("provenance", {})
        provenance = ProvenanceChain.model_validate(provenance_data) if provenance_data else ProvenanceChain()
        provenance.answer = answer

        return {
            "question": question,
            "answer": answer,
            "provenance": provenance.model_dump(),
            "pageindex_result": nav,
        }

    def verify_claim(self, claim: str, doc_id: Optional[str] = None) -> dict:
        """Audit mode: verify or refute a claim."""
        chain = self.audit.verify(claim, doc_id)
        return {
            "claim": claim,
            "verified": chain.verified,
            "audit_note": chain.audit_note,
            "citations": [c.model_dump() for c in chain.citations],
        }
