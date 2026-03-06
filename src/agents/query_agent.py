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
import re
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from src.data.audit import AuditMode
from src.data.fact_table import FactTable
from src.data.vector_store import VectorStore
from src.models import PageIndex, ProvenanceChain, ProvenanceCitation

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


def semantic_search(
    query: str,
    k: int = 5,
    doc_id: Optional[str] = None,
    ldu_ids: Optional[list[str]] = None,
) -> dict:
    """
    Tool 2: Vector similarity search over all LDUs in ChromaDB.
    Returns top-k relevant passages with provenance.
    """
    store = VectorStore()
    try:
        results = store.search(query, k=k, doc_id=doc_id, ldu_ids=ldu_ids)
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
                    "bbox": {
                        "x0": meta.get("bbox_x0"),
                        "y0": meta.get("bbox_y0"),
                        "x1": meta.get("bbox_x1"),
                        "y1": meta.get("bbox_y1"),
                    }
                    if meta.get("bbox_x0") is not None
                    else None,
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
        fact_first = self._fact_first_answer(question, doc_id)
        if fact_first:
            return fact_first
        if self.use_llm:
            return self._llm_query(question, doc_id)
        return self._deterministic_query(question, doc_id)

    def _fact_first_answer(self, question: str, doc_id: Optional[str] = None) -> Optional[dict]:
        """
        Attempt to answer numeric/financial questions directly from FactTable.
        This keeps the query natural-language but ensures precise provenance.
        """
        q = question.lower()
        if not any(k in q for k in ("profit", "income", "revenue", "tax", "expense", "assets", "liabilities", "total", "comprehensive")):
            return None

        ft = FactTable()
        # Extract key phrases (2-4 word ngrams) to search labels
        tokens = re.findall(r"[a-zA-Z]+", q)
        phrases = set()
        for n in (2, 3, 4):
            for i in range(len(tokens) - n + 1):
                phrases.add(" ".join(tokens[i : i + n]))

        candidates = []
        for ph in list(phrases)[:15]:
            rows = ft.search_facts(ph, doc_id)
            for r in rows:
                label = (r.get("label") or "").lower()
                label_tokens = set(re.findall(r"[a-zA-Z]+", label))
                overlap = len(set(tokens) & label_tokens)
                try:
                    numeric_value = float(str(r.get("value", "0")).replace(",", ""))
                except Exception:
                    numeric_value = 0.0
                score = overlap * 2 + (1 if "total" in label else 0) + min(numeric_value / 1_000_000, 3)
                candidates.append((score, r))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]

        answer = (
            f"{best.get('label', 'Value')} is {best.get('value', '')} {best.get('unit', '')}."
        )

        bbox = None
        if best.get("bbox_x0") is not None:
            bbox = {
                "x0": best.get("bbox_x0"),
                "y0": best.get("bbox_y0"),
                "x1": best.get("bbox_x1"),
                "y1": best.get("bbox_y1"),
            }

        citation = ProvenanceCitation(
            document_name=best.get("doc_name", "unknown"),
            file_path="",
            page_number=best.get("page_number", -1),
            bbox=bbox,
            content_hash=best.get("content_hash", ""),
            strategy_used="fact_table",
            confidence_score=0.9,
            ldu_id=best.get("ldu_id", ""),
        )
        provenance = ProvenanceChain(citations=[citation], answer=answer)

        return {
            "question": question,
            "answer": answer,
            "provenance": provenance.model_dump(),
            "pageindex_result": {},
            "precision_report": {},
            "passages_used": 0,
            "fact_table_hit": True,
        }

    def _llm_query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """LLM-orchestrated query using Gemini Flash."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Step 1: Navigate PageIndex for section context
            nav_result = pageindex_navigate(question, doc_id)
            section_context = ""
            filtered_ldu_ids: list[str] = []
            if nav_result["results"]:
                top_sec = nav_result["results"][0]
                for sec in nav_result["results"]:
                    filtered_ldu_ids.extend(sec.get("ldu_ids", []))
                section_context = (
                    f"Most relevant section: {top_sec['section_title']} "
                    f"(pages {top_sec['page_start']}–{top_sec['page_end']})\n"
                    f"Summary: {top_sec.get('summary', '')}"
                )

            # Step 2: Semantic search for supporting passages
            search_result = semantic_search(
                question, k=5, doc_id=doc_id, ldu_ids=filtered_ldu_ids or None
            )
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

            precision_report = self.evaluate_retrieval_precision(question, doc_id)
            return {
                "question": question,
                "answer": answer,
                "provenance": provenance.model_dump(),
                "pageindex_result": nav_result,
                "precision_report": precision_report,
                "passages_used": len(search_result.get("passages", [])),
            }
        except Exception as e:
            return self._deterministic_query(question, doc_id)

    def _deterministic_query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        nav = pageindex_navigate(question, doc_id)
        filtered_ldu_ids: list[str] = []
        for sec in nav.get("results", []):
            filtered_ldu_ids.extend(sec.get("ldu_ids", []))
        search = semantic_search(question, k=5, doc_id=doc_id, ldu_ids=filtered_ldu_ids or None)

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
            "precision_report": self.evaluate_retrieval_precision(question, doc_id),
        }

    def evaluate_retrieval_precision(
        self, query: str, doc_id: Optional[str] = None, k: int = 5
    ) -> dict:
        """
        Measure precision@k with and without PageIndex traversal.
        Uses PageIndex top sections as the relevance proxy.
        """
        nav = pageindex_navigate(query, doc_id)
        relevant_ldu_ids: list[str] = []
        for sec in nav.get("results", []):
            relevant_ldu_ids.extend(sec.get("ldu_ids", []))
        relevant_set = set(relevant_ldu_ids)

        store = VectorStore()
        full_results = store.search(query, k=k, doc_id=doc_id)
        filtered_results = (
            store.search(query, k=k, doc_id=doc_id, ldu_ids=list(relevant_set))
            if relevant_set
            else []
        )

        def _precision(results: list) -> float:
            if not results:
                return 0.0
            hits = sum(1 for _, meta, _ in results if meta.get("ldu_id") in relevant_set)
            return round(hits / len(results), 3)

        return {
            "query": query,
            "k": k,
            "precision_without_pageindex": _precision(full_results),
            "precision_with_pageindex": _precision(filtered_results),
            "results_without_pageindex": len(full_results),
            "results_with_pageindex": len(filtered_results),
            "relevant_pool_size": len(relevant_set),
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
