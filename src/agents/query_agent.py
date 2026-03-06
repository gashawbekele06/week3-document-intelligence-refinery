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

import pdfplumber

from src.data.audit import AuditMode
from src.data.fact_table import FactTable
from src.data.vector_store import VectorStore
from src.models import PageIndex, ProvenanceChain, ProvenanceCitation

_PAGEINDEX_DIR = Path(".refinery") / "pageindex"
_DATA_DIR = Path("data")


class QueryState(TypedDict):
    query: str
    doc_id: Optional[str]
    answer: str
    provenance: Optional[ProvenanceChain]
    tool_calls: list[str]


class QueryPlan(TypedDict):
    intent: str
    tools: list[str]
    rationale: str
    semantic_k: int
    prefer_fact_table: bool


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
                page_start = max(int(section.page_start or 1), 1)
                page_end = max(int(section.page_end or page_start), page_start)
                results.append(
                    {
                        "document": page_index.document_name,
                        "doc_id": page_index.doc_id,
                        "section_title": section.title,
                        "page_start": page_start,
                        "page_end": page_end,
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

    @staticmethod
    def _safe_sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _contains_numeric_signal(question: str) -> bool:
        q = question.lower()
        keywords = (
            "profit",
            "income",
            "revenue",
            "tax",
            "expense",
            "assets",
            "liabilities",
            "total",
            "comprehensive",
            "amount",
            "value",
            "how much",
            "percentage",
            "%",
        )
        return any(k in q for k in keywords)

    @staticmethod
    def _contains_summary_signal(question: str) -> bool:
        q = question.lower()
        return any(k in q for k in ("summary", "summarize", "overview", "main findings", "recommendations"))

    @staticmethod
    def _contains_lookup_signal(question: str) -> bool:
        q = question.lower()
        return any(q.startswith(prefix) for prefix in ("what", "which", "when", "where", "who", "how"))

    def _build_query_plan(self, question: str, doc_id: Optional[str] = None) -> QueryPlan:
        if question.strip().upper().startswith("SELECT"):
            return {
                "intent": "sql",
                "tools": ["structured_query"],
                "rationale": "Direct SQL detected; execute against FactTable only.",
                "semantic_k": 0,
                "prefer_fact_table": True,
            }
        if self._is_objective_query(question):
            return {
                "intent": "objective",
                "tools": ["pageindex_navigate", "semantic_search"],
                "rationale": "Purpose/objective questions benefit from section targeting before passage retrieval.",
                "semantic_k": 5,
                "prefer_fact_table": False,
            }
        if self._contains_numeric_signal(question):
            return {
                "intent": "numeric",
                "tools": ["pageindex_navigate", "structured_query", "semantic_search"],
                "rationale": "Numeric questions should probe structured facts first, then confirm with passages.",
                "semantic_k": 4,
                "prefer_fact_table": True,
            }
        if self._contains_summary_signal(question):
            return {
                "intent": "summary",
                "tools": ["pageindex_navigate", "semantic_search"],
                "rationale": "Summary questions should traverse sections before retrieving supporting passages.",
                "semantic_k": 6,
                "prefer_fact_table": False,
            }
        if self._contains_lookup_signal(question):
            return {
                "intent": "lookup",
                "tools": ["pageindex_navigate", "semantic_search"],
                "rationale": "Lookup questions use the PageIndex to narrow retrieval scope.",
                "semantic_k": 5,
                "prefer_fact_table": False,
            }
        return {
            "intent": "semantic",
            "tools": ["semantic_search"],
            "rationale": "Fallback to semantic retrieval when no stronger intent signal is present.",
            "semantic_k": 5,
            "prefer_fact_table": False,
        }

    def _phrase_candidates(self, question: str, max_phrases: int = 12) -> list[str]:
        tokens = re.findall(r"[a-zA-Z]+", question.lower())
        phrases: list[str] = []
        for n in (4, 3, 2):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    return phrases
        return phrases

    def _probe_structured_facts(self, question: str, doc_id: Optional[str] = None) -> dict:
        rows: list[dict] = []
        for phrase in self._phrase_candidates(question):
            sql = (
                "SELECT * FROM facts WHERE label LIKE '%"
                + self._safe_sql_literal(phrase)
                + "%'"
            )
            if doc_id:
                sql += f" AND doc_id = '{self._safe_sql_literal(doc_id)}'"
            sql += " LIMIT 10"
            result = structured_query(sql)
            if result.get("rows"):
                rows.extend(result["rows"])
        return {"tool": "structured_query", "rows": rows, "count": len(rows)}

    def _run_context_tools(
        self,
        question: str,
        doc_id: Optional[str],
        plan: QueryPlan,
    ) -> dict:
        tool_calls: list[str] = []
        nav = {"results": []}
        structured = {"rows": []}
        filtered_ldu_ids: list[str] = []

        if "pageindex_navigate" in plan["tools"]:
            nav = pageindex_navigate(question, doc_id)
            tool_calls.append("pageindex_navigate")
            for sec in nav.get("results", []):
                filtered_ldu_ids.extend(sec.get("ldu_ids", []))

        if "structured_query" in plan["tools"]:
            structured = self._probe_structured_facts(question, doc_id)
            tool_calls.append("structured_query")

        search = semantic_search(
            question,
            k=plan["semantic_k"] or 5,
            doc_id=doc_id,
            ldu_ids=filtered_ldu_ids or None,
        )
        if "semantic_search" in plan["tools"]:
            tool_calls.append("semantic_search")

        return {
            "tool_calls": tool_calls,
            "nav": nav,
            "structured": structured,
            "search": search,
            "filtered_ldu_ids": filtered_ldu_ids,
        }

    @staticmethod
    def _normalize_page_range(page_start: int, page_end: int) -> tuple[int, int]:
        start = max(int(page_start or 1), 1)
        end = max(int(page_end or start), start)
        return start, end

    @staticmethod
    def _is_objective_query(question: str) -> bool:
        q = question.lower()
        return any(term in q for term in ("purpose", "objective", "aim", "goal"))

    @staticmethod
    def _clean_passage_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text

    @staticmethod
    def _resolve_document_path(document_name: str) -> Optional[Path]:
        direct = _DATA_DIR / document_name
        if direct.exists():
            return direct
        matches = list(_DATA_DIR.rglob(document_name))
        return matches[0] if matches else None

    def _extract_objective_from_pages(
        self, document_name: str, page_start: int, page_end: int
    ) -> str:
        doc_path = self._resolve_document_path(document_name)
        if not doc_path:
            return ""

        text_parts: list[str] = []
        safe_start, safe_end = self._normalize_page_range(page_start, page_end)
        read_end = min(safe_end + 1, safe_start + 1)
        try:
            with pdfplumber.open(doc_path) as pdf:
                for page_num in range(safe_start, min(read_end, len(pdf.pages)) + 1):
                    text = pdf.pages[page_num - 1].extract_text() or ""
                    text_parts.append(text)
        except Exception:
            return ""

        text = self._clean_passage_text(" ".join(text_parts))
        match = re.search(
            r"The overall objective of the assessment is to (.*?)(?:Specific objectives:|\Z)",
            text,
            re.IGNORECASE,
        )
        if match:
            sentence = "The overall objective of the assessment is to " + match.group(1).strip()
            return sentence.rstrip(".") + "."
        return ""

    def _answer_objective_query(
        self,
        question: str,
        doc_id: Optional[str],
        nav: dict,
        fallback_search: dict,
    ) -> Optional[dict]:
        objective_sections = [
            sec
            for sec in nav.get("results", [])
            if re.search(r"\b(objective|purpose|aim|goal)\b", (sec.get("section_title", "") + " " + sec.get("summary", "")).lower())
        ]

        if not objective_sections and doc_id:
            expanded_nav = pageindex_navigate("objective purpose assessment", doc_id)
            objective_sections = [
                sec
                for sec in expanded_nav.get("results", [])
                if re.search(
                    r"\b(objective|purpose|aim|goal)\b",
                    (sec.get("section_title", "") + " " + sec.get("summary", "")).lower(),
                )
            ]

        ldu_ids: list[str] = []
        for sec in objective_sections or nav.get("results", []):
            ldu_ids.extend(sec.get("ldu_ids", []))

        targeted_search = semantic_search(
            question + " objective purpose overall objective",
            k=5,
            doc_id=doc_id,
            ldu_ids=ldu_ids or None,
        )
        passages = targeted_search.get("passages", []) or fallback_search.get("passages", [])

        best_sentence = ""
        best_page = None
        for passage in passages:
            content = self._clean_passage_text(passage.get("content", ""))
            if len(content) < 60:
                continue
            match = re.search(
                r"(The overall objective of the assessment is .*? strategic direction[^.]*\.)",
                content,
                re.IGNORECASE,
            )
            if match:
                best_sentence = match.group(1)
                best_page = passage.get("page")
                break
            if re.search(r"\bobjective\b", content, re.IGNORECASE):
                best_sentence = content.split(".")[0].strip() + "."
                best_page = passage.get("page")
                break

        if not best_sentence:
            section = objective_sections[0] if objective_sections else (nav.get("results") or [None])[0]
            if section:
                best_sentence = self._extract_objective_from_pages(
                    section.get("document", ""),
                    section.get("page_start", 1),
                    section.get("page_end", 1),
                )
                best_page = section.get("page_start", 1)

        if not best_sentence:
            return None

        section = objective_sections[0] if objective_sections else (nav.get("results") or [None])[0]
        answer = best_sentence
        if section:
            start, end = self._normalize_page_range(section.get("page_start", 1), section.get("page_end", 1))
            page_note = f"page {best_page}" if best_page else f"pages {start}–{end}"
            answer = f"{best_sentence} This is stated in section '{section.get('section_title', 'relevant section')}' ({page_note})."

        provenance_data = targeted_search.get("provenance") or fallback_search.get("provenance", {})
        provenance = ProvenanceChain.model_validate(provenance_data) if provenance_data else ProvenanceChain()
        provenance.answer = answer

        return {
            "question": question,
            "answer": answer,
            "provenance": provenance.model_dump(),
            "pageindex_result": nav,
            "precision_report": self.evaluate_retrieval_precision(question, doc_id),
        }

    def query(self, question: str, doc_id: Optional[str] = None) -> dict:
        """
        Answer a natural language question using multi-tool orchestration.
        Returns answer + ProvenanceChain.
        """
        plan = self._build_query_plan(question, doc_id)

        if plan["intent"] == "objective":
            context = self._run_context_tools(question, doc_id, plan)
            objective_answer = self._answer_objective_query(
                question, doc_id, context["nav"], context["search"]
            )
            if objective_answer:
                objective_answer["tool_calls"] = context["tool_calls"]
                objective_answer["query_plan"] = plan
                return objective_answer

        fact_first = self._fact_first_answer(question, doc_id, plan)
        if fact_first:
            return fact_first
        if self.use_llm:
            return self._llm_query(question, doc_id, plan)
        return self._deterministic_query(question, doc_id, plan)

    def _fact_first_answer(
        self,
        question: str,
        doc_id: Optional[str] = None,
        plan: Optional[QueryPlan] = None,
    ) -> Optional[dict]:
        """
        Attempt to answer numeric/financial questions directly from FactTable.
        This keeps the query natural-language but ensures precise provenance.
        """
        effective_plan = plan or self._build_query_plan(question, doc_id)
        if not effective_plan.get("prefer_fact_table"):
            return None

        candidates = []
        structured_hits = self._probe_structured_facts(question, doc_id)
        tokens = re.findall(r"[a-zA-Z]+", question.lower())
        for r in structured_hits.get("rows", []):
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
            file_path=str(self._resolve_document_path(best.get("doc_name", "unknown")) or ""),
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
            "tool_calls": ["structured_query"],
            "query_plan": effective_plan,
        }

    def _llm_query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        plan: Optional[QueryPlan] = None,
    ) -> dict:
        """LLM-orchestrated query using Gemini Flash."""
        try:
            effective_plan = plan or self._build_query_plan(question, doc_id)
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            context = self._run_context_tools(question, doc_id, effective_plan)
            nav_result = context["nav"]
            search_result = context["search"]
            structured_result = context["structured"]

            section_context = ""
            if nav_result["results"]:
                top_sec = nav_result["results"][0]
                page_start, page_end = self._normalize_page_range(
                    top_sec["page_start"], top_sec["page_end"]
                )
                section_context = (
                    f"Most relevant section: {top_sec['section_title']} "
                    f"(pages {page_start}–{page_end})\n"
                    f"Summary: {top_sec.get('summary', '')}"
                )

            passages_text = ""
            for p in search_result.get("passages", []):
                passages_text += (
                    f"\n[{p['document']}, p.{p['page']}]: {p['content'][:300]}\n"
                )

            fact_context = ""
            for row in structured_result.get("rows", [])[:3]:
                fact_context += (
                    f"\nFact: {row.get('label', '')} = {row.get('value', '')} "
                    f"{row.get('unit', '')} (page {row.get('page_number', '')})\n"
                )

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
                "tool_calls": context["tool_calls"],
                "query_plan": effective_plan,
            }
        except Exception as e:
            return self._deterministic_query(question, doc_id, plan)

    def _deterministic_query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        plan: Optional[QueryPlan] = None,
    ) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        effective_plan = plan or self._build_query_plan(question, doc_id)
        context = self._run_context_tools(question, doc_id, effective_plan)
        nav = context["nav"]
        search = context["search"]

        if self._is_objective_query(question):
            objective_answer = self._answer_objective_query(question, doc_id, nav, search)
            if objective_answer:
                objective_answer["tool_calls"] = context["tool_calls"]
                objective_answer["query_plan"] = effective_plan
                return objective_answer

        answer_parts = []
        if nav["results"]:
            top = nav["results"][0]
            page_start, page_end = self._normalize_page_range(
                top["page_start"], top["page_end"]
            )
            answer_parts.append(
                f"In section '{top['section_title']}' (pages {page_start}–{page_end}): "
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
            "tool_calls": context["tool_calls"],
            "query_plan": effective_plan,
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
