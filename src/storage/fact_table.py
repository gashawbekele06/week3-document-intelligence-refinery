"""
FactTable: SQLite-backed key-value fact extractor for financial documents.
Extracts structured monetary and numerical facts with provenance.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from src.models import LDU

_DB_PATH = Path(".refinery") / "fact_table.db"

_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id      TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    doc_name     TEXT NOT NULL,
    label        TEXT NOT NULL,
    value        TEXT NOT NULL,
    unit         TEXT,
    period       TEXT,
    page_number  INTEGER,
    ldu_id       TEXT,
    content_hash TEXT,
    bbox_x0      REAL,
    bbox_y0      REAL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_label ON facts(label);
"""

# Patterns for financial fact extraction
_MONEY_PATTERN = re.compile(
    r"([\w\s,/\"'{}\[\]]+?)[\s:]+(?:ETB|USD|Birr|\$|€)?\s*([\d,]+(?:\.\d+)?)\s*"
    r"(?:(million|billion|thousand|M|B|K))?\b",
    re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(
    r"([\w\s,/\"'{}\[\]]+?)[\s:]+(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


class FactTable:
    """
    Extracts and stores financial/numerical key-value facts into SQLite.
    Supports precise SQL queries for fact retrieval.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_DDL)
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        """Lightweight migration to ensure bbox columns exist on older DBs."""
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        needed = {
            "bbox_x0": "REAL",
            "bbox_y0": "REAL",
            "bbox_x1": "REAL",
            "bbox_y1": "REAL",
        }
        for col, col_type in needed.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {col_type}")

    def ingest_ldus(self, ldus: List[LDU], doc_name: str) -> int:
        """Extract facts from LDUs and store in SQLite. Returns count inserted."""
        facts = []
        for ldu in ldus:
            facts.extend(self._extract_facts(ldu, doc_name))

        if not facts:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO facts
                   (fact_id, doc_id, doc_name, label, value, unit, period, page_number, ldu_id, content_hash,
                    bbox_x0, bbox_y0, bbox_x1, bbox_y1)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                facts,
            )
        return len(facts)

    def _extract_facts(self, ldu: LDU, doc_name: str) -> list:
        """Extract money/percentage facts from a single LDU."""
        results = []
        page = ldu.page_refs[0] if ldu.page_refs else -1
        bbox = ldu.bounding_box
        import uuid as _uuid

        for match in _MONEY_PATTERN.finditer(ldu.content):
            label = match.group(1).strip().rstrip(":").strip()
            value = match.group(2).replace(",", "")
            unit = match.group(3) or "ETB"
            if len(label) > 200 or len(label) < 3:
                continue
            period = self._extract_period(ldu.content)
            fact_id = _uuid.uuid5(_uuid.NAMESPACE_URL, f"{ldu.doc_id}:{label}:{value}").hex
            results.append((
                fact_id, ldu.doc_id, doc_name, label, value, unit, period,
                page, ldu.ldu_id, ldu.content_hash,
                float(bbox.x0) if bbox else None,
                float(bbox.y0) if bbox else None,
                float(bbox.x1) if bbox else None,
                float(bbox.y1) if bbox else None,
            ))

        for match in _PERCENT_PATTERN.finditer(ldu.content):
            label = match.group(1).strip().rstrip(":").strip()
            value = match.group(2)
            if len(label) > 200 or len(label) < 3:
                continue
            period = self._extract_period(ldu.content)
            fact_id = _uuid.uuid5(_uuid.NAMESPACE_URL, f"{ldu.doc_id}:{label}:{value}%").hex
            results.append((
                fact_id, ldu.doc_id, doc_name, label, f"{value}%", "%", period,
                page, ldu.ldu_id, ldu.content_hash,
                float(bbox.x0) if bbox else None,
                float(bbox.y0) if bbox else None,
                float(bbox.x1) if bbox else None,
                float(bbox.y1) if bbox else None,
            ))
        return results

    def _extract_period(self, text: str) -> str:
        """Extract fiscal period mention from text."""
        m = re.search(r"(Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4}/\d{2,4}|\d{4})", text)
        return m.group(0) if m else ""

    def query(self, sql: str) -> List[dict]:
        """Execute a SQL query against the fact table. Returns list of row dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]

    def search_facts(self, label_query: str, doc_id: Optional[str] = None) -> List[dict]:
        """Full-text search over fact labels."""
        params: list[str] = [f"%{label_query}%"]
        sql = "SELECT * FROM facts WHERE label LIKE ?"
        if doc_id:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " LIMIT 20"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
