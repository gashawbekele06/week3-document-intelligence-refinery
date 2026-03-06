import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import query_agent as qa_module
from src.agents.query_agent import QueryAgent


def test_query_plan_numeric_prefers_structured_tools():
    agent = QueryAgent(use_llm=False)
    plan = agent._build_query_plan("What is the total comprehensive income for FY 2024?", "doc-1")

    assert plan["intent"] == "numeric"
    assert plan["prefer_fact_table"] is True
    assert plan["tools"] == ["pageindex_navigate", "structured_query", "semantic_search"]


def test_numeric_query_uses_structured_query_tool(monkeypatch):
    calls = []

    def fake_structured_query(sql: str):
        calls.append(sql)
        return {
            "tool": "structured_query",
            "rows": [
                {
                    "doc_name": "sample.pdf",
                    "label": "total comprehensive income",
                    "value": "12345",
                    "unit": "ETB",
                    "page_number": 7,
                    "content_hash": "sha256:test",
                    "ldu_id": "ldu-1",
                    "bbox_x0": None,
                    "bbox_y0": None,
                    "bbox_x1": None,
                    "bbox_y1": None,
                }
            ],
            "count": 1,
        }

    monkeypatch.setattr(qa_module, "structured_query", fake_structured_query)

    agent = QueryAgent(use_llm=False)
    result = agent.query("What is the total comprehensive income for FY 2024?", "doc-1")

    assert calls, "structured_query tool was not invoked"
    assert result["query_plan"]["intent"] == "numeric"
    assert "structured_query" in result["tool_calls"]
    assert "12345" in result["answer"]
    assert result["fact_table_hit"] is True
