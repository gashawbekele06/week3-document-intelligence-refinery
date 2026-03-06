import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.models import ExtractedDocument, TextBlock


def test_pageindex_builds_nested_sections_with_valid_ranges():
    headings = [
        TextBlock(text="1. Introduction", page=1, is_header=True, reading_order=0),
        TextBlock(text="1.1. Background", page=1, is_header=True, reading_order=1),
        TextBlock(text="2. Findings", page=2, is_header=True, reading_order=0),
    ]
    body_blocks = [
        TextBlock(text="Intro overview paragraph.", page=1, reading_order=2),
        TextBlock(text="Background details paragraph.", page=1, reading_order=3),
        TextBlock(text="Findings paragraph with ETB 1000.", page=2, reading_order=1),
    ]

    doc = ExtractedDocument(
        doc_id="doc-index",
        filename="sample.pdf",
        strategy_used="fast_text",
        confidence_score=1.0,
        page_count=2,
        text_blocks=[*headings, *body_blocks],
        section_headings=headings,
        full_text=" ".join(b.text for b in body_blocks),
    )

    ldus = ChunkingEngine().chunk(doc)
    page_index = PageIndexBuilder(use_llm=False).build(doc, ldus)

    assert len(page_index.sections) == 2
    assert page_index.sections[0].title == "1. Introduction"
    assert page_index.sections[0].child_sections
    assert page_index.sections[0].child_sections[0].title == "1.1. Background"
    assert page_index.sections[0].page_start == 1
    assert page_index.sections[0].page_end == 1
    assert page_index.sections[1].title == "2. Findings"
    assert page_index.sections[1].page_start == 2
    assert page_index.sections[1].page_end == 2
    assert page_index.sections[0].ldu_ids
    assert page_index.sections[1].ldu_ids
