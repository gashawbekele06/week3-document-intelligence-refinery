import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.chunker import ChunkingEngine
from src.models import ExtractedDocument, TextBlock, BBox, ExtractedTable, TableCell, ExtractedFigure


def _make_long_text(length: int) -> str:
    return "word " * length


def test_chunking_rules_enforced():
    # Heading
    heading = TextBlock(text="1. Executive Summary", page=1, is_header=True)

    # Long paragraph to force split (> max_tokens)
    long_text = TextBlock(text=_make_long_text(1500), page=1, is_header=False)

    # List bullets that should stay together
    list_block1 = TextBlock(text="1. First bullet", page=2, is_header=False)
    list_block2 = TextBlock(text="2. Second bullet", page=2, is_header=False)

    # Table
    table = ExtractedTable(
        table_id="t_1_0",
        bbox=BBox(x0=0, y0=0, x1=100, y1=100, page=2),
        page=2,
        headers=["A", "B"],
        rows=[["1", "2"], ["3", "4"]],
        cells=[
            TableCell(value="A", row=0, col=0, is_header=True),
            TableCell(value="B", row=0, col=1, is_header=True),
            TableCell(value="1", row=1, col=0),
            TableCell(value="2", row=1, col=1),
        ],
        caption="Test table",
    )

    # Figure with caption
    fig = ExtractedFigure(
        figure_id="f_1_0",
        bbox=BBox(x0=0, y0=0, x1=50, y1=50, page=3),
        page=3,
        caption="Test figure",
    )

    doc = ExtractedDocument(
        doc_id="doc1",
        filename="sample.pdf",
        strategy_used="fast_text",
        confidence_score=1.0,
        page_count=3,
        text_blocks=[heading, long_text, list_block1, list_block2],
        tables=[table],
        figures=[fig],
        full_text="",
        section_headings=[heading],
    )

    engine = ChunkingEngine()
    ldus = engine.chunk(doc)

    # Heading present
    heading_ldus = [l for l in ldus if l.chunk_type == "heading"]
    assert heading_ldus, "Heading LDU missing"

    # Long text should split into multiple chunks that respect min/max tokens
    text_ldus = [l for l in ldus if l.chunk_type == "text"]
    assert len(text_ldus) >= 2, "Long text not split"
    for l in text_ldus:
        assert l.token_count <= engine.max_tokens + 10  # allow small overage due to estimate
        assert l.token_count >= engine.min_tokens_for_split or len(text_ldus) == 1
        assert l.parent_section is not None

    # List blocks stay together
    list_ldus = [l for l in ldus if l.chunk_type == "list"]
    assert len(list_ldus) == 1
    assert len(list_ldus[0].page_refs) >= 1

    # Table rule: headers present and metadata captured
    table_ldus = [l for l in ldus if l.chunk_type == "table"]
    assert table_ldus and "headers" in table_ldus[0].metadata

    # Figure rule: caption stored in metadata
    fig_ldus = [l for l in ldus if l.chunk_type == "figure"]
    assert fig_ldus and "caption" in fig_ldus[0].metadata

    # Cross references may be empty but should not error
    for l in ldus:
        assert l.content_hash, "Content hash missing"
