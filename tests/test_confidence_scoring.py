import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import ExtractedDocument, TextBlock, ExtractedFigure
from src.strategies.fast_text import FastTextExtractor


def test_fast_text_confidence_high_with_dense_text_and_fonts():
    extractor = FastTextExtractor()
    doc = ExtractedDocument(
        doc_id="doc1",
        filename="file.pdf",
        strategy_used="fast_text",
        confidence_score=0.0,
        page_count=1,
        text_blocks=[
            TextBlock(text="a" * 1200, page=1, font_name="Times-Roman", font_size=12)
        ],
        tables=[],
        figures=[],
        full_text="",
        section_headings=[],
    )

    score = extractor.confidence(doc)
    assert 0.9 <= score <= 1.0


def test_fast_text_confidence_low_with_sparse_text_and_images():
    extractor = FastTextExtractor()
    doc = ExtractedDocument(
        doc_id="doc2",
        filename="file.pdf",
        strategy_used="fast_text",
        confidence_score=0.0,
        page_count=1,
        text_blocks=[TextBlock(text="short", page=1)],
        tables=[],
        figures=[ExtractedFigure(figure_id="f1", page=1)],
        full_text="",
        section_headings=[],
    )

    score = extractor.confidence(doc)
    assert score < 0.2