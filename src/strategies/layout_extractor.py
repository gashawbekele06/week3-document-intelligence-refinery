"""
Strategy B: Layout-Aware Extractor using Docling (IBM Research).
Handles multi-column, table-heavy and mixed content layouts.
Normalizes Docling's DoclingDocument output to ExtractedDocument schema.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import List

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TableCell,
    TextBlock,
)
from src.strategies.base import BaseExtractor


class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Docling-based layout-aware extraction.
    Falls back gracefully to pdfplumber if Docling is unavailable.
    """

    name = "layout_extractor"

    def __init__(self):
        self._docling_available = self._check_docling()

    def _check_docling(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            return True
        except ImportError:
            return False

    def _extract_with_docling(self, file_path: Path) -> ExtractedDocument:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        start = time.time()
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document

        text_blocks: List[TextBlock] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        section_headings: List[TextBlock] = []

        # ── Extract text blocks from DoclingDocument ──────────────────────
        try:
            for item, level in doc.iterate_items():
                from docling.datamodel.document import TextItem, TableItem, PictureItem, SectionHeaderItem
                page_num = 1
                bbox_obj = None
                if hasattr(item, "prov") and item.prov:
                    prov = item.prov[0]
                    page_num = getattr(prov, "page_no", 1)
                    bb = getattr(prov, "bbox", None)
                    if bb:
                        try:
                            bbox_obj = BBox(
                                x0=float(bb.l),
                                y0=float(bb.t),
                                x1=float(bb.r),
                                y1=float(bb.b),
                                page=page_num,
                            )
                        except Exception:
                            pass

                if hasattr(item, "text") and item.text:
                    is_header = type(item).__name__ in ("SectionHeaderItem", "DocTitle")
                    block = TextBlock(
                        text=item.text,
                        bbox=bbox_obj,
                        page=page_num,
                        is_header=is_header,
                    )
                    text_blocks.append(block)
                    if is_header:
                        section_headings.append(block)

                elif type(item).__name__ == "TableItem":
                    try:
                        df = item.export_to_dataframe()
                        headers = list(df.columns.astype(str))
                        rows = [list(row.astype(str)) for _, row in df.iterrows()]
                        cells: List[TableCell] = []
                        for ci, h in enumerate(headers):
                            cells.append(TableCell(value=h, row=0, col=ci, is_header=True))
                        for ri, row in enumerate(rows, start=1):
                            for ci, val in enumerate(row):
                                cells.append(TableCell(value=val, row=ri, col=ci))

                        ext_table = ExtractedTable(
                            table_id=f"t_{page_num}_{len(tables)}",
                            bbox=bbox_obj,
                            page=page_num,
                            headers=headers,
                            rows=rows,
                            cells=cells,
                        )
                        tables.append(ext_table)
                    except Exception:
                        pass

                elif type(item).__name__ == "PictureItem":
                    fig = ExtractedFigure(
                        figure_id=f"f_{page_num}_{len(figures)}",
                        bbox=bbox_obj,
                        page=page_num,
                    )
                    figures.append(fig)
        except Exception as e:
            # Fallback: export markdown and treat as single text block
            try:
                md = doc.export_to_markdown()
                text_blocks.append(TextBlock(text=md, page=1))
            except Exception:
                pass

        page_count = max(
            (b.page for b in text_blocks if b.page > 0), default=1
        )
        full_text = "\n\n".join(b.text for b in text_blocks)

        result_doc = ExtractedDocument(
            doc_id=file_path.stem + "_docling",
            filename=file_path.name,
            strategy_used="layout_extractor",
            confidence_score=0.0,
            page_count=max(page_count, 1),
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text=full_text,
            section_headings=section_headings,
            processing_time_sec=round(time.time() - start, 2),
        )
        result_doc.confidence_score = self.confidence(result_doc)
        return result_doc

    def _extract_with_plumber_fallback(self, file_path: Path) -> ExtractedDocument:
        """Fallback to pdfplumber with layout heuristics when Docling unavailable."""
        from src.strategies.fast_text import FastTextExtractor
        doc = FastTextExtractor().extract(file_path)
        doc.strategy_used = "layout_extractor"
        return doc

    def extract(self, file_path: Path) -> ExtractedDocument:
        if self._docling_available:
            try:
                return self._extract_with_docling(file_path)
            except Exception as e:
                pass
        return self._extract_with_plumber_fallback(file_path)

    def confidence(self, doc: ExtractedDocument) -> float:
        """Layout extractor confidence based on structured content ratio."""
        if doc.page_count == 0:
            return 0.0
        has_text = len(doc.text_blocks) > 0
        total_chars = sum(len(b.text) for b in doc.text_blocks)
        text_score = min(total_chars / (doc.page_count * 200), 1.0)
        table_score = min(len(doc.tables) * 0.1, 0.3)
        return round(min(0.5 * text_score + 0.3 * table_score + 0.2 * int(has_text), 1.0), 3)
