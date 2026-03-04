"""
Strategy A: Fast Text Extractor
Uses pdfplumber for rapid extraction of native-digital PDFs.
Computes multi-signal confidence score and triggers escalation on low confidence.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional
import uuid

import pdfplumber

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TableCell,
    TextBlock,
)
from src.strategies.base import BaseExtractor

_MIN_CHARS_PER_PAGE = 100
_MIN_CHAR_DENSITY = 10.0  # chars/1000 pt²
_MAX_IMAGE_RATIO = 0.50   # if images > 50% → low confidence


class FastTextExtractor(BaseExtractor):
    """
    Strategy A: pdfplumber-based extraction.
    Confidence signals:
    - character count per page
    - character density (chars/page area)
    - image-to-page area ratio
    - presence of font metadata
    """

    name = "fast_text"

    def extract(self, file_path: Path) -> ExtractedDocument:
        start = time.time()
        text_blocks: List[TextBlock] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        section_headings: List[TextBlock] = []
        full_text_parts: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    # ── Text Blocks ──────────────────────────────────────────
                    words = page.extract_words(
                        extra_attrs=["fontname", "size"],
                        keep_blank_chars=False,
                    ) or []

                    page_text = page.extract_text() or ""
                    full_text_parts.append(page_text)

                    # Group by lines using y-coord clustering
                    for word in words:
                        size = float(word.get("size", 12))
                        is_header = size >= 14
                        block = TextBlock(
                            text=word.get("text", ""),
                            bbox=BBox(
                                x0=float(word.get("x0", 0)),
                                y0=float(word.get("top", 0)),
                                x1=float(word.get("x1", 0)),
                                y1=float(word.get("bottom", 0)),
                                page=page_num,
                            ),
                            page=page_num,
                            font_name=word.get("fontname"),
                            font_size=size,
                            is_header=is_header,
                        )
                        text_blocks.append(block)
                        if is_header:
                            section_headings.append(block)

                    # ── Tables ───────────────────────────────────────────────
                    for t_idx, table in enumerate(page.find_tables() or []):
                        try:
                            raw_data = table.extract() or []
                            if not raw_data:
                                continue
                            headers = [str(h or "") for h in raw_data[0]]
                            rows = [
                                [str(c or "") for c in row]
                                for row in raw_data[1:]
                            ]
                            cells: List[TableCell] = []
                            for ri, row in enumerate(raw_data):
                                for ci, val in enumerate(row):
                                    cells.append(
                                        TableCell(
                                            value=str(val or ""),
                                            row=ri,
                                            col=ci,
                                            is_header=(ri == 0),
                                        )
                                    )
                            bbox_raw = table.bbox  # (x0, top, x1, bottom)
                            table_id = f"t_{page_num}_{t_idx}"
                            ext_table = ExtractedTable(
                                table_id=table_id,
                                bbox=BBox(
                                    x0=bbox_raw[0],
                                    y0=bbox_raw[1],
                                    x1=bbox_raw[2],
                                    y1=bbox_raw[3],
                                    page=page_num,
                                )
                                if bbox_raw
                                else None,
                                page=page_num,
                                headers=headers,
                                rows=rows,
                                cells=cells,
                            )
                            tables.append(ext_table)
                        except Exception:
                            continue

                    # ── Figures ──────────────────────────────────────────────
                    for img_idx, img in enumerate(page.images or []):
                        try:
                            fig = ExtractedFigure(
                                figure_id=f"f_{page_num}_{img_idx}",
                                bbox=BBox(
                                    x0=float(img.get("x0", 0)),
                                    y0=float(img.get("y0", 0)),
                                    x1=float(img.get("x1", 0)),
                                    y1=float(img.get("y1", 0)),
                                    page=page_num,
                                ),
                                page=page_num,
                            )
                            figures.append(fig)
                        except Exception:
                            continue
                except Exception:
                    continue

        doc_id = file_path.stem + "_" + file_path.stat().__hash__().__str__()[:8]
        result = ExtractedDocument(
            doc_id=doc_id,
            filename=file_path.name,
            strategy_used="fast_text",
            confidence_score=0.0,  # computed below
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text="\n\n".join(full_text_parts),
            section_headings=section_headings,
            processing_time_sec=round(time.time() - start, 2),
        )
        result.confidence_score = self.confidence(result)
        return result

    def confidence(self, doc: ExtractedDocument) -> float:
        """
        Multi-signal confidence score:
        - text_signal: chars per page vs minimum threshold
        - image_signal: penalize high image ratios
        - font_signal: bonus for font metadata presence
        """
        if doc.page_count == 0:
            return 0.0

        total_chars = sum(len(b.text) for b in doc.text_blocks)
        chars_per_page = total_chars / doc.page_count

        # Signal 1: character volume
        text_signal = min(chars_per_page / (_MIN_CHARS_PER_PAGE * 10), 1.0)

        # Signal 2: image ratio penalty (approximated from figure count)
        pages_with_figures = len(set(f.page for f in doc.figures))
        image_ratio = pages_with_figures / max(doc.page_count, 1)
        image_penalty = max(0.0, 1.0 - (image_ratio / _MAX_IMAGE_RATIO))

        # Signal 3: font metadata presence
        has_fonts = any(b.font_name for b in doc.text_blocks[:50])
        font_bonus = 0.1 if has_fonts else 0.0

        score = (0.6 * text_signal + 0.3 * image_penalty + font_bonus)
        return round(min(max(score, 0.0), 1.0), 3)
