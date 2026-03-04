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
_TARGET_CHAR_DENSITY = 150.0  # chars/1000 pt² for clean digital pages
_MAX_IMAGE_RATIO = 0.50   # if images > 50% → low confidence


def _compute_char_density(page) -> float:
    """Chars per 1000 pt² page area, using text fallback when chars missing."""
    try:
        chars = page.chars or []
        text = "".join(c.get("text", "") for c in chars if c.get("text", "").strip())
        if not text:
            extracted = page.extract_text() or ""
            text = extracted
        width = page.width or 1
        height = page.height or 1
        area_1000pt2 = (width * height) / 1000.0
        return len(text) / area_1000pt2 if area_1000pt2 > 0 else 0.0
    except Exception:
        return 0.0


def _compute_image_ratio(page) -> float:
    """Fraction of page area covered by embedded images."""
    try:
        page_area = (page.width or 1) * (page.height or 1)
        image_area = sum(
            (im["x1"] - im["x0"]) * (im["y1"] - im["y0"])
            for im in (page.images or [])
            if im.get("x1") and im.get("x0") and im.get("y1") and im.get("y0")
        )
        return min(image_area / page_area, 1.0) if page_area > 0 else 0.0
    except Exception:
        return 0.0


def _has_font_metadata(pdf) -> bool:
    """Check if PDF embeds font metadata (digital signal)."""
    try:
        for page in pdf.pages[:3]:
            if page.chars:
                for c in page.chars:
                    if c.get("fontname"):
                        return True
    except Exception:
        pass
    return False


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

        char_densities: List[float] = []
        image_ratios: List[float] = []
        has_fonts = False

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            has_fonts = _has_font_metadata(pdf)
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    # Signals
                    char_densities.append(_compute_char_density(page))
                    image_ratios.append(_compute_image_ratio(page))

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
            metadata={
                "char_density_mean": round(sum(char_densities) / max(len(char_densities), 1), 3),
                "image_ratio_mean": round(sum(image_ratios) / max(len(image_ratios), 1), 4),
                "has_font_metadata": bool(has_fonts),
                "char_densities": char_densities,
                "image_ratios": image_ratios,
            },
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

        signals = doc.metadata or {}
        char_density_mean = signals.get("char_density_mean", 0.0)
        image_ratio_mean = signals.get("image_ratio_mean", 0.0)
        has_fonts = signals.get("has_font_metadata", False) or any(b.font_name for b in doc.text_blocks[:50])

        if char_density_mean == 0.0 and doc.text_blocks:
            # Fallback: approximate density from text volume when page geometry is absent (e.g., synthetic docs in tests)
            char_density_mean = _TARGET_CHAR_DENSITY * min(chars_per_page / _MIN_CHARS_PER_PAGE, 1.0)

        # Signal 1: character volume across the doc
        text_signal = min(chars_per_page / _MIN_CHARS_PER_PAGE, 1.0)

        # Signal 2: character density (chars per 1000 pt²)
        density_signal = min(char_density_mean / _TARGET_CHAR_DENSITY, 1.0)

        # Signal 3: image ratio penalty (mean page image coverage)
        image_penalty = max(0.0, 1.0 - (image_ratio_mean / _MAX_IMAGE_RATIO))

        # Signal 4: font metadata presence bonus
        font_bonus = 0.1 if has_fonts else 0.0

        score = 0.45 * text_signal + 0.35 * density_signal + 0.15 * image_penalty + font_bonus

        # Strongly downweight likely scanned pages (image-dominant)
        if image_ratio_mean > 0.8:
            score = min(score, 0.2)

        return round(min(max(score, 0.0), 1.0), 3)
