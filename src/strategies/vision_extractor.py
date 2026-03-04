"""
Strategy C: Vision-Augmented Extractor using Gemini Flash.
Handles scanned PDFs, handwriting, and low-confidence situations.
Includes budget_guard to prevent runaway API costs.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import List, Optional
import json

import yaml

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TextBlock,
)
from src.strategies.base import BaseExtractor

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"


def _load_rules() -> dict:
    with open(_RULES_PATH) as f:
        return yaml.safe_load(f)


_EXTRACTION_PROMPT = """You are an enterprise document intelligence system analyzing a PDF page image.
Extract ALL visible content with precise structure:

Return a JSON object with exactly this schema:
{
  "page_number": <int>,
  "text_blocks": [
    {"text": "<full paragraph text>", "is_header": <bool>, "reading_order": <int>}
  ],
  "tables": [
    {
      "caption": "<table caption if visible>",
      "headers": ["col1", "col2", ...],
      "rows": [["val1", "val2", ...], ...]
    }
  ],
  "figures": [
    {"caption": "<figure caption if visible>"}
  ]
}

Rules:
1. Preserve ALL text including headers, footers, and footnotes
2. Extract tables with EXACT column alignment — never merge cells incorrectly
3. For numerical tables: preserve decimal points, currency symbols, and units exactly
4. Return ONLY valid JSON, no markdown fences
"""


class BudgetGuard:
    """Tracks cumulative API cost per document and enforces budget cap."""

    def __init__(self, rules: dict):
        budget = rules.get("budget", {})
        self.max_cost_usd = budget.get("max_cost_per_doc_usd", 0.10)
        self.cost_per_1m_input = budget.get("token_cost_per_1m_input", 0.075)
        self.cost_per_1m_output = budget.get("token_cost_per_1m_output", 0.30)
        self.max_pages = budget.get("max_pages_vision_per_doc", 50)
        self.total_cost = 0.0
        self.pages_processed = 0

    def can_process_page(self) -> bool:
        return self.total_cost < self.max_cost_usd and self.pages_processed < self.max_pages

    def record_usage(self, input_tokens: int, output_tokens: int) -> float:
        cost = (
            input_tokens * self.cost_per_1m_input / 1_000_000
            + output_tokens * self.cost_per_1m_output / 1_000_000
        )
        self.total_cost += cost
        self.pages_processed += 1
        return cost


class VisionExtractor(BaseExtractor):
    """
    Strategy C: Gemini Flash vision-based extraction.
    Converts PDF pages to images and sends structured extraction prompts.
    Budget guard prevents runaway API costs.
    """

    name = "vision_extractor"

    def __init__(self):
        self.rules = _load_rules()
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel("gemini-1.5-flash")
        return self._client

    def _pdf_page_to_image_bytes(self, file_path: Path, page_num: int) -> Optional[bytes]:
        """Render PDF page to PNG bytes at 150 DPI using pymupdf."""
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(file_path))
            page = doc[page_num - 1]
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        except Exception:
            return None

    def _extract_page(
        self, client, img_bytes: bytes, page_num: int, budget_guard: BudgetGuard
    ) -> dict:
        """Send one page image to Gemini Flash and parse structured response."""
        try:
            import google.generativeai as genai
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_bytes))
            response = client.generate_content(
                [_EXTRACTION_PROMPT, img],
                generation_config={"temperature": 0.1, "max_output_tokens": 2048},
            )
            text = response.text or "{}"
            # Strip markdown fences if present
            if "```" in text:
                text = text.split("```")[1] if "```json" in text else text.split("```")[1]
                text = text.strip().lstrip("json").strip()

            data = json.loads(text)
            # Record usage estimate (Gemini Flash doesn't always return token counts)
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 1000) if usage else 1000
            output_tokens = getattr(usage, "candidates_token_count", 500) if usage else 500
            budget_guard.record_usage(input_tokens, output_tokens)
            return data
        except Exception as e:
            return {"page_number": page_num, "text_blocks": [], "tables": [], "figures": []}

    def extract(self, file_path: Path) -> ExtractedDocument:
        start = time.time()
        budget_guard = BudgetGuard(self.rules)

        if not self.api_key:
            # No API key: return empty document
            return ExtractedDocument(
                doc_id=file_path.stem + "_vision",
                filename=file_path.name,
                strategy_used="vision_extractor",
                confidence_score=0.0,
                page_count=0,
                metadata={"error": "GEMINI_API_KEY not set"},
            )

        try:
            import fitz
            pdf_doc = fitz.open(str(file_path))
            page_count = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            page_count = 1

        client = self._get_client()
        text_blocks: List[TextBlock] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        section_headings: List[TextBlock] = []
        full_text_parts: List[str] = []

        for page_num in range(1, page_count + 1):
            if not budget_guard.can_process_page():
                break

            img_bytes = self._pdf_page_to_image_bytes(file_path, page_num)
            if not img_bytes:
                continue

            page_data = self._extract_page(client, img_bytes, page_num, budget_guard)

            # ── Text Blocks ────────────────────────────────────────────────
            for order, tb in enumerate(page_data.get("text_blocks", [])):
                text = tb.get("text", "").strip()
                if not text:
                    continue
                is_header = tb.get("is_header", False)
                block = TextBlock(
                    text=text,
                    page=page_num,
                    is_header=is_header,
                    reading_order=order,
                )
                text_blocks.append(block)
                if is_header:
                    section_headings.append(block)
                full_text_parts.append(text)

            # ── Tables ────────────────────────────────────────────────────
            for t_idx, tbl in enumerate(page_data.get("tables", [])):
                headers = [str(h) for h in tbl.get("headers", [])]
                rows = [[str(c) for c in row] for row in tbl.get("rows", [])]
                if not headers and not rows:
                    continue
                ext_table = ExtractedTable(
                    table_id=f"t_{page_num}_{t_idx}",
                    page=page_num,
                    headers=headers,
                    rows=rows,
                    caption=tbl.get("caption"),
                )
                tables.append(ext_table)

            # ── Figures ──────────────────────────────────────────────────
            for f_idx, fig in enumerate(page_data.get("figures", [])):
                ext_fig = ExtractedFigure(
                    figure_id=f"f_{page_num}_{f_idx}",
                    page=page_num,
                    caption=fig.get("caption"),
                )
                figures.append(ext_fig)

        result = ExtractedDocument(
            doc_id=file_path.stem + "_vision",
            filename=file_path.name,
            strategy_used="vision_extractor",
            confidence_score=0.0,
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text="\n\n".join(full_text_parts),
            section_headings=section_headings,
            cost_estimate_usd=round(budget_guard.total_cost, 6),
            processing_time_sec=round(time.time() - start, 2),
            metadata={
                "budget_remaining_usd": round(
                    budget_guard.max_cost_usd - budget_guard.total_cost, 6
                ),
                "pages_processed_by_vision": budget_guard.pages_processed,
            },
        )
        result.confidence_score = self.confidence(result)
        return result

    def confidence(self, doc: ExtractedDocument) -> float:
        """Vision extraction is our highest-confidence strategy when it succeeds."""
        if doc.page_count == 0:
            return 0.0
        has_text = len(doc.text_blocks) > 0
        text_score = min(
            sum(len(b.text) for b in doc.text_blocks) / (doc.page_count * 100), 1.0
        )
        return round(0.7 * text_score + 0.3 * int(has_text), 3)
