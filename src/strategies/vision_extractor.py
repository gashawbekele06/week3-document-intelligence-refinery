"""
Strategy C: Vision-Augmented Extractor using OpenAI GPT-4o / GPT-4o-mini (with OpenRouter fallback).
Handles scanned PDFs, handwriting, and low-confidence situations.
Includes budget_guard to prevent runaway API costs.
"""

from __future__ import annotations

import base64
import re
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
import json
import requests

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from src.models import (
    BBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TextBlock,
)
from src.strategies.base import BaseExtractor

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _load_rules() -> dict:
    with open(_RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_json_payload(raw_text: str) -> dict:
    """Parse model output that may contain fences or extra prose around JSON."""
    text = (raw_text or "{}").strip()

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
        text = text.lstrip("json").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise RuntimeError("Model returned invalid JSON")


_EXTRACTION_PROMPT = """You are an enterprise document intelligence system analyzing a PDF page image.
Extract ALL visible content with precise structure:

Return ONLY a JSON object with exactly this schema — no markdown, no explanations:
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
1. Preserve ALL text including headers, footers, footnotes
2. Tables: exact column alignment, never merge cells
3. Numerical values: keep decimals, currency, units exact
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
            input_tokens * self.cost_per_1m_input / 1_000_000 +
            output_tokens * self.cost_per_1m_output / 1_000_000
        )
        self.total_cost += cost
        self.pages_processed += 1
        return cost


class VisionExtractor(BaseExtractor):
    """Strategy C: Vision extraction via OpenAI (preferred) or OpenRouter fallback."""

    name = "vision_extractor"

    def __init__(self):
        if load_dotenv:
            load_dotenv(override=True)
        self.rules = _load_rules()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_project_id = os.getenv("OPENAI_PROJECT") or os.getenv("OPENAI_PROJECT_ID")
        self.openai_disabled_reason: Optional[str] = None
        self.openrouter_disabled_reason: Optional[str] = None
        self.openai_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        raw_model = os.getenv("OPENROUTER_MODEL") or self.rules.get("budget", {}).get(
            "vision_model", "openai/gpt-4o-mini"
        )
        self.model = self._normalize_model_name(raw_model)
        self.openrouter_models = self._build_openrouter_models(self.model)

    def _normalize_model_name(self, model: str) -> str:
        model = (model or "").strip()
        if "/" in model:
            return model
        aliases = {
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt-4o": "openai/gpt-4o",
            "gemini-1.5-flash": "google/gemini-2.0-flash-001",
            "gemini-2.0-flash": "google/gemini-2.0-flash-001",
        }
        return aliases.get(model, model)

    def _build_openrouter_models(self, primary_model: str) -> List[str]:
        candidates = [
            primary_model,
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-001",
        ]
        seen = set()
        ordered = []
        for model in candidates:
            normalized = self._normalize_model_name(model)
            if normalized and normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        return ordered

    def _pdf_page_to_image_bytes(self, file_path: Path, page_num: int) -> Optional[bytes]:
        try:
            import fitz
            doc = fitz.open(str(file_path))
            page = doc[page_num - 1]
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        except Exception as e:
            print(f"Page {page_num} render failed: {e}")
            return None

    def _call_openai(self, image_b64: str) -> Tuple[dict, int, int]:
        """Call OpenAI Vision API with correct format."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        if self.openai_api_key.startswith("sk-proj-") and self.openai_project_id:
            headers["OpenAI-Project"] = self.openai_project_id

        prompts = [
            _EXTRACTION_PROMPT,
            _EXTRACTION_PROMPT
            + "\n\nReturn MINIFIED valid JSON only. Escape all inner quotes and newlines inside text values.",
        ]

        last_error = None
        for prompt in prompts:
            payload = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
                "response_format": {"type": "json_object"},
            }

            try:
                resp = requests.post(_OPENAI_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                parsed = _parse_json_payload(content)
                usage = data.get("usage", {})
                return parsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

            except requests.exceptions.HTTPError as e:
                error_text = e.response.text[:200] if e.response else str(e)
                raise RuntimeError(f"OpenAI HTTP error {e.response.status_code}: {error_text}")
            except requests.exceptions.HTTPError as e:
                error_text = e.response.text[:200] if e.response else str(e)
                last_error = RuntimeError(
                    f"OpenRouter HTTP error {e.response.status_code}: {error_text}"
                )
                continue
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"OpenAI call failed: {last_error}")

    def _call_openrouter(self, image_b64: str) -> Tuple[dict, int, int]:
        """Fallback to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "document-refinery",
            "Content-Type": "application/json",
        }

        last_error = None
        for model_name in self.openrouter_models:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            }

            try:
                resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                parsed = _parse_json_payload(content)
                usage = data.get("usage", {})
                self.model = model_name
                return parsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"OpenRouter call failed: {str(last_error)}")

    def _extract_page(
        self, img_bytes: bytes, page_num: int, budget_guard: BudgetGuard
    ) -> dict:
        """Extract structured content from one page image."""
        try:
            image_b64 = base64.b64encode(img_bytes).decode("utf-8")

            if self.openai_api_key and not self.openai_disabled_reason:
                try:
                    page_data, input_tokens, output_tokens = self._call_openai(image_b64)
                    budget_guard.record_usage(input_tokens, output_tokens)
                    page_data.setdefault("page_number", page_num)
                    return page_data
                except Exception as e:
                    if "HTTP error 401" in str(e):
                        self.openai_disabled_reason = "OpenAI authentication failed"
                    print(f"OpenAI failed (page {page_num}): {e} — trying OpenRouter")
                    if self.api_key and not self.openrouter_disabled_reason:
                        page_data, input_tokens, output_tokens = self._call_openrouter(image_b64)
                        budget_guard.record_usage(input_tokens, output_tokens)
                        page_data.setdefault("page_number", page_num)
                        return page_data
                    raise
            else:
                if self.openrouter_disabled_reason:
                    raise RuntimeError(self.openrouter_disabled_reason)
                page_data, input_tokens, output_tokens = self._call_openrouter(image_b64)
                budget_guard.record_usage(input_tokens, output_tokens)
                page_data.setdefault("page_number", page_num)
                return page_data

        except Exception as e:
            if "OpenRouter HTTP error 401" in str(e):
                self.openrouter_disabled_reason = "OpenRouter authentication failed"
            elif "OpenRouter HTTP error 402" in str(e):
                self.openrouter_disabled_reason = "OpenRouter account has no available credits"
            elif "OpenRouter HTTP error 403" in str(e):
                self.openrouter_disabled_reason = "OpenRouter request forbidden"
            return {
                "page_number": page_num,
                "text_blocks": [],
                "tables": [],
                "figures": [],
                "error": str(e)[:200],
            }

    def extract(self, file_path: Path) -> ExtractedDocument:
        start = time.time()
        budget_guard = BudgetGuard(self.rules)

        if not self.api_key and not self.openai_api_key:
            return ExtractedDocument(
                doc_id=file_path.stem + "_vision_fail",
                filename=file_path.name,
                strategy_used=self.name,
                confidence_score=0.0,
                page_count=0,
                metadata={"error": "No API key set (OPENAI_API_KEY or OPENROUTER_API_KEY)"},
            )

        try:
            import fitz
            pdf_doc = fitz.open(str(file_path))
            page_count = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            page_count = 1

        text_blocks = []
        tables = []
        figures = []
        section_headings = []
        full_text_parts = []
        errors = []

        for page_num in range(1, page_count + 1):
            if not budget_guard.can_process_page():
                errors.append("Budget cap reached — stopped processing")
                break

            if self.openrouter_disabled_reason and (
                not self.openai_api_key or self.openai_disabled_reason
            ):
                errors.append(self.openrouter_disabled_reason)
                break

            img_bytes = self._pdf_page_to_image_bytes(file_path, page_num)
            if not img_bytes:
                errors.append(f"p{page_num}: failed to render page to image")
                continue

            page_data = self._extract_page(img_bytes, page_num, budget_guard)
            if "error" in page_data:
                errors.append(f"p{page_num}: {page_data['error']}")
                continue

            # Parse text blocks
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

            # Parse tables
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

            # Parse figures
            for f_idx, fig in enumerate(page_data.get("figures", [])):
                ext_fig = ExtractedFigure(
                    figure_id=f"f_{page_num}_{f_idx}",
                    page=page_num,
                    caption=fig.get("caption"),
                )
                figures.append(ext_fig)

        duration = time.time() - start

        result = ExtractedDocument(
            doc_id=file_path.stem + "_vision",
            filename=file_path.name,
            strategy_used=self.name,
            confidence_score=0.0,
            page_count=page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text="\n\n".join(full_text_parts),
            section_headings=section_headings,
            cost_estimate_usd=round(budget_guard.total_cost, 6),
            processing_time_sec=round(duration, 2),
            metadata={
                "budget_remaining_usd": round(budget_guard.max_cost_usd - budget_guard.total_cost, 6),
                "pages_processed_by_vision": budget_guard.pages_processed,
                "model": self.openai_model if self.openai_api_key else self.model,
                "provider": (
                    "openai"
                    if self.openai_api_key and not self.openai_disabled_reason
                    else "openrouter"
                ),
                "openai_disabled_reason": self.openai_disabled_reason,
                "openrouter_disabled_reason": self.openrouter_disabled_reason,
                "errors": errors[:10],
            },
        )

        result.confidence_score = self.confidence(result)
        return result

    def confidence(self, doc: ExtractedDocument) -> float:
        """Vision extraction confidence score."""
        if doc.page_count == 0:
            return 0.0
        has_text = len(doc.text_blocks) > 0
        text_score = min(
            sum(len(b.text) for b in doc.text_blocks) / (doc.page_count * 100 + 1), 1.0
        )
        structure_bonus = 0.25 if doc.tables or doc.figures else 0.0
        score = 0.65 * text_score + 0.35 * int(has_text) + structure_bonus
        return round(min(max(score, 0.0), 0.98), 3)