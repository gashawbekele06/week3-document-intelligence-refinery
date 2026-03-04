"""
ExtractionRouter: reads DocumentProfile and delegates to the appropriate strategy.
Implements the escalation guard: A→B→C on low confidence.
Logs every extraction to .refinery/extraction_ledger.jsonl.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from src.models import DocumentProfile, ExtractedDocument
from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
_LEDGER_PATH = Path(".refinery") / "extraction_ledger.jsonl"


def _load_rules() -> dict:
    with open(_RULES_PATH) as f:
        return yaml.safe_load(f)


class ExtractionRouter:
    """
    Strategy pattern router with confidence-gated escalation.

    Routing logic (from extraction_rules.yaml):
    - Profile says fast_text_sufficient → try A first; escalate to B if confidence < 0.5
    - Profile says needs_layout_model → use B directly; escalate to C if confidence < 0.4
    - Profile says needs_vision_model → use C directly

    The escalation guard ensures bad extractions never silently pass downstream.
    """

    def __init__(self, ledger_path: Optional[Path] = None):
        self.rules = _load_rules()
        thresholds = self.rules.get("confidence_thresholds", {})
        self.fast_text_min = thresholds.get("fast_text_min", 0.5)
        self.layout_min = thresholds.get("layout_min", 0.4)

        self.extractor_a = FastTextExtractor()
        self.extractor_b = LayoutExtractor()
        self.extractor_c = VisionExtractor()

        self.ledger_path = ledger_path or _LEDGER_PATH
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def route(self, file_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        """
        Route document to appropriate extractor based on profile.
        Applies escalation guard: low-confidence results cascade to higher-tier strategy.
        """
        cost_estimate = profile.estimated_cost
        doc = None
        escalation_path: list[str] = []

        if cost_estimate == "fast_text_sufficient":
            doc, escalation_path = self._try_with_escalation(
                file_path,
                start_extractor=self.extractor_a,
                start_threshold=self.fast_text_min,
                fallback_extractor=self.extractor_b,
                fallback_threshold=self.layout_min,
                final_extractor=self.extractor_c,
            )

        elif cost_estimate == "needs_layout_model":
            doc, escalation_path = self._try_with_escalation(
                file_path,
                start_extractor=self.extractor_b,
                start_threshold=self.layout_min,
                fallback_extractor=self.extractor_c,
                fallback_threshold=0.0,  # C is final
                final_extractor=None,
            )

        else:  # needs_vision_model
            doc = self.extractor_c.extract(file_path)
            escalation_path = ["vision_extractor"]

        # Attach doc_id from profile
        doc.doc_id = profile.doc_id

        # Log to ledger
        self._log_ledger(
            profile=profile,
            doc=doc,
            escalation_path=escalation_path,
        )
        return doc

    def _try_with_escalation(
        self,
        file_path: Path,
        start_extractor: BaseExtractor,
        start_threshold: float,
        fallback_extractor: BaseExtractor,
        fallback_threshold: float,
        final_extractor: Optional[BaseExtractor],
    ) -> tuple[ExtractedDocument, list[str]]:
        """
        Try start_extractor → if confidence < threshold, escalate to fallback → then final.
        Returns (ExtractedDocument, escalation_path).
        """
        escalation_path = [start_extractor.name]
        doc = start_extractor.extract(file_path)

        if doc.confidence_score < start_threshold:
            escalation_path.append(fallback_extractor.name)
            doc = fallback_extractor.extract(file_path)

            if final_extractor and doc.confidence_score < fallback_threshold:
                escalation_path.append(final_extractor.name)
                doc = final_extractor.extract(file_path)

        return doc, escalation_path

    def _log_ledger(
        self,
        profile: DocumentProfile,
        doc: ExtractedDocument,
        escalation_path: list[str],
    ) -> None:
        """Append an extraction record to .refinery/extraction_ledger.jsonl."""
        entry = {
            "doc_id": profile.doc_id,
            "filename": profile.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_used": doc.strategy_used,
            "escalation_path": escalation_path,
            "confidence_score": doc.confidence_score,
            "cost_estimate_usd": doc.cost_estimate_usd,
            "processing_time_sec": doc.processing_time_sec,
            "page_count": doc.page_count,
            "origin_type": profile.origin_type,
            "layout_complexity": profile.layout_complexity,
            "domain_hint": profile.domain_hint,
            "text_blocks": len(doc.text_blocks),
            "tables_extracted": len(doc.tables),
            "figures_extracted": len(doc.figures),
        }
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
