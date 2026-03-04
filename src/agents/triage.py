"""
Stage 1: The Triage Agent
Classifies a document to produce DocumentProfile governing all downstream decisions.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import yaml

from src.models import DocumentProfile

_RULES_PATH = Path(__file__).parent.parent.parent / "rubric" / "extraction_rules.yaml"
_REFINERY_DIR = Path(".refinery") / "profiles"


def _load_rules() -> dict:
    with open(_RULES_PATH) as f:
        return yaml.safe_load(f)


def _doc_id(file_path: Path) -> str:
    """Deterministic document ID based on file path and size."""
    raw = f"{file_path.resolve()}:{file_path.stat().st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── Signal Extraction ────────────────────────────────────────────────────────

def _compute_char_density(page) -> float:
    """
    Chars per 1000 pt² page area.
    A near-zero value indicates a scanned or image-only page.
    """
    try:
        chars = page.chars or []
        text = "".join(c.get("text", "") for c in chars if c.get("text", "").strip())

        # Fallback: some PDFs (including our PyMuPDF-generated fixtures) do not
        # populate `page.chars` but still return text via `extract_text`.
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
    """Fraction of page area covered by embedded image objects."""
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
    """Check if the PDF contains font metadata (sign of native digital text)."""
    try:
        for page in pdf.pages[:3]:
            if page.chars:
                for c in page.chars:
                    if c.get("fontname"):
                        return True
    except Exception:
        pass
    return False


def _is_form_fillable(pdf) -> bool:
    """Detect AcroForm annotation (fillable form PDF)."""
    try:
        trailer = pdf.doc.trailer
        if trailer and "/Root" in str(trailer):
            root = pdf.doc.trailer.get("/Root", {})
            if hasattr(root, "get") and root.get("/AcroForm"):
                return True
    except Exception:
        pass
    return False


# ─── Origin Type Detection ────────────────────────────────────────────────────

def _detect_origin_type(
    char_densities: List[float],
    image_ratios: List[float],
    has_fonts: bool,
    is_form: bool,
    rules: dict,
) -> str:
    if is_form:
        return "form_fillable"
    mean_density = sum(char_densities) / max(len(char_densities), 1)
    mean_image = sum(image_ratios) / max(len(image_ratios), 1)
    scanned_max = rules["origin_detection"]["scanned_max_char_density"]
    scanned_img_min = rules["origin_detection"]["scanned_min_image_ratio"]
    digital_min = rules["origin_detection"]["digital_min_char_density"]
    mixed_lower = rules["origin_detection"]["mixed_image_ratio_lower"]

    # If there is effectively no text and no font metadata, treat as scanned even if image ratio is low/unknown.
    if mean_density < scanned_max * 0.2 and not has_fonts:
        return "scanned_image"

    # High image coverage is a strong scanned signal even if an OCR text layer exists.
    if mean_image >= max(0.8, scanned_img_min):
        return "scanned_image"

    if mean_density < scanned_max and mean_image > scanned_img_min:
        return "scanned_image"
    if mixed_lower <= mean_image <= scanned_img_min:
        return "mixed"
    # Strong signal: embedded fonts + low image coverage → native digital
    if has_fonts and mean_image < scanned_img_min:
        return "native_digital"
    if mean_density >= digital_min and mean_image < mixed_lower:
        return "native_digital"
    if mean_density >= digital_min * 0.3:
        return "native_digital"
    return "mixed"


# ─── Layout Complexity Detection ──────────────────────────────────────────────

def _detect_layout_complexity(
    pages_sample,
    rules: dict,
) -> str:
    """
    Uses x-coordinate clustering of text block centers to count columns.
    Also estimates table and figure coverage.
    """
    x_centers: List[float] = []
    total_table_area = 0.0
    total_figure_area = 0.0
    total_page_area = 0.0

    for page in pages_sample:
        try:
            width = page.width or 1
            height = page.height or 1
            page_area = width * height
            total_page_area += page_area

            # Text block x-centers
            words = page.extract_words()
            for w in (words or []):
                cx = (float(w.get("x0", 0)) + float(w.get("x1", 0))) / 2
                x_centers.append(cx)

            # Table area estimate
            tables = page.find_tables()
            for t in (tables or []):
                bbox = t.bbox
                if bbox:
                    ta = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    total_table_area += ta

            # Figure area estimate
            for img in (page.images or []):
                try:
                    fa = (img["x1"] - img["x0"]) * (img["y1"] - img["y0"])
                    total_figure_area += fa
                except Exception:
                    pass
        except Exception:
            continue

    table_ratio = total_table_area / max(total_page_area, 1)
    figure_ratio = total_figure_area / max(total_page_area, 1)

    table_heavy_thresh = rules["layout_detection"]["table_heavy_min_ratio"]
    figure_heavy_thresh = rules["layout_detection"]["figure_heavy_min_ratio"]
    multi_col_clusters = rules["layout_detection"]["multi_column_min_clusters"]

    if table_ratio > table_heavy_thresh and figure_ratio > figure_heavy_thresh:
        return "mixed"
    if table_ratio > table_heavy_thresh:
        return "table_heavy"
    if figure_ratio > figure_heavy_thresh:
        return "figure_heavy"

    # Column detection via x-center clustering
    if x_centers:
        page_width = pages_sample[0].width or 600
        n_clusters = _count_x_clusters(x_centers, page_width, multi_col_clusters)
        if n_clusters >= multi_col_clusters:
            return "multi_column"

    return "single_column"


def _count_x_clusters(x_centers: List[float], page_width: float, target_clusters: int) -> int:
    """Simple gap-based x-cluster counting to detect multi-column layouts."""
    if not x_centers:
        return 1
    sorted_x = sorted(x_centers)
    # Normalize to 0-1
    norm = [(x / page_width) for x in sorted_x]
    # Find gaps > 0.15 page width
    gaps = [norm[i + 1] - norm[i] for i in range(len(norm) - 1)]
    significant_gaps = sum(1 for g in gaps if g > 0.15)
    return significant_gaps + 1


# ─── Domain Hint Classification ───────────────────────────────────────────────

class DomainClassifier:
    """
    Pluggable strategy interface for domain classification.
    Swap `classify` with a VLM-based classifier without changing call sites.
    """

    def __init__(self, rules: dict):
        self.keywords: Dict[str, List[str]] = rules.get("domain_keywords", {})

    def classify(self, text_sample: str) -> Tuple[str, float]:
        """Returns (domain, confidence) based on keyword density."""
        text_lower = text_sample.lower()
        scores: Dict[str, int] = {}
        for domain, kws in self.keywords.items():
            score = sum(1 for kw in kws if kw in text_lower)
            scores[domain] = score

        if not scores or max(scores.values()) == 0:
            return "general", 0.3

        total = sum(scores.values())
        best = max(scores, key=lambda k: scores[k])
        confidence = scores[best] / total if total > 0 else 0.3
        return best, round(confidence, 3)


# ─── Cost Estimation ─────────────────────────────────────────────────────────

def _estimate_cost(origin: str, layout: str) -> str:
    if origin == "scanned_image":
        return "needs_vision_model"
    if origin in ("native_digital", "form_fillable") and layout == "single_column":
        return "fast_text_sufficient"
    return "needs_layout_model"


# ─── Main Triage Agent ────────────────────────────────────────────────────────

class TriageAgent:
    """
    Stage 1 agent: Classifies a document and produces a DocumentProfile.
    Profile is saved to .refinery/profiles/{doc_id}.json for downstream stages.
    """

    def __init__(self, refinery_dir: Optional[Path] = None):
        self.rules = _load_rules()
        self.domain_classifier = DomainClassifier(self.rules)
        self.refinery_dir = refinery_dir or _REFINERY_DIR
        self.refinery_dir.mkdir(parents=True, exist_ok=True)

    def triage(self, file_path: str | Path) -> DocumentProfile:
        """Classify a document and return its DocumentProfile."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        doc_id = _doc_id(file_path)
        cached_path = self.refinery_dir / f"{doc_id}.json"
        if cached_path.exists():
            return DocumentProfile.model_validate_json(cached_path.read_text())

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            sample_pages = pdf.pages[: min(10, page_count)]

            # Compute per-page signals
            char_densities = [_compute_char_density(p) for p in sample_pages]
            image_ratios = [_compute_image_ratio(p) for p in sample_pages]
            has_fonts = _has_font_metadata(pdf)
            is_form = _is_form_fillable(pdf)

            # Extract text sample for domain classification
            text_parts = []
            for p in sample_pages[:5]:
                try:
                    t = p.extract_text() or ""
                    text_parts.append(t[:500])
                except Exception:
                    pass
            text_sample = " ".join(text_parts)

            origin_type = _detect_origin_type(
                char_densities, image_ratios, has_fonts, is_form, self.rules
            )
            layout_complexity = _detect_layout_complexity(sample_pages, self.rules)
            domain, lang_conf = self.domain_classifier.classify(text_sample)
            estimated_cost = _estimate_cost(origin_type, layout_complexity)

            mean_density = sum(char_densities) / max(len(char_densities), 1)
            mean_image = sum(image_ratios) / max(len(image_ratios), 1)

            profile = DocumentProfile(
                doc_id=doc_id,
                filename=file_path.name,
                file_path=str(file_path.resolve()),
                page_count=page_count,
                origin_type=origin_type,
                layout_complexity=layout_complexity,
                language="en",
                language_confidence=0.9,
                domain_hint=domain,
                estimated_cost=estimated_cost,
                char_density_mean=round(mean_density, 2),
                image_ratio_mean=round(mean_image, 4),
                has_font_metadata=has_fonts,
                is_form_fillable=is_form,
            )

        # Save profile
        cached_path.write_text(profile.model_dump_json(indent=2))
        return profile
