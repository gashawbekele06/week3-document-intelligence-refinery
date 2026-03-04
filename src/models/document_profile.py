from __future__ import annotations
from typing import Optional, Literal, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field


class DocumentProfile(BaseModel):
    """Stage 1 output: full classification of a document before extraction begins."""

    model_config = ConfigDict(use_enum_values=True)

    doc_id: str = Field(..., description="SHA-256 hash of file path + mtime")
    filename: str
    file_path: str
    page_count: int

    # ── Classification Dimensions ───────────────────────────────────────────
    origin_type: Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
    layout_complexity: Literal[
        "single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"
    ]
    language: str = Field(default="en", description="ISO 639-1 language code")
    language_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"]

    # ── Extraction Cost Estimate ────────────────────────────────────────────
    estimated_cost: Literal[
        "fast_text_sufficient", "needs_layout_model", "needs_vision_model"
    ]

    # ── Raw Signals (stored for audit) ─────────────────────────────────────
    char_density_mean: float = Field(
        description="Mean chars per 1000 pt² across pages", ge=0.0
    )
    image_ratio_mean: float = Field(
        description="Mean fraction of page area occupied by images", ge=0.0, le=1.0
    )
    has_font_metadata: bool = Field(
        default=False, description="PDF embeds font/encoding data"
    )
    is_form_fillable: bool = Field(default=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
