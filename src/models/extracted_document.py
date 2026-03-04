from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """Spatial bounding box in PDF point coordinates (origin = bottom-left)."""

    model_config = ConfigDict()

    x0: float
    y0: float
    x1: float
    y1: float
    page: int = -1

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}


class TextBlock(BaseModel):
    """A contiguous block of extracted text with spatial context."""

    model_config = ConfigDict()

    text: str
    bbox: Optional[BBox] = None
    page: int = -1
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_header: bool = False
    reading_order: int = 0


class TableCell(BaseModel):
    value: str
    row: int
    col: int
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1


class ExtractedTable(BaseModel):
    """A structured table with headers and rows, retaining bounding box."""

    model_config = ConfigDict()

    table_id: str
    bbox: Optional[BBox] = None
    page: int = -1
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)
    caption: Optional[str] = None
    reading_order: int = 0

    def to_markdown(self) -> str:
        """Render table as Markdown for embedding."""
        if not self.headers:
            return ""
        header_row = "| " + " | ".join(self.headers) + " |"
        separator = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        data_rows = "\n".join(
            "| " + " | ".join(str(cell) for cell in row) + " |" for row in self.rows
        )
        return f"{header_row}\n{separator}\n{data_rows}"


class ExtractedFigure(BaseModel):
    """A figure block with its caption."""

    model_config = ConfigDict()

    figure_id: str
    bbox: Optional[BBox] = None
    page: int = -1
    caption: Optional[str] = None
    image_bytes: Optional[bytes] = None
    reading_order: int = 0


class ExtractedDocument(BaseModel):
    """
    Normalized intermediate representation output by ALL three extraction strategies.
    Every strategy adapter must populate this schema before downstream stages.
    """

    model_config = ConfigDict()

    doc_id: str
    filename: str
    strategy_used: Literal["fast_text", "layout_extractor", "vision_extractor"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    page_count: int

    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[ExtractedTable] = Field(default_factory=list)
    figures: List[ExtractedFigure] = Field(default_factory=list)

    # Flat full text for convenience (reading-order sorted)
    full_text: str = ""

    # Section headings detected
    section_headings: List[TextBlock] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)
    cost_estimate_usd: float = 0.0
    processing_time_sec: float = 0.0
