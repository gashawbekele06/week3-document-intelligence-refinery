import fitz
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.triage import TriageAgent


def _make_pdf(tmp_path: Path, text: str, filename: str = "sample.pdf") -> Path:
    """Create a minimal single-page PDF with provided text using PyMuPDF."""
    out = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter points
    page.insert_textbox((72, 72, 540, 720), text)
    doc.save(out)
    doc.close()
    return out


def test_triage_native_digital_single_column(tmp_path):
    text = """
    Revenue grew to 4.2B birr while profit rose. This annual report includes balance sheet
    information and auditor notes. No images are present, text is single column.
    """
    pdf_path = _make_pdf(tmp_path, text)

    agent = TriageAgent(refinery_dir=tmp_path / "profiles")
    profile = agent.triage(pdf_path)

    assert profile.origin_type == "native_digital"
    assert profile.layout_complexity == "single_column"
    assert profile.domain_hint == "financial"  # keyword-based classifier
    assert profile.estimated_cost == "fast_text_sufficient"

    cached = (tmp_path / "profiles" / f"{profile.doc_id}.json")
    assert cached.exists()


def test_triage_scanned_image_escalates_to_vision(monkeypatch, tmp_path):
    # Create a placeholder PDF; signals will be monkeypatched to emulate scanned
    pdf_path = _make_pdf(tmp_path, "filler")

    # Force low char density and high image ratio to trigger scanned_image
    monkeypatch.setattr("src.agents.triage._compute_char_density", lambda _: 0.0)
    monkeypatch.setattr("src.agents.triage._compute_image_ratio", lambda _: 0.9)

    agent = TriageAgent(refinery_dir=tmp_path / "profiles_scanned")
    profile = agent.triage(pdf_path)

    assert profile.origin_type == "scanned_image"
    assert profile.estimated_cost == "needs_vision_model"