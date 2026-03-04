# DOMAIN_NOTES.md — Document Intelligence Refinery
## Phase 0: Domain Onboarding Primer (living memo)

This note captures the *rules of the road* for triage, routing, chunking, and cost controls. Thresholds are governed by `rubric/extraction_rules.yaml`; changing that file reconfigures behavior without code edits.

---

## 1) Signals & Thresholds (Triage Cheatsheet)

- **Character density (chars / 1000 pt²)**
  - `< 2` **and** no embedded fonts → force `scanned_image` (handles near-zero text layers)
  - `< 10` **and** image ratio > 0.70 → `scanned_image`
  - `≥ 50` with image ratio < 0.30 → `native_digital`
- **Image ratio** (page area with image objects)
  - `≥ 0.80` → strong scanned signal even if OCR text exists
  - `0.30–0.70` → `mixed`; multi-column or table-heavy still routes to layout
- **Fonts present?**
  - Embedded font metadata boosts confidence and biases toward `native_digital`
- **Form fillable**
  - `/AcroForm` present → `form_fillable` origin and `fast_text_sufficient` cost hint
- **Layout cues**
  - X-center clusters ≥ 2 → `multi_column`
  - Table area > 0.30 or figure area > 0.40 of page → `table_heavy` / `figure_heavy`

Routing thresholds (from rules):
- Strategy A (FastText/pdfplumber): `fast_text_min = 0.50` → below escalates to B
- Strategy B (Docling): `layout_min = 0.40` → below escalates to C
- Min chars/page: 100 for healthy A-confidence
- Budget guard for Vision: max 50 pages, $0.10/doc

---

## 2) Routing at a Glance

| Origin / Layout | Primary Strategy | Escalation |
|---|---|---|
| `native_digital` + `single_column` | A: FastText (pdfplumber) | if conf < 0.5 → B |
| `native_digital` + `multi_column` / `table_heavy` / `figure_heavy` | B: Layout (Docling) | if conf < 0.4 → C |
| `mixed` (0.30–0.70 image ratio) | B: Layout (Docling) | if conf < 0.4 → C |
| `scanned_image` or image ratio ≥ 0.80, or density < 2 w/o fonts | C: Vision (Gemini 1.5 Flash) | budget-capped |
| `form_fillable` | A by default | escalates like native_digital |

Cost hint (`estimated_cost`):
- `needs_vision_model` when origin is `scanned_image`
- `fast_text_sufficient` when origin is digital/form and layout is single_column
- `needs_layout_model` otherwise

---

## 3) Corpus Coverage (current run)

- Profiles in `.refinery/profiles/`: **16**
- Class mix: **native_digital=10, mixed=3, scanned_image=3** (form_fillable=0 observed)
- Ledger populated in `.refinery/extraction_ledger.jsonl`
- PageIndex entries written under `.refinery/pageindex/`

---

## 4) Archetypes & Failure Modes (what we saw)

**Native digital — annual report (CBE 2023-24)**
- Char density ~380–450; multi-column front matter + dense tables
- Failures: ordering breaks on multi-column; tables across pages lose headers; footnotes merge
- Mitigation: Strategy B (Docling) for layout-aware ordering and header carry-forward

**Scanned audit packets (e.g., 2013 E.C. PDFs)**
- Char density ~0; image ratio ~0.95–0.99
- Failures: pdfplumber yields zero text; stamps/signatures misread; low DPI OCR noise
- Mitigation: Strategy C (Vision). Prefer 300 DPI renders; respect 50-page cap

**Mixed narrative + embedded image tables (FTA Survey 2022)**
- Char density ~120–180; image ratio ~0.20–0.35
- Failures: image-rendered tables missed by Strategy A; hierarchical numbering flattened
- Mitigation: Strategy B detects table boxes even when rasterized; preserves list hierarchy

**Table-heavy fiscal reports (Tax Expenditure)**
- Char density ~400+; tables dominate (>0.30 area)
- Failures: rowspan/colspan collapse; continuation pages lose headers; percent values parsed as text
- Mitigation: Strategy B table merger + header carry-forward; chunk tables as atomic LDUs

---

## 5) Chunking & PageIndex Constitution

- Max tokens per LDU: 512; split only if >256 tokens
- Tables never split from headers; lists kept intact unless over max
- Figure captions stored as metadata; headers become `parent_section`
- Cross-references resolved and stored as relationships
- PageIndex stores section tree + summaries + entities under `.refinery/pageindex/`

---

## 6) Cost & Performance Guardrails

| Metric | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| Tooling | pdfplumber | Docling | Gemini 1.5 Flash |
| Speed | ~0.5s/page (CPU) | ~5s/page (CPU) | ~3s/page (API) |
| Cost | $0 | $0 | ~$0.000038/page input est.; cap $0.10/doc |
| Strength | Clean text | Multi-column & tables | Scanned / image-heavy |
| Weakness | Scanned, complex layout | Pure scans | Budget, latency |

Client-friendly framing:
> Three tiers: free fast-text for clean PDFs; layout-aware for tables/columns at no API cost; vision last-resort for scans, capped at $0.10/doc and 50 pages. We auto-escalate when confidence dips (<0.5 for A, <0.4 for B).

---

## 7) Provenance Model

Every fact keeps spatial anchors (`bbox`) plus `strategy_used`, `page_number`, and `content_hash`. This mirrors Week 1’s immutable `content_hash`, letting auditors jump to exact PDF coordinates for any extracted claim.
