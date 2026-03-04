# Document Intelligence Refinery 🏭

> **Production-grade, 5-stage agentic pipeline for enterprise document intelligence**

## Overview

The Document Intelligence Refinery transforms heterogeneous documents (PDFs, scanned images, financial reports) into structured, spatially-indexed, provenance-tracked knowledge — queryable via natural language.

**Architecture:**
```
Document → Triage Agent → Extraction Router → Chunking Engine → PageIndex Builder → Query Agent
             (Stage 1)        (Stage 2)           (Stage 3)         (Stage 4)         (Stage 5)
```

**Three Extraction Strategies:**
| Strategy | Tool | Trigger | Cost |
|---|---|---|---|
| A — Fast Text | pdfplumber | native_digital + single_column | Free |
| B — Layout-Aware | Docling | multi_column / table_heavy / mixed | Free (local) |
| C — Vision | Gemini 1.5 Flash | scanned_image / low confidence | ~$0.0004/page |

---

## Setup (< 10 minutes)

### 1. Prerequisites
```bash
# Python 3.11+
python --version

# uv (fast package installer)
pip install uv
```

### 2. Install Dependencies
```bash
cd /path/to/week3-document-intelligence-refinery
uv pip install -e ".[dev]" --python 3.11
```

### 3. Configure API Keys
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Verify Installation
```bash
python -m pytest tests/ -v
python -m src.main --help
```

---

## Usage

### Single Document Pipeline
```bash
# Stage 1: Classify document
python -m src.main triage "CBE ANNUAL REPORT 2023-24.pdf"

# Stage 2: Extract content
python -m src.main extract "CBE ANNUAL REPORT 2023-24.pdf"

# Stage 3: Chunk into LDUs
python -m src.main chunk "CBE ANNUAL REPORT 2023-24.pdf"

# Stage 4: Build PageIndex
python -m src.main index "CBE ANNUAL REPORT 2023-24.pdf"

# Full pipeline (all stages including vector store + fact table)
python -m src.main ingest "CBE ANNUAL REPORT 2023-24.pdf"
```

### Querying
```bash
# Ask a natural language question
python -m src.main query "What was the total loan disbursement in 2024?" --doc-id <doc_id>

# Audit mode: verify a claim
python -m src.main query "Revenue was 4.2 billion ETB" --audit
```

### Bulk Processing (Corpus)
```bash
# Process up to 12 documents from the data/ directory
python -m src.main process-corpus --max 12
```

---

## Project Structure

```
week3-document-intelligence-refinery/
├── src/
│   ├── models/              # Pydantic schemas
│   │   ├── document_profile.py   # DocumentProfile
│   │   ├── extracted_document.py # ExtractedDocument + BBox
│   │   ├── ldu.py               # Logical Document Unit
│   │   ├── page_index.py        # PageIndex + Section
│   │   └── provenance.py        # ProvenanceChain
│   ├── agents/              # Pipeline stages
│   │   ├── triage.py        # Stage 1: Triage Agent
│   │   ├── extractor.py     # Stage 2: ExtractionRouter
│   │   ├── chunker.py       # Stage 3: ChunkingEngine
│   │   ├── indexer.py       # Stage 4: PageIndexBuilder
│   │   └── query_agent.py   # Stage 5: QueryAgent
│   ├── strategies/          # Extraction strategies
│   │   ├── fast_text.py     # Strategy A
│   │   ├── layout_extractor.py  # Strategy B
│   │   └── vision_extractor.py  # Strategy C
│   ├── data/                # Data layer
│   │   ├── vector_store.py  # ChromaDB
│   │   ├── fact_table.py    # SQLite FactTable
│   │   └── audit.py         # AuditMode
│   └── main.py              # CLI entry point
├── tests/
│   ├── test_triage.py
│   ├── test_confidence_scoring.py
│   └── test_chunking.py
├── rubric/
│   └── extraction_rules.yaml  # Externalized config
├── data/                    # 50 corpus documents
├── .refinery/               # Pipeline artifacts
│   ├── profiles/            # DocumentProfile JSONs
│   ├── pageindex/           # PageIndex JSONs
│   └── extraction_ledger.jsonl
├── DOMAIN_NOTES.md          # Phase 0 deliverable
└── Dockerfile
```

---

## Configuration

All thresholds and routing rules are in `rubric/extraction_rules.yaml`. **A new document type can be onboarded by modifying only this file — no code changes required.**

Key settings:
```yaml
confidence_thresholds:
  fast_text_min: 0.50    # Escalate A→B below this
  layout_min: 0.40       # Escalate B→C below this

budget:
  max_cost_per_doc_usd: 0.10   # Hard limit for Vision strategy
```

---

## Docker

```bash
docker build -t document-refinery .
docker run --env GEMINI_API_KEY=your_key \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/.refinery:/app/.refinery \
           document-refinery \
           python -m src.main ingest "CBE ANNUAL REPORT 2023-24.pdf"
```

---

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Cost Analysis

| Strategy | Tool | Cost/page | Cost/100-page doc |
|---|---|---|---|
| A — Fast Text | pdfplumber | $0 | $0 |
| B — Layout | Docling | $0 | $0 |
| C — Vision | Gemini 1.5 Flash | ~$0.0004 | ~$0.04 |

**Budget guard** caps Vision strategy at $0.10/document (configurable).
