# Document Intelligence Refinery 🏭

Document Intelligence Refinery is a 5-stage PDF intelligence pipeline that turns heterogeneous documents into:

- structured text, tables, and figures
- logical document units (LDUs)
- a navigable `PageIndex`
- a searchable vector store
- a provenance-aware fact table for natural-language and SQL querying

It is built for assignment-style demos and audit-friendly document QA: every answer can point back to a source page, content hash, and spatial anchor when available.

## What it does

The pipeline processes PDF documents through five stages:

```text
PDF
  → Stage 1: Triage
  → Stage 2: Extraction
  → Stage 3: Chunking
  → Stage 4: PageIndex
  → Stage 5: Query / Audit
```

### Stage 1 — Triage

Classifies each document into:

- `origin_type`: `native_digital`, `mixed`, `scanned_image`, `form_fillable`
- `layout_complexity`: `single_column`, `multi_column`, `table_heavy`, `figure_heavy`, `mixed`
- `domain_hint`: `financial`, `legal`, `technical`, `medical`, `general`

### Stage 2 — Extraction router

Selects one of three extraction strategies:

| Strategy | Use case | Backend |
|---|---|---|
| A — Fast text | clean digital PDFs | `pdfplumber` |
| B — Layout-aware | multi-column, table-heavy, mixed layouts | `Docling` |
| C — Vision | scanned/image-heavy documents | OpenAI / OpenRouter vision |

### Stage 3 — Chunking

Builds provenance-preserving LDUs with rules such as:

- keep tables atomic
- preserve list structure
- carry parent headers into child chunks
- keep chunk hashes spatially anchored

### Stage 4 — PageIndex

Builds a hierarchical, section-aware navigation tree for targeted retrieval.

### Stage 5 — Query and audit

Supports:

- natural-language QA
- provenance output
- fact-first answering for numeric questions
- SQL queries over extracted facts
- claim verification in audit mode

## Key features

- **Agentic routing** based on PDF characteristics
- **Local-first extraction** for most digital documents
- **Budget-capped vision fallback** for scans
- **Vector search + fact table** working together
- **Page-aware provenance** with page number, LDU id, and content hash
- **CLI workflow** for triage, extraction, chunking, indexing, ingest, and querying

## Quick start

### 1. Prerequisites

- Python 3.11+
- Linux/macOS/WSL recommended

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Then edit `.env` as needed.

#### Common environment options

```dotenv
# Optional: enables Gemini-based summaries / LLM synthesis
GEMINI_API_KEY=

# Optional: enables OpenAI vision for scanned PDFs
OPENAI_API_KEY=
OPENAI_PROJECT=

# Optional: enables OpenRouter vision fallback
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Notes:

- Most **digital** PDFs work without any paid API key.
- **Scanned** PDFs need either `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
- If no LLM key is present, the query agent falls back to deterministic retrieval.

### 5. Check the CLI

```bash
python -m src.main --help
```

## CLI usage

All document names can be given either as:

- a filename that exists in `data/`
- or a direct path to a PDF

### Triage a document

```bash
python -m src.main triage "CBE ANNUAL REPORT 2023-24.pdf"
```

### Extract content

```bash
python -m src.main extract "CBE ANNUAL REPORT 2023-24.pdf"
python -m src.main extract "CBE ANNUAL REPORT 2023-24.pdf" --json
```

### Chunk into LDUs

```bash
python -m src.main chunk "CBE ANNUAL REPORT 2023-24.pdf"
```

### Build the PageIndex

```bash
python -m src.main index "CBE ANNUAL REPORT 2023-24.pdf"
```

### Run the full pipeline

```bash
python -m src.main ingest "CBE ANNUAL REPORT 2023-24.pdf"
```

The ingest command runs:

1. triage
2. extraction
3. chunking
4. PageIndex building
5. vector ingestion
6. fact extraction

### Ask natural-language questions

```bash
python -m src.main query "What is the total comprehensive income for FY 2024?"
```

Restrict to a specific document:

```bash
python -m src.main query "What is the main purpose of the assessment?" --doc-id 2b7cf3753e01ab50
```

### Run SQL over the fact table

```bash
python -m src.main query "SELECT label, value, unit, page_number FROM facts LIMIT 10"
```

### Audit a claim

```bash
python -m src.main query "Revenue was 4.2 billion ETB" --audit
```

### Process multiple PDFs

```bash
python -m src.main process-corpus --max 12
```

## Recommended demo flow

For a clean rubric/demo walkthrough:

### Example 1 — annual report

```bash
python -m src.main ingest "CBE ANNUAL REPORT 2023-24.pdf"
python -m src.main query "What is the total comprehensive income for FY 2024?"
```

### Example 2 — FTA report

```bash
python -m src.main ingest "fta_performance_survey_final_report_2022.pdf"
python -m src.main query "What are the main implementation challenges of FTA initiatives identified in the report?" --doc-id 2b7cf3753e01ab50
```

### Example 3 — SQL-backed fact inspection

```bash
python -m src.main query "SELECT doc_name, label, value, unit, page_number FROM facts ORDER BY rowid DESC LIMIT 10"
```

## Output artifacts

The pipeline writes outputs into `.refinery/`:

```text
.refinery/
├── profiles/               # Stage 1 DocumentProfile JSON files
├── pageindex/              # Stage 4 PageIndex JSON files
├── chromadb/               # Vector store persistence
├── extraction_ledger.jsonl # Extraction metadata log
└── fact_table.db           # SQLite fact table
```

Important files:

- `.refinery/profiles/<doc_id>.json` — triage result
- `.refinery/pageindex/<doc_id>.json` — hierarchical index
- `.refinery/fact_table.db` — structured facts for SQL and fact-first QA

## How routing works

Routing rules live in:

- `rubric/extraction_rules.yaml`

Typical behavior:

- `native_digital + single_column` → Strategy A
- `native_digital + multi_column/table_heavy/figure_heavy` → Strategy B
- `mixed` → Strategy B
- `scanned_image` → Strategy C

This file is the main control surface for tuning behavior without changing core code.

## Project structure

```text
week3-document-intelligence-refinery/
├── src/
│   ├── agents/
│   │   ├── triage.py
│   │   ├── extractor.py
│   │   ├── chunker.py
│   │   ├── indexer.py
│   │   └── query_agent.py
│   ├── strategies/
│   │   ├── fast_text.py
│   │   ├── layout_extractor.py
│   │   └── vision_extractor.py
│   ├── data/
│   │   ├── vector_store.py
│   │   ├── fact_table.py
│   │   └── audit.py
│   ├── models/
│   └── main.py
├── data/
├── rubric/
│   └── extraction_rules.yaml
├── tests/
├── DOMAIN_NOTES.md
├── README.md
└── pyproject.toml
```

## Testing

Run unit tests:

```bash
pytest tests -v
```

If you want a quick smoke test:

```bash
python -m src.main triage "CBE ANNUAL REPORT 2023-24.pdf"
python -m src.main ingest "CBE ANNUAL REPORT 2023-24.pdf"
```

## Troubleshooting

### 1. Query answer looks too vague

- make sure the document was ingested first
- pass `--doc-id` to scope retrieval
- inspect `.refinery/pageindex/<doc_id>.json` if section ranges look odd

### 2. Scanned PDF extraction fails

Check that at least one of these is configured:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

Also note:

- OpenRouter may fail if the account has no credits
- OpenAI vision calls may require a valid project/key combination

### 3. Docling / OCR logs appear during extraction

Those are usually informational, not failures.

### 4. Start clean

To reset local artifacts:

```bash
rm -rf .refinery
```

Then rerun `ingest` on the documents you want.

## Notes

- `DOMAIN_NOTES.md` documents the current heuristics, routing logic, and cost guardrails.
- Provenance is a first-class output: answers should be traceable back to source pages.
- The repository is tuned for document-intelligence coursework, but the pipeline structure is reusable.

## License

See `LICENSE`.