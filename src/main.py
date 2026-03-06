#!/usr/bin/env python3
"""
Document Intelligence Refinery — CLI Entry Point
Usage: python -m src.main [COMMAND] [OPTIONS]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(
    name="refinery",
    help="Document Intelligence Refinery — 5-stage agentic pipeline",
    add_completion=False,
)
console = Console()
DATA_DIR = Path("data")

# Load environment variables from .env (for API keys)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass


def _get_data_path(filename: str) -> Path:
    """Resolve document path (absolute or relative to data/)."""
    p = Path(filename)
    if p.exists():
        return p
    data_p = DATA_DIR / filename
    if data_p.exists():
        return data_p
    raise FileNotFoundError(f"Document not found: {filename}")


@app.command()
def triage(
    document: str = typer.Argument(..., help="Document filename or path"),
    output_json: bool = typer.Option(False, "--json", help="Print raw JSON"),
):
    """Stage 1: Classify a document and produce a DocumentProfile."""
    from src.agents.triage import TriageAgent

    doc_path = _get_data_path(document)
    console.print(f"\n[bold cyan]🔍 Triaging:[/bold cyan] {doc_path.name}")

    agent = TriageAgent()
    profile = agent.triage(doc_path)

    if output_json:
        console.print_json(profile.model_dump_json(indent=2))
        return

    t = Table(title=f"DocumentProfile — {profile.filename}", show_header=True)
    t.add_column("Field", style="cyan")
    t.add_column("Value", style="green")

    t.add_row("doc_id", profile.doc_id)
    t.add_row("origin_type", f"[bold]{profile.origin_type}[/bold]")
    t.add_row("layout_complexity", profile.layout_complexity)
    t.add_row("domain_hint", profile.domain_hint)
    t.add_row("estimated_cost", f"[yellow]{profile.estimated_cost}[/yellow]")
    t.add_row("page_count", str(profile.page_count))
    t.add_row("char_density_mean", f"{profile.char_density_mean:.1f}")
    t.add_row("image_ratio_mean", f"{profile.image_ratio_mean:.3f}")
    t.add_row("has_font_metadata", str(profile.has_font_metadata))
    t.add_row("language", profile.language)

    console.print(t)
    console.print(f"[dim]Saved to .refinery/profiles/{profile.doc_id}.json[/dim]")


@app.command()
def extract(
    document: str = typer.Argument(..., help="Document filename or path"),
    output_json: bool = typer.Option(False, "--json", help="Print raw JSON"),
):
    """Stage 2: Extract content using the optimal strategy (with escalation)."""
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter

    doc_path = _get_data_path(document)
    console.print(f"\n[bold cyan]📄 Extracting:[/bold cyan] {doc_path.name}")

    profile = TriageAgent().triage(doc_path)
    console.print(
        f"[dim]Profile: {profile.origin_type} / {profile.layout_complexity} / {profile.domain_hint}[/dim]"
    )

    router = ExtractionRouter()
    doc = router.route(doc_path, profile)

    if output_json:
        console.print_json(doc.model_dump_json(indent=2))
        return

    t = Table(title=f"Extraction Result — {doc.filename}")
    t.add_column("Field", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("strategy_used", f"[bold]{doc.strategy_used}[/bold]")
    t.add_row("confidence_score", f"[{'green' if doc.confidence_score > 0.5 else 'red'}]{doc.confidence_score}[/]")
    t.add_row("page_count", str(doc.page_count))
    t.add_row("text_blocks", str(len(doc.text_blocks)))
    t.add_row("tables_extracted", str(len(doc.tables)))
    t.add_row("figures_extracted", str(len(doc.figures)))
    t.add_row("processing_time", f"{doc.processing_time_sec}s")
    t.add_row("cost_estimate", f"${doc.cost_estimate_usd:.6f}")
    console.print(t)

    if doc.tables:
        console.print(f"\n[bold]📊 First Table (as JSON):[/bold]")
        first_table = doc.tables[0]
        console.print_json(first_table.model_dump_json(indent=2))

    console.print(f"\n[dim]Logged to .refinery/extraction_ledger.jsonl[/dim]")


@app.command()
def chunk(
    document: str = typer.Argument(..., help="Document filename or path"),
    show_sample: int = typer.Option(3, "--sample", help="Number of LDUs to show"),
):
    """Stage 3: Chunk extracted content into Logical Document Units."""
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine

    doc_path = _get_data_path(document)
    console.print(f"\n[bold cyan]✂️  Chunking:[/bold cyan] {doc_path.name}")

    profile = TriageAgent().triage(doc_path)
    router = ExtractionRouter()
    extracted = router.route(doc_path, profile)

    engine = ChunkingEngine()
    ldus = engine.chunk(extracted)

    console.print(f"[green]✓ Generated {len(ldus)} LDUs[/green]")

    # Show type distribution
    from collections import Counter
    types = Counter(l.chunk_type for l in ldus)
    t = Table(title="LDU Type Distribution")
    t.add_column("Type", style="cyan")
    t.add_column("Count", style="green")
    for ctype, count in types.most_common():
        t.add_row(ctype, str(count))
    console.print(t)

    # Show sample LDUs
    for i, ldu in enumerate(ldus[:show_sample]):
        console.print(
            Panel(
                f"[cyan]Type:[/cyan] {ldu.chunk_type}  "
                f"[cyan]Pages:[/cyan] {ldu.page_refs}  "
                f"[cyan]Section:[/cyan] {ldu.parent_section or 'root'}\n\n"
                f"{ldu.content[:300]}{'...' if len(ldu.content) > 300 else ''}",
                title=f"LDU #{i+1} [{ldu.ldu_id}]",
            )
        )


@app.command()
def index(
    document: str = typer.Argument(..., help="Document filename or path"),
    show_tree: bool = typer.Option(True, "--tree/--no-tree"),
):
    """Stage 4: Build the PageIndex tree for the document."""
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine
    from src.agents.indexer import PageIndexBuilder

    doc_path = _get_data_path(document)
    console.print(f"\n[bold cyan]🗂  Indexing:[/bold cyan] {doc_path.name}")

    profile = TriageAgent().triage(doc_path)
    router = ExtractionRouter()
    extracted = router.route(doc_path, profile)
    engine = ChunkingEngine()
    ldus = engine.chunk(extracted)

    builder = PageIndexBuilder()
    page_index = builder.build(extracted, ldus)

    console.print(f"[green]✓ Built PageIndex with {len(page_index.sections)} root sections[/green]")

    if show_tree:
        tree = Tree(f"[bold]{doc_path.name}[/bold] (PageIndex)")
        for section in page_index.sections:
            node = tree.add(
                f"[cyan]{section.title}[/cyan] "
                f"[dim](pp.{section.page_start}–{section.page_end})[/dim]"
            )
            if section.summary:
                node.add(f"[dim italic]{section.summary[:100]}...[/dim italic]")
            for child in section.child_sections:
                child_node = node.add(
                    f"[green]{child.title}[/green] "
                    f"[dim](pp.{child.page_start}–{child.page_end})[/dim]"
                )
        console.print(tree)

    console.print(f"\n[dim]Saved to .refinery/pageindex/{page_index.doc_id}.json[/dim]")


@app.command()
def ingest(
    document: str = typer.Argument(..., help="Document filename or path"),
):
    """Full pipeline: triage → extract → chunk → index → vector store → fact table."""
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine
    from src.agents.indexer import PageIndexBuilder
    from src.data.vector_store import VectorStore
    from src.data.fact_table import FactTable

    doc_path = _get_data_path(document)
    console.print(f"\n[bold magenta]🏭 FULL PIPELINE:[/bold magenta] {doc_path.name}\n")

    with console.status("[cyan]Stage 1: Triaging..."):
        profile = TriageAgent().triage(doc_path)
    console.print(f"[green]✓[/green] Triage: {profile.origin_type} / {profile.domain_hint}")

    with console.status("[cyan]Stage 2: Extracting..."):
        router = ExtractionRouter()
        extracted = router.route(doc_path, profile)
    console.print(
        f"[green]✓[/green] Extraction: {extracted.strategy_used} "
        f"(confidence: {extracted.confidence_score:.2f})"
    )

    with console.status("[cyan]Stage 3: Chunking..."):
        engine = ChunkingEngine()
        ldus = engine.chunk(extracted)
    console.print(f"[green]✓[/green] Chunking: {len(ldus)} LDUs")

    with console.status("[cyan]Stage 4: Indexing..."):
        builder = PageIndexBuilder()
        page_index = builder.build(extracted, ldus)
    console.print(f"[green]✓[/green] PageIndex: {len(page_index.sections)} sections")

    with console.status("[cyan]Ingesting into vector store..."):
        vs = VectorStore()
        ingested = vs.ingest(ldus)
    console.print(f"[green]✓[/green] Vector Store: {ingested} LDUs ingested")

    with console.status("[cyan]Extracting facts..."):
        ft = FactTable()
        facts_count = ft.ingest_ldus(ldus, doc_path.name)
    console.print(f"[green]✓[/green] Fact Table: {facts_count} facts extracted")

    console.print(f"\n[bold green]✅ Pipeline complete for {doc_path.name}[/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Restrict to specific document"),
    audit: bool = typer.Option(False, "--audit", help="Verify as claim instead of answering"),
):
    """Stage 5: Ask a question and get an answer with provenance."""
    from src.agents.query_agent import QueryAgent
    from src.data.fact_table import FactTable

    agent = QueryAgent()

    if audit:
        console.print(f"\n[bold yellow]🔍 Auditing claim:[/bold yellow] {question}\n")
        result = agent.verify_claim(question, doc_id)
        verified = result.get("verified")
        icon = "✅" if verified else "❌"
        console.print(Panel(
            f"{icon} [bold]{'VERIFIED' if verified else 'UNVERIFIABLE'}[/bold]\n\n"
            f"Note: {result.get('audit_note', '')}\n\n"
            f"Citations: {len(result.get('citations', []))}",
            title="Audit Result",
        ))
        for c in result.get("citations", []):
            console.print(f"  📄 {c['document_name']}, page {c['page_number']}")
    else:
        console.print(f"\n[bold cyan]❓ Query:[/bold cyan] {question}\n")

        if question.strip().upper().startswith("SELECT"):
            ft = FactTable()
            try:
                rows = ft.query(question)
            except Exception as e:
                console.print(Panel(str(e), title="SQL Error", style="red"))
                return

            if not rows:
                console.print(Panel("No rows returned.", title="SQL Result"))
                return

            table = Table(title="SQL Result")
            for col in rows[0].keys():
                table.add_column(str(col), style="cyan")
            for row in rows:
                table.add_row(*[str(row.get(col, "")) for col in rows[0].keys()])
            console.print(table)
            return

        result = agent.query(question, doc_id)
        console.print(Panel(result.get("answer", "No answer"), title="Answer"))

        provenance = result.get("provenance", {})
        citations = provenance.get("citations", [])
        if citations:
            console.print(f"\n[bold]📋 ProvenanceChain ({len(citations)} sources):[/bold]")
            for c in citations:
                console.print(
                    f"  • {c['document_name']}, page {c['page_number']} "
                    f"[dim]({c['strategy_used']}, score: {c['confidence_score']:.2f})[/dim]"
                )


@app.command()
def process_corpus(
    class_name: Optional[str] = typer.Option(
        None, "--class", help="Filter by class: A, B, C, D"
    ),
    max_docs: int = typer.Option(12, "--max", help="Maximum documents to process"),
):
    """Process multiple corpus documents for demo preparation."""
    class_map = {
        "A": ["CBE", "Annual_Report", "ETS", "EthSwitch"],
        "B": ["Audit_Report", "Audit Report", "2018_Audited", "2019_Audited", "2020_Audited", "2021_Audited", "2022_Audited"],
        "C": ["fta_performance"],
        "D": ["tax_expenditure", "Consumer_Price"],
    }

    docs = list(DATA_DIR.glob("*.pdf"))[:max_docs]
    console.print(f"[bold]Processing {len(docs)} documents...[/bold]")

    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine
    from src.agents.indexer import PageIndexBuilder
    from src.data.vector_store import VectorStore
    from src.data.fact_table import FactTable

    triage_agent = TriageAgent()
    router = ExtractionRouter()
    chunker = ChunkingEngine()
    indexer = PageIndexBuilder(use_llm=False)  # skip LLM for bulk
    vs = VectorStore()
    ft = FactTable()

    for doc_path in docs:
        try:
            console.print(f"\n[cyan]→ {doc_path.name}[/cyan]")
            profile = triage_agent.triage(doc_path)
            extracted = router.route(doc_path, profile)
            ldus = chunker.chunk(extracted)
            indexer.build(extracted, ldus)
            vs.ingest(ldus)
            ft.ingest_ldus(ldus, doc_path.name)
            console.print(
                f"  [green]✓[/green] {profile.origin_type}/{profile.layout_complexity} "
                f"strategy={extracted.strategy_used} ldus={len(ldus)}"
            )
        except Exception as e:
            console.print(f"  [red]✗[/red] {doc_path.name}: {e}")


if __name__ == "__main__":
    app()
