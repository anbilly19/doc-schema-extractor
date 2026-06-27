"""CLI interface using Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="dse",
    help="doc-schema-extractor: Template-guided PDF/XLSX extraction pipeline",
    add_completion=False,
)
templates_app = typer.Typer(help="Manage stored templates")
app.add_typer(templates_app, name="templates")

console = Console()


def _get_backend(backend: str, model: str | None):
    from doc_schema_extractor.backends import OllamaBackend, OpenAIBackend
    import os

    if backend == "openai":
        m = model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        return OpenAIBackend(model=m)
    else:
        m = model or os.getenv("OLLAMA_DEFAULT_MODEL", "gemma4:e4b-it-qat")
        return OllamaBackend(model=m)


@app.command()
def extract(
    file: Path = typer.Argument(..., help="PDF or XLSX file to extract"),
    backend: str = typer.Option("ollama", help="LLM backend: ollama or openai"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    store: Path = typer.Option(Path("templates/store.json"), help="Template store path"),
    output: Optional[Path] = typer.Option(None, help="Save result to JSON file"),
    threshold: float = typer.Option(0.75, help="Template match threshold"),
):
    """Extract structured data from a PDF or XLSX file."""
    from doc_schema_extractor import Extractor

    llm_backend = _get_backend(backend, model)
    extractor = Extractor(backend=llm_backend, store_path=store, match_threshold=threshold)

    with console.status(f"Processing {file.name}..."):
        result = extractor.extract(file)

    # Display result
    status = "[green]HIT[/green]" if not result.llm_used else "[yellow]MISS → LLM[/yellow]"
    console.print(f"\n[bold]Result:[/bold] {status}")
    console.print(f"Template: [cyan]{result.template_id}[/cyan]")
    console.print(f"Match score: {result.match_score:.2f}")
    if result.llm_used:
        console.print(f"LLM: {result.llm_backend}/{result.llm_model}")
    console.print(f"Validation: {'[green]PASSED[/green]' if result.validation_passed else '[red]FAILED[/red]'}")
    if result.validation_errors:
        for err in result.validation_errors:
            console.print(f"  [red]• {err}[/red]")

    console.print("\n[bold]Extracted data:[/bold]")
    rprint(result.data)

    if output:
        output.write_text(result.model_dump_json(indent=2))
        console.print(f"\nSaved to [cyan]{output}[/cyan]")


@app.command()
def batch(
    folder: Path = typer.Argument(..., help="Folder containing PDF/XLSX files"),
    backend: str = typer.Option("ollama", help="LLM backend: ollama or openai"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    store: Path = typer.Option(Path("templates/store.json"), help="Template store path"),
    output: Optional[Path] = typer.Option(None, help="Save results to JSON file"),
    threshold: float = typer.Option(0.75, help="Template match threshold"),
):
    """Batch extract from all PDF/XLSX files in a folder."""
    from doc_schema_extractor import Extractor

    files = list(folder.glob("*.pdf")) + list(folder.glob("*.xlsx"))
    if not files:
        console.print("[red]No PDF or XLSX files found.[/red]")
        raise typer.Exit(1)

    llm_backend = _get_backend(backend, model)
    extractor = Extractor(backend=llm_backend, store_path=store, match_threshold=threshold)

    results = []
    table = Table(title="Batch Extraction Results")
    table.add_column("File", style="cyan")
    table.add_column("Template", style="magenta")
    table.add_column("Score")
    table.add_column("LLM Used")
    table.add_column("Valid")

    for f in files:
        with console.status(f"Processing {f.name}..."):
            result = extractor.extract(f)
        results.append(json.loads(result.model_dump_json()))
        table.add_row(
            f.name,
            result.template_id or "N/A",
            f"{result.match_score:.2f}",
            "Yes" if result.llm_used else "No",
            "✓" if result.validation_passed else "✗",
        )

    console.print(table)

    if output:
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        console.print(f"Saved to [cyan]{output}[/cyan]")


@templates_app.command("list")
def templates_list(
    store: Path = typer.Option(Path("templates/store.json"), help="Template store path"),
):
    """List all stored templates."""
    from doc_schema_extractor.template_store import TemplateStore

    ts = TemplateStore(store)
    all_templates = ts.list_all()

    if not all_templates:
        console.print("[yellow]No templates stored yet.[/yellow]")
        return

    table = Table(title="Stored Templates")
    table.add_column("ID", style="cyan")
    table.add_column("Doc Type")
    table.add_column("Supplier")
    table.add_column("Keywords")
    table.add_column("Hits", justify="right")
    table.add_column("Version", justify="right")

    for t in all_templates:
        table.add_row(
            t.template_id,
            t.fingerprint.doc_type,
            t.fingerprint.supplier_hint,
            ", ".join(t.fingerprint.required_keywords[:3]) + "...",
            str(t.hit_count),
            str(t.version),
        )
    console.print(table)


@templates_app.command("show")
def templates_show(
    template_id: str = typer.Argument(..., help="Template ID to show"),
    store: Path = typer.Option(Path("templates/store.json"), help="Template store path"),
):
    """Show full details of a template."""
    from doc_schema_extractor.template_store import TemplateStore

    ts = TemplateStore(store)
    template = ts.get(template_id)
    if not template:
        console.print(f"[red]Template '{template_id}' not found.[/red]")
        raise typer.Exit(1)
    rprint(json.loads(template.model_dump_json(indent=2)))


@templates_app.command("delete")
def templates_delete(
    template_id: str = typer.Argument(..., help="Template ID to delete"),
    store: Path = typer.Option(Path("templates/store.json"), help="Template store path"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a stored template (forces LLM re-learn on next document)."""
    from doc_schema_extractor.template_store import TemplateStore

    if not confirm:
        typer.confirm(f"Delete template '{template_id}'?", abort=True)

    ts = TemplateStore(store)
    if ts.delete(template_id):
        console.print(f"[green]Deleted template '{template_id}'.[/green]")
    else:
        console.print(f"[red]Template '{template_id}' not found.[/red]")
