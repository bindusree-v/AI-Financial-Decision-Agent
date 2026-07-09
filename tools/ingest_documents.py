"""
Document Ingestion Utility

Ingest financial documents (PDFs, text files) into the sector-specific
RAG vector stores. Run this script to populate the knowledge base before
running research queries.

Usage:
    python tools/ingest_documents.py --sector IT --path ./documents/tcs_annual_report.pdf
    python tools/ingest_documents.py --sector Pharma --path ./documents/ --recursive
    python tools/ingest_documents.py --list  # Show collection stats
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from tools.rag import get_rag_tool

console = Console()
logging.basicConfig(level=logging.INFO)


def ingest_file(sector: str, file_path: Path, metadata: dict | None = None) -> None:
    """Ingest a single file into the sector RAG store."""
    rag = get_rag_tool(sector)
    meta = metadata or {}

    if file_path.suffix.lower() == ".pdf":
        console.print(f"  [cyan]Ingesting PDF:[/cyan] {file_path.name}")
        count = rag.ingest_pdf(str(file_path), metadata=meta)
    elif file_path.suffix.lower() in (".txt", ".md"):
        console.print(f"  [cyan]Ingesting text:[/cyan] {file_path.name}")
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        count = rag.ingest_text(text, source=str(file_path), metadata=meta)
    else:
        console.print(f"  [yellow]Skipping unsupported file:[/yellow] {file_path.name}")
        return

    console.print(f"  [green]✓[/green] Added {count} chunks from {file_path.name}")


def ingest_directory(sector: str, dir_path: Path, recursive: bool = False) -> None:
    """Ingest all supported files in a directory."""
    pattern = "**/*" if recursive else "*"
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in dir_path.glob(pattern) if f.suffix.lower() in supported and f.is_file()]

    if not files:
        console.print(f"[yellow]No supported files found in {dir_path}[/yellow]")
        return

    console.print(f"Found {len(files)} files to ingest into [{sector}] collection.")
    for f in files:
        ingest_file(sector, f)


def show_stats() -> None:
    """Display stats for all RAG collections."""
    table = Table(title="RAG Collection Statistics")
    table.add_column("Sector", style="cyan")
    table.add_column("Collection", style="white")
    table.add_column("Documents", style="green")
    table.add_column("Persist Dir", style="dim")

    for sector in ["it", "pharma", "general"]:
        rag = get_rag_tool(sector)
        stats = rag.collection_stats()
        table.add_row(
            sector.upper(),
            stats["collection"],
            str(stats["document_count"]),
            stats["persist_dir"],
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest financial documents into RAG")
    parser.add_argument("--sector", choices=["IT", "Pharma", "general"], help="Target sector")
    parser.add_argument("--path", type=str, help="File or directory path to ingest")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directory")
    parser.add_argument("--list", action="store_true", help="Show collection stats")
    parser.add_argument("--company", type=str, help="Company name metadata tag")
    parser.add_argument("--year", type=str, help="Year metadata tag (e.g., 2024)")
    args = parser.parse_args()

    if args.list:
        show_stats()
        return

    if not args.sector or not args.path:
        parser.print_help()
        sys.exit(1)

    target = Path(args.path)
    metadata: dict = {}
    if args.company:
        metadata["company"] = args.company
    if args.year:
        metadata["year"] = args.year

    if target.is_file():
        ingest_file(args.sector, target, metadata)
    elif target.is_dir():
        ingest_directory(args.sector, target, args.recursive)
    else:
        console.print(f"[red]Path not found: {target}[/red]")
        sys.exit(1)

    console.print()
    show_stats()


if __name__ == "__main__":
    main()
