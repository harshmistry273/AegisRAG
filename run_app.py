"""
AegisRAG Master CLI & Application Launcher
Supports launching the Streamlit Web Dashboard, FastAPI REST Server, CLI Query Engine, or Benchmark Suite.
Cross-platform compatible with Windows cp1252 unicode fallback protection.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_streamlit():
    """Launch Streamlit web application."""
    import subprocess
    app_path = root_dir / "ui" / "streamlit_app.py"
    print(f"\n[AegisRAG] Starting Streamlit Web Dashboard on http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def run_api():
    """Launch FastAPI server with Uvicorn."""
    import uvicorn
    from src.config import settings
    print(f"\n[AegisRAG] Starting FastAPI REST Microservice on http://{settings.HOST}:{settings.PORT} ...")
    print(f"[AegisRAG] Interactive API Docs available at http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run("src.api.app:app", host=settings.HOST, port=settings.PORT, reload=True)


def run_benchmark():
    """Execute evaluation benchmark suite."""
    from rich.console import Console
    from rich.table import Table
    from src.evaluation.benchmark import rag_benchmark
    from src.retrieval.vector_store import vector_store
    from src.retrieval.bm25_retriever import bm25_retriever
    from src.ingestion.pipeline import ingestion_pipeline
    from src.core.default_data import DEFAULT_KNOWLEDGE

    console = Console(safe_box=True)
    console.print("\n[bold cyan]>> AegisRAG Evaluation & Benchmark Runner[/bold cyan]")

    # Ensure sample docs are loaded if empty
    if len(vector_store.documents) == 0:
        console.print("[yellow]Indexing default technical knowledge base...[/yellow]")
        for item in DEFAULT_KNOWLEDGE:
            chunks = ingestion_pipeline.process_raw_text(item["text"], source_name=item["source"])
            vector_store.add_documents(chunks)
        bm25_retriever.index_documents(vector_store.documents)

    with console.status("[bold green]Executing comparative evaluations on Groq LPU...[/bold green]"):
        results = rag_benchmark.run_comparative_benchmark()

    # Render summary table
    table = Table(title="AegisRAG vs Naive RAG Benchmark Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Naive Baseline RAG", justify="center")
    table.add_column("AegisRAG (Advanced CRAG)", justify="center")
    table.add_column("Relative Gain", justify="center", style="bold green")

    table.add_row(
        "Faithfulness / Groundedness",
        f"{round(results['naive_rag']['avg_faithfulness'] * 100, 1)}%",
        f"{round(results['advanced_crag']['avg_faithfulness'] * 100, 1)}%",
        f"+{results['improvements']['faithfulness_gain']}%"
    )
    table.add_row(
        "Context Relevance",
        f"{round(results['naive_rag']['avg_relevance'] * 100, 1)}%",
        f"{round(results['advanced_crag']['avg_relevance'] * 100, 1)}%",
        f"+{results['improvements']['relevance_gain']}%"
    )
    table.add_row(
        "End-to-End Latency",
        f"{results['naive_rag']['avg_latency_ms']} ms",
        f"{results['advanced_crag']['avg_latency_ms']} ms",
        "(Includes 6 agentic reflection nodes)"
    )

    console.print(table)


def run_cli():
    """Interactive command-line query session."""
    from rich.console import Console
    from rich.panel import Panel
    from src.agentic.crag_workflow import crag_workflow
    from src.retrieval.vector_store import vector_store
    from src.retrieval.bm25_retriever import bm25_retriever
    from src.ingestion.pipeline import ingestion_pipeline
    from src.core.default_data import DEFAULT_KNOWLEDGE

    console = Console(safe_box=True)
    console.print(Panel.fit("[bold cyan]>> AegisRAG Interactive CLI Session[/bold cyan]\nType your question or 'exit' to quit.", border_style="cyan"))

    if len(vector_store.documents) == 0:
        console.print("[yellow]Auto-indexing default technical documents...[/yellow]")
        for item in DEFAULT_KNOWLEDGE:
            chunks = ingestion_pipeline.process_raw_text(item["text"], source_name=item["source"])
            vector_store.add_documents(chunks)
        bm25_retriever.index_documents(vector_store.documents)

    while True:
        try:
            q = input("\nQuestion > ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                break

            console.print("[dim]Executing Corrective RAG pipeline on Groq LPU...[/dim]")
            res = crag_workflow.run(q, mode="advanced")

            console.print(Panel(res["answer"], title="[bold green]Synthesized Answer[/bold green]", border_style="green"))
            console.print(f"[dim]Latency: {res['total_latency_ms']} ms | Faithfulness: {round(res['faithfulness_score']*100, 1)}% | Relevance: {round(res['relevance_score']*100, 1)}%[/dim]")
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AegisRAG Runner")
    parser.add_argument(
        "--mode",
        choices=["streamlit", "api", "cli", "benchmark"],
        default="streamlit",
        help="Execution mode (default: streamlit)"
    )
    args = parser.parse_args()

    if args.mode == "streamlit":
        run_streamlit()
    elif args.mode == "api":
        run_api()
    elif args.mode == "benchmark":
        run_benchmark()
    elif args.mode == "cli":
        run_cli()
