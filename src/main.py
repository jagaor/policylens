import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src import ingestor, retriever
from src.config import Settings

APP_VERSION = "0.1.0"
HEADER = f"PolicyLens v{APP_VERSION} — Motor RAG de Cumplimiento IAM"

app = typer.Typer(help="PolicyLens: consulta tus políticas IAM citando el párrafo exacto.")
console = Console()


def _print_header() -> None:
    """Imprime la cabecera de versión común a todos los comandos."""
    console.print(f"[bold cyan]{HEADER}[/bold cyan]")


def _load_settings() -> Settings:
    """Carga la configuración desde el entorno y el fichero .env."""
    return Settings()


def _ensure_vectorstore(settings: Settings) -> None:
    """Comprueba que el vectorstore existe antes de consultar."""
    store_path = Path(settings.vectorstore_path)
    if not store_path.exists() or not any(store_path.iterdir()):
        console.print(
            "[red]No hay índice vectorial disponible.[/red] "
            "Ejecuta primero: [bold]policylens ingest data/sample_policies[/bold]"
        )
        raise typer.Exit(code=1)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Fichero o directorio con políticas a indexar."),
) -> None:
    """Ingesta y vectoriza documentos PDF, TXT o Markdown."""
    _print_header()
    settings = _load_settings()

    target = Path(path)
    if not target.exists():
        console.print(f"[red]Ruta no encontrada:[/red] {path}")
        raise typer.Exit(code=1)

    console.print(f"Indexando [bold]{target}[/bold] ...")
    try:
        result = ingestor.ingest_path(str(target), settings=settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error durante la ingesta:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Ingesta completada.[/green] "
        f"Documentos: [bold]{result['documents']}[/bold] · "
        f"Chunks: [bold]{result['chunks']}[/bold]"
    )


@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Pregunta de cumplimiento IAM."),
    top_k: int = typer.Option(3, "--top-k", "-k", min=1, max=10, help="Número de chunks a recuperar."),
) -> None:
    """Responde una pregunta citando los párrafos exactos de las políticas."""
    _print_header()
    settings = _load_settings()
    _ensure_vectorstore(settings)

    try:
        result = retriever.answer(question, top_k=top_k, settings=settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error al consultar:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n[bold]Fragmentos recuperados[/bold]")
    for idx, chunk in enumerate(result.get("chunks", []), start=1):
        page = chunk.get("page")
        page_txt = f"pág. {page}" if page is not None else "pág. n/d"
        score = chunk.get("score", 0.0)
        console.print(
            f"  [cyan]{idx}.[/cyan] {chunk.get('doc_name', 'desconocido')} "
            f"({page_txt}) · score={score:.3f}"
        )

    console.print("\n[bold]Respuesta[/bold]")
    console.print(result.get("answer", "(sin respuesta)"))

    sources = result.get("sources", [])
    if sources:
        console.print("\n[bold]Fuentes[/bold]")
        for source in sources:
            pages = ", ".join(str(p) for p in source.get("pages", []))
            console.print(f"  FUENTE: {source['doc_name']} (pp. {pages})")


@app.command("list-docs")
def list_docs() -> None:
    """Lista los documentos indexados con su número de chunks y fecha."""
    _print_header()
    settings = _load_settings()
    _ensure_vectorstore(settings)

    try:
        documents = retriever.list_documents(settings=settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error al listar documentos:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not documents:
        console.print("[yellow]No hay documentos indexados.[/yellow]")
        return

    table = Table(title="Documentos indexados")
    table.add_column("doc_id", style="cyan", no_wrap=True)
    table.add_column("nombre")
    table.add_column("chunks", justify="right")
    table.add_column("fecha de ingesta")

    for doc in documents:
        table.add_row(
            str(doc.get("doc_id", "")),
            str(doc.get("doc_name", "")),
            str(doc.get("chunks", 0)),
            str(doc.get("ingested_at", "")),
        )

    console.print(table)


@app.command()
def clear() -> None:
    """Elimina todo el índice vectorial tras confirmación interactiva."""
    _print_header()
    settings = _load_settings()

    confirmed = typer.confirm(
        "Esto borrará todo el índice vectorial. ¿Continuar?", default=False
    )
    if not confirmed:
        console.print("[yellow]Operación cancelada.[/yellow]")
        raise typer.Exit(code=0)

    try:
        retriever.clear_store(settings=settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error al limpiar el índice:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[green]Índice vectorial eliminado.[/green]")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrumpido por el usuario.[/yellow]")
        sys.exit(130)
