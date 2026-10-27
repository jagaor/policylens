from __future__ import annotations

import os
import tempfile
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import ingestor, retriever

# Aplicación FastAPI que expone el motor RAG de PolicyLens por HTTP.
app = FastAPI(
    title="PolicyLens",
    version="0.1.0",
    description="Motor RAG para consultas de cumplimiento IAM con citas exactas.",
)


class QueryRequest(BaseModel):
    """Petición de consulta al motor RAG."""

    question: str = Field(..., min_length=1, description="Pregunta de cumplimiento IAM")
    top_k: int = Field(3, ge=1, le=20, description="Número de chunks a recuperar")


class ChunkCitation(BaseModel):
    """Fragmento de política devuelto como evidencia."""

    source: str
    page: int | None = None
    text: str
    score: float | None = None


class QueryResponse(BaseModel):
    """Respuesta del motor RAG con la contestación y sus fuentes."""

    question: str
    answer: str
    chunks: List[ChunkCitation]
    sources: List[str]


class IngestRequest(BaseModel):
    """Petición de ingesta de un documento en texto plano."""

    content: str = Field(..., min_length=1, description="Contenido textual del documento")
    filename: str = Field(..., min_length=1, description="Nombre lógico del documento")


class IngestResponse(BaseModel):
    """Resumen del resultado de la ingesta."""

    docs_ingested: int
    chunks_created: int


class DocInfo(BaseModel):
    """Información de un documento indexado."""

    doc_id: str
    source: str
    chunk_count: int


@app.get("/")
def root() -> dict:
    """Comprobación de estado del servicio."""
    return {"name": "PolicyLens", "version": "0.1.0", "status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    """Responde una pregunta de cumplimiento citando los párrafos exactos."""
    try:
        result = retriever.answer(payload.question, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001 - se propaga como error HTTP controlado
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {exc}") from exc

    chunks = [
        ChunkCitation(
            source=c.get("source", "desconocido"),
            page=c.get("page"),
            text=c.get("text", ""),
            score=c.get("score"),
        )
        for c in result.get("chunks", [])
    ]
    sources = sorted({c.source for c in chunks})
    return QueryResponse(
        question=payload.question,
        answer=result.get("answer", ""),
        chunks=chunks,
        sources=sources,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    """Ingesta un documento recibido como texto plano."""
    suffix = os.path.splitext(payload.filename)[1] or ".txt"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=suffix,
            prefix="policylens_",
            delete=False,
        ) as tmp:
            tmp.write(payload.content)
            tmp_path = tmp.name

        stats = ingestor.ingest_path(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error durante la ingesta: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return IngestResponse(
        docs_ingested=int(stats.get("docs_ingested", 1)),
        chunks_created=int(stats.get("chunks_created", 0)),
    )


@app.get("/docs", response_model=List[DocInfo])
def list_docs() -> List[DocInfo]:
    """Lista los documentos actualmente indexados en el vectorstore."""
    try:
        docs = retriever.list_documents()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error al listar documentos: {exc}") from exc

    return [
        DocInfo(
            doc_id=d.get("doc_id", d.get("source", "")),
            source=d.get("source", ""),
            chunk_count=int(d.get("chunk_count", 0)),
        )
        for d in docs
    ]


@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: str) -> dict:
    """Eliminación selectiva de documentos (planificada para v0.2)."""
    raise HTTPException(
        status_code=501,
        detail=(
            f"La eliminación selectiva del documento '{doc_id}' está planificada "
            "para la versión v0.2. Por ahora, use el comando 'clear' del CLI "
            "para vaciar el vectorstore completo."
        ),
    )


def start() -> None:
    """Arranca el servidor Uvicorn en el puerto 8000."""
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
