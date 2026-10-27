import threading
from typing import Any

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from src.config import settings

# Nombre de la colección única del proyecto
COLLECTION_NAME = "policylens"

# Modelo de embeddings cargado una sola vez (singleton perezoso y thread-safe)
_model_lock = threading.Lock()
_model_instance: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Devuelve la instancia única del modelo SentenceTransformer.

    La carga se hace de forma perezosa para no penalizar el arranque del CLI
    ni los tests que no necesiten generar embeddings.
    """
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model_instance


def _get_client() -> chromadb.PersistentClient:
    """Crea un cliente persistente de ChromaDB apuntando al directorio configurado."""
    settings.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.VECTORSTORE_PATH))


def get_collection() -> chromadb.Collection:
    """Devuelve la colección 'policylens', creándola si no existe todavía."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Document]) -> int:
    """Vectoriza y almacena una lista de chunks en la colección.

    Cada Document debe traer en `metadata` al menos las claves `doc_id`,
    `source` y `page`. Devuelve el número de chunks efectivamente añadidos.
    """
    if not chunks:
        return 0

    model = _get_model()
    collection = get_collection()

    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for idx, chunk in enumerate(chunks):
        texto = chunk.page_content.strip()
        if not texto:
            continue

        vector = model.encode(texto, normalize_embeddings=True).tolist()
        meta = dict(chunk.metadata or {})

        doc_id = str(meta.get("doc_id", "desconocido"))
        page = int(meta.get("page", 0))
        source = str(meta.get("source", "desconocido"))

        embeddings.append(vector)
        documents.append(texto)
        metadatas.append(
            {
                "doc_id": doc_id,
                "source": source,
                "page": page,
                "chunk_index": idx,
                "ingested_at": str(meta.get("ingested_at", "")),
            }
        )
        ids.append(f"{doc_id}:{page}:{idx}")

    if not ids:
        return 0

    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return len(ids)


def query_chunks(query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Recupera los Top-K chunks más similares a un embedding de consulta.

    Devuelve una lista de diccionarios con las claves `text`, `source`,
    `page`, `doc_id` y `score` (1 - distancia coseno).
    """
    collection = get_collection()
    resultado = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],
    )

    docs = (resultado.get("documents") or [[]])[0]
    metas = (resultado.get("metadatas") or [[]])[0]
    dists = (resultado.get("distances") or [[]])[0]

    salida: list[dict[str, Any]] = []
    for texto, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        salida.append(
            {
                "text": texto,
                "source": meta.get("source", "desconocido"),
                "page": int(meta.get("page", 0)),
                "doc_id": meta.get("doc_id", "desconocido"),
                "score": round(1.0 - float(dist), 4),
            }
        )
    return salida


def list_all_docs() -> list[dict[str, Any]]:
    """Devuelve un resumen de los documentos indexados, agrupados por doc_id."""
    collection = get_collection()
    datos = collection.get(include=["metadatas"])
    metadatas = datos.get("metadatas") or []

    resumen: dict[str, dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue
        doc_id = str(meta.get("doc_id", "desconocido"))
        entrada = resumen.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "source": meta.get("source", "desconocido"),
                "chunk_count": 0,
                "ingested_at": meta.get("ingested_at", ""),
            },
        )
        entrada["chunk_count"] += 1
        if meta.get("ingested_at") and not entrada["ingested_at"]:
            entrada["ingested_at"] = meta["ingested_at"]

    return sorted(resumen.values(), key=lambda d: d["doc_id"])


def clear_all() -> None:
    """Elimina la colección completa y la vuelve a crear vacía."""
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Si la colección no existía, no hay nada que borrar
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
