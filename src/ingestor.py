import hashlib
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from . import embedder


def _sha8(path: str) -> str:
    """Devuelve los primeros 8 caracteres del SHA-256 del path normalizado."""
    normalized = str(Path(path).resolve())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]


def _load_file(filepath: Path) -> List[Document]:
    """Carga un fichero y devuelve una lista de Documents con metadatos.

    Soporta PDF (extrae texto página a página) y TXT/MD (lectura directa).
    """
    suffix = filepath.suffix.lower()
    doc_id = _sha8(str(filepath))
    source = filepath.name

    if suffix == ".pdf":
        documents = []
        with fitz.open(filepath) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()
                if not text.strip():
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "page": page_num, "doc_id": doc_id},
                    )
                )
        return documents

    if suffix in (".txt", ".md"):
        text = filepath.read_text(encoding="utf-8")
        return [
            Document(
                page_content=text,
                metadata={"source": source, "page": 1, "doc_id": doc_id},
            )
        ]

    raise ValueError(f"Extensión no soportada: {suffix} (fichero: {filepath})")


def ingest_path(path: str) -> dict:
    """Ingesta un fichero o directorio completo y lo indexa en el vectorstore.

    Devuelve un diccionario con el número de documentos y chunks procesados.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"La ruta no existe: {path}")

    if root.is_file():
        files = [root]
    else:
        files = [
            f
            for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md")
        ]

    if not files:
        return {"docs_ingested": 0, "chunks_created": 0}

    all_docs: List[Document] = []
    for filepath in files:
        all_docs.extend(_load_file(filepath))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=64,
    )
    chunks = splitter.split_documents(all_docs)

    if chunks:
        embedder.add_chunks(chunks)

    return {"docs_ingested": len(files), "chunks_created": len(chunks)}
