import os
from typing import Any

from src import embedder
from src.config import settings


SYSTEM_PROMPT = (
    "Eres un asistente experto en cumplimiento de políticas de seguridad IAM y PAM. "
    "Responde a la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado. "
    "Si el contexto no contiene información suficiente para responder, indícalo explícitamente. "
    "Cita siempre las fuentes usando el formato [fuente, página N] al final de cada afirmación relevante. "
    "No inventes información que no aparezca en el contexto."
)


def _build_context(chunks: list[dict]) -> str:
    """Ensambla el contexto concatenando los chunks con su fuente y página."""
    partes: list[str] = []
    for chunk in chunks:
        source = chunk.get("source", "desconocido")
        page = chunk.get("page", "N/A")
        text = chunk.get("text", "").strip()
        partes.append(f"[Fuente: {source} | Página: {page}]\n{text}")
    return "\n\n".join(partes)


def _call_llm(question: str, context: str) -> str:
    """Invoca al LLM configurado o al mock determinista si no hay clave."""
    api_key = settings.OPENAI_API_KEY
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=settings.OPENAI_BASE_URL)
        completion = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta: {question}",
                },
            ],
            temperature=0.0,
        )
        return completion.choices[0].message.content or ""

    from src import mock_llm

    return mock_llm.generate_mock_answer(question, context)


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    """Agrupa las páginas por documento fuente, deduplicando."""
    agrupado: dict[str, list[Any]] = {}
    for chunk in chunks:
        source = chunk.get("source", "desconocido")
        page = chunk.get("page")
        agrupado.setdefault(source, [])
        if page is not None and page not in agrupado[source]:
            agrupado[source].append(page)
    return [
        {"source": source, "pages": sorted(pages, key=lambda p: (p is None, p))}
        for source, pages in agrupado.items()
    ]


def answer(question: str, top_k: int = 3) -> dict:
    """Responde a una pregunta de cumplimiento citando los chunks recuperados."""
    embedding = embedder._model.encode(question).tolist()
    chunks = embedder.query_chunks(embedding, top_k)
    context = _build_context(chunks)
    respuesta = _call_llm(question, context)
    return {
        "question": question,
        "answer": respuesta,
        "chunks": chunks,
        "sources": _dedupe_sources(chunks),
    }


def list_documents() -> list[dict]:
    """Devuelve la lista de documentos indexados en el vectorstore."""
    return embedder.list_all_docs()


def clear_store() -> None:
    """Elimina todo el contenido del vectorstore persistente."""
    embedder.clear_all()
