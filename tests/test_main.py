import os
import sys
from pathlib import Path

import pytest

# Aseguramos que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src import ingestor, mock_llm, embedder
from langchain.schema import Document


@pytest.fixture
def tmp_policy_file(tmp_path):
    """Crea un fichero TXT temporal con una política IAM de muestra."""
    contenido = (
        "POL-IAM-002: Política de Gestión de Identidades y Accesos\n"
        "1. Principio de mínimo privilegio: los usuarios solo deben tener los permisos\n"
        "   estrictamente necesarios para realizar sus funciones.\n"
        "2. Revisión periódica de accesos: se realizarán revisiones trimestrales de\n"
        "   los roles y permisos asignados.\n"
        "3. Separación de funciones: ninguna persona debe poder ejecutar todas las\n"
        "   fases de un proceso crítico.\n"
        "4. Autenticación multifactor: obligatoria para accesos remotos y cuentas\n"
        "   privilegiadas.\n"
        "5. Registro de auditoría: todas las acciones administrativas deben quedar\n"
        "   registradas.\n"
    )
    archivo = tmp_path / "POL-IAM-002.txt"
    archivo.write_text(contenido, encoding="utf-8")
    return archivo


def test_ingest_txt_file(tmp_policy_file):
    """Verifica que la ingesta de un fichero TXT produce al menos un documento y un chunk."""
    resultado = ingestor.ingest_path(str(tmp_policy_file))
    assert resultado["docs_ingested"] >= 1
    assert resultado["chunks_created"] >= 1


def test_mock_llm_generates_text():
    """Comprueba que el mock LLM devuelve una respuesta no vacía con la fuente citada."""
    contexto = "El mínimo privilegio implica [Fuente: POL-IAM-002.txt | Página: 1]"
    respuesta = mock_llm.generate_mock_answer("¿Qué es mínimo privilegio?", contexto)
    assert isinstance(respuesta, str)
    assert len(respuesta) > 0
    assert "POL-IAM-002" in respuesta


def test_config_defaults():
    """Verifica los valores por defecto de configuración."""
    settings = Settings()
    assert settings.CHUNK_SIZE == 512
    assert settings.TOP_K == 3


def test_embedder_add_and_query(tmp_path, monkeypatch):
    """Añade un chunk de prueba, genera embedding y consulta el vectorstore aislado."""
    # Aislamos el vectorstore en un directorio temporal
    vectorstore_path = tmp_path / "vectorstore"
    monkeypatch.setattr("src.config.settings.VECTORSTORE_PATH", str(vectorstore_path))

    # Creamos un documento de prueba con metadatos
    doc = Document(
        page_content="El principio de mínimo privilegio exige otorgar solo los permisos necesarios.",
        metadata={"source": "POL-IAM-002.txt", "page": 1, "section": "Principios"},
    )

    # Añadimos el chunk al vectorstore (esto crea la colección si no existe)
    embedder.add_chunks([doc])

    # Generamos el embedding de la consulta y recuperamos resultados
    resultados = embedder.query_chunks("¿Qué es mínimo privilegio?", top_k=1)

    assert len(resultados) >= 1
    primer_resultado = resultados[0]
    assert "content" in primer_resultado
    assert "metadata" in primer_resultado
    assert "score" in primer_resultado
