"""Módulo de LLM simulado para PolicyLens.

Proporciona una respuesta determinista cuando no hay una API key de LLM
configurada. Es útil para demos, tests y entornos sin conectividad.
No realiza ninguna llamada a servicios externos.
"""

import re
from typing import List, Tuple


def is_available() -> bool:
    """Indica si el mock está disponible. Siempre devuelve True."""
    return True


def _extract_first_sentences(text: str, max_sentences: int = 3) -> str:
    """Extrae las primeras `max_sentences` oraciones del texto."""
    # Dividir por signos de puntuación seguidos de espacio o final
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = sentences[:max_sentences]
    return ' '.join(selected).strip()


def _parse_sources(context: str) -> List[Tuple[str, str]]:
    """Busca el patrón '[Fuente: X | Página: Y]' en el contexto y devuelve
    una lista de tuplas (fuente, página)."""
    pattern = r'\[Fuente:\s*([^|]+?)\s*\|\s*Página:\s*([^\]]+?)\]'
    matches = re.findall(pattern, context)
    return [(m[0].strip(), m[1].strip()) for m in matches]


def generate_mock_answer(question: str, context: str) -> str:
    """Genera una respuesta simulada a partir de la pregunta y el contexto.

    Extrae las primeras 3 oraciones del contexto, construye una respuesta
    con una plantilla fija y añade al final la lista de fuentes encontradas
    mediante el patrón '[Fuente: X | Página: Y]'.

    Args:
        question: Pregunta del usuario (no se usa en la respuesta, pero se
            mantiene por compatibilidad con la interfaz).
        context: Texto de contexto recuperado de los documentos.

    Returns:
        Respuesta simulada en español.
    """
    extracto = _extract_first_sentences(context, max_sentences=3)
    if not extracto:
        extracto = "No se encontró información relevante en los documentos indexados."

    respuesta = (
        f"Según la documentación indexada: {extracto} "
        "Para mayor detalle, consulte los documentos citados."
    )

    fuentes = _parse_sources(context)
    if fuentes:
        respuesta += "\n\nFuentes:"
        for fuente, pagina in fuentes:
            respuesta += f"\n- {fuente} (página {pagina})"

    return respuesta
