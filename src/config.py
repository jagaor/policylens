import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración centralizada de PolicyLens.

    Todos los valores pueden sobrescribirse mediante variables de entorno
    o mediante el fichero `.env` situado en la raíz del proyecto.
    """

    # Clave de API del proveedor LLM compatible con OpenAI. Si está vacía,
    # el sistema recurre al LLM determinista de `mock_llm.py`.
    OPENAI_API_KEY: str = Field(default="")

    # URL base del endpoint compatible con la API de OpenAI. Permite apuntar
    # a servidores locales (Ollama, vLLM, LM Studio) o a proveedores alternativos.
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")

    # Identificador del modelo LLM utilizado para generar la respuesta final.
    LLM_MODEL: str = Field(default="gpt-4o-mini")

    # Tamaño en tokens de cada chunk al dividir los documentos de políticas.
    CHUNK_SIZE: int = Field(default=512, gt=0)

    # Número de chunks recuperados por similitud coseno para construir el contexto.
    TOP_K: int = Field(default=3, gt=0)

    # Ruta en disco donde ChromaDB persiste la colección vectorial.
    VECTORSTORE_PATH: Path = Field(default=Path("data/vectorstore"))

    # Carga automática del fichero .env y validación de tipos en la inicialización.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    def model_post_init(self, __context: object) -> None:
        """Garantiza que el directorio del vectorstore exista tras la carga."""
        # Creamos el directorio de persistencia de ChromaDB si aún no existe.
        os.makedirs(self.VECTORSTORE_PATH, exist_ok=True)


# Instancia singleton reutilizada por el resto de módulos del proyecto.
settings = Settings()
