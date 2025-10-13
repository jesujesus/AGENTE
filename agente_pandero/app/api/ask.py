from fastapi import APIRouter, HTTPException
from app.models.ask_models import AskRequest, AskResponse
from app.services.vector_store import VectorStore
from app.services.gemini_client import GeminiClient
from pydantic_settings import BaseSettings
from datetime import datetime

# 👈 IMPORTANTE: define el router ANTES de usar @router.post
router = APIRouter()

class Settings(BaseSettings):
    PROJECT_ID: str | None = None
    DATASET: str = "vector_db"
    TABLE_EMBEDDINGS: str = "t_embeddings_prueba"

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Inicializa servicios (una sola vez por proceso)
vector_store = VectorStore(
    project_id=settings.PROJECT_ID,
    dataset=settings.DATASET,
    table=settings.TABLE_EMBEDDINGS,
    embed_model=settings.GEMINI_EMBED_MODEL,
)

gemini = GeminiClient(
    api_key=settings.GEMINI_API_KEY,
    model_name=settings.GEMINI_MODEL,
)

@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    """
    Orquesta: (1) búsqueda semántica en BigQuery → (2) respuesta natural con Gemini.
    """
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="'question' no puede estar vacío")

    # 1) Recuperar contexto (chunks)
    chunks = vector_store.search(payload.question, top_k=payload.top_k)

    # 2) Construir prompts
    context_text = "\n\n".join(
        [f"Fuente {i+1}: {c['texto']}" for i, c in enumerate(chunks)]
    ) if chunks else ""

    sys_prompt = payload.system_prompt or (
        "Eres un asistente experto en procesos corporativos de Pandero. "
        "Responde de forma clara, concisa y con tono profesional-cercano. "
        "Usa solo la información del contexto y el historial (si lo hay). "
        "Si no hay suficiente info, dilo. Responde en español."
    )
    user_prompt = (
        f"Pregunta: {payload.question}\n\n"
        f"Contexto:\n{context_text}\n\n"
        "Si usas fragmentos, referencia 'Fuente 1', 'Fuente 2', etc."
    )

    # 3) Llamar a Gemini
    answer = gemini.generate(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=payload.temperature,
        max_output_tokens=payload.max_output_tokens,
    )

    sources = [
        {
            "id": c.get("file_hash"),
            "ruta": c.get("ruta"),
            "score": 1 - float(c.get("distancia", 0.0)),  # mayor es mejor
            "preview": (c.get("texto") or "")[:280],
        }
        for c in chunks
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        metadata={
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "retrieved": len(chunks),
            "model": settings.GEMINI_MODEL,
        },
    )
