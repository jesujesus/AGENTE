from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str = Field(..., description="Contenido del mensaje")

class AskRequest(BaseModel):
    question: str = Field(..., description="Pregunta del usuario")
    top_k: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(512, ge=64, le=4096)
    system_prompt: Optional[str] = None
    session_id: Optional[str] = Field(None, description="ID de sesión del chat (opcional)")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Historial breve del chat")

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
