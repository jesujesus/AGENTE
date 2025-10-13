from typing import Optional, List

# Librería oficial de Gemini
try:
    import google.generativeai as genai
except Exception as e:  # pragma: no cover
    genai = None


class GeminiClient:
    """
    Cliente mínimo para:
      - embed(text): retornar embedding (list[float])
      - generate(system_prompt, user_prompt, ...): texto generado
    """

    def __init__(self, api_key: Optional[str], model_name: str = "gemini-1.5-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurado en .env")
        if genai is None:
            raise RuntimeError(
                "Falta dependencia 'google-generativeai'. Añádela a requirements.txt"
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def embed(self, text: str, embed_model: str = "text-embedding-004") -> List[float]:
        """Devuelve embedding para el texto usando el modelo indicado."""
        if not text:
            return []
        res = genai.embed_content(model=embed_model, content=text)
        # Estructura típica: {"embedding": {"values": [...]}}
        return res["embedding"]["values"]

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 512,
    ) -> str:
        """Genera una respuesta con tono más natural usando el contexto recuperado."""
        final_prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = self.model.generate_content(
            final_prompt,
            generation_config={
                "temperature": float(temperature),
                "max_output_tokens": int(max_output_tokens),
            },
        )
        return (getattr(resp, "text", None) or "No pude generar respuesta.").strip()
