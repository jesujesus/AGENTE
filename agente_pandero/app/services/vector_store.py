from typing import List, Dict, Any
from google.cloud import bigquery
from app.services.gemini_client import GeminiClient
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ID: str | None = None
    DATASET: str = "vector_db"
    TABLE_EMBEDDINGS: str = "t_embeddings_prueba"

    GEMINI_API_KEY: str | None = None
    GEMINI_EMBED_MODEL: str = "text-embedding-004"

    class Config:
        env_file = ".env"
        extra = "ignore"


class VectorStore:
    """
    Búsqueda vectorial en BigQuery con embeddings ARRAY<FLOAT64>.
    Requiere una columna 'embedding' compatible con COSINE_DISTANCE.
    """

    def __init__(self, project_id: str | None, dataset: str, table: str, embed_model: str = "text-embedding-004"):
        cfg = Settings()
        self.project_id = project_id or cfg.PROJECT_ID
        self.dataset = dataset or cfg.DATASET
        self.table = table or cfg.TABLE_EMBEDDINGS
        self.embed_model = embed_model or cfg.GEMINI_EMBED_MODEL

        # Clientes
        self.bq = bigquery.Client(project=self.project_id)
        self.gemini = GeminiClient(api_key=cfg.GEMINI_API_KEY)

    def _embed(self, text: str) -> List[float]:
        return self.gemini.embed(text, embed_model=self.embed_model)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Consulta BigQuery por similitud coseno y retorna top_k fragmentos."""
        embedding = self._embed(query)
        if not embedding:
            return []

        sql = f"""
            SELECT
              texto,
              file_hash,
              ruta,
              COSINE_DISTANCE(embedding, @embedding) AS distancia
            FROM `{self.project_id}.{self.dataset}.{self.table}`
            ORDER BY distancia ASC
            LIMIT @k
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding),
                bigquery.ScalarQueryParameter("k", "INT64", top_k),
            ]
        )
        rows = self.bq.query(sql, job_config=job_config).result()
        return [
            {
                "texto": r.get("texto"),
                "file_hash": r.get("file_hash"),
                "ruta": r.get("ruta"),
                "distancia": float(r.get("distancia")),
            }
            for r in rows
        ]
