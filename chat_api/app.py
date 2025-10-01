import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import bigquery

# Inicializar FastAPI
app = FastAPI()

# Configuración
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET", "vector_db")
TABLE_EMBEDDINGS = os.getenv("TABLE_EMBEDDINGS", "t_embeddings_prueba")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# URL de Embeddings
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"

# Cliente de BigQuery
bq_client = bigquery.Client(project=PROJECT_ID)

# Modelo de request
class Pregunta(BaseModel):
    pregunta: str


@app.post("/chat")
def chat_endpoint(input_data: Pregunta):
    pregunta = input_data.pregunta

    # 1. Generar embedding de la pregunta
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": pregunta}]}
    }
    resp = requests.post(GEMINI_EMBED_URL, json=payload)
    if not resp.ok:
        return {"error": f"Gemini API error: {resp.text}"}

    embedding = resp.json().get("embedding", {}).get("values", [])
    if not embedding:
        return {"error": "No se pudo obtener embedding"}

    # 2. Consultar en BigQuery los más cercanos
    query = f"""
        SELECT 
          texto,
          file_hash,
          ruta,
          cosine_distance(embedding, @embedding) AS distancia
        FROM `{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}`
        ORDER BY distancia ASC
        LIMIT 3
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding)
        ]
    )
    results = bq_client.query(query, job_config=job_config).result()

    # 3. Preparar respuesta
    fragmentos = [{"texto": row["texto"], "distancia": row["distancia"]} for row in results]

    return {"respuesta": fragmentos}
