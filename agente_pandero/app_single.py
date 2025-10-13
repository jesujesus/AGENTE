from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import logging
import requests
import re
import unicodedata

# ====== ENV ======
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us")
DATASET = os.getenv("DATASET", "vectordb_doc_procesos")
TABLE_EMBEDDINGS = os.getenv("TABLE_EMBEDDINGS", "t_embeddings_prueba")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "embedding-001")  # igual que ingesta

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# ====== Logs ======
log = logging.getLogger("agente_pandero_single")
logging.basicConfig(level=logging.INFO)

# ====== SDKs lazy ======
genai = None
bigquery = None

def lazy_imports():
    global genai, bigquery
    if genai is None:
        try:
            import google.generativeai as _genai
            genai = _genai
        except Exception as e:
            raise RuntimeError(
                "No se pudo importar 'google-generativeai'. Instala con: pip install google-generativeai"
            ) from e
    if bigquery is None:
        try:
            from google.cloud import bigquery as _bq
            bigquery = _bq
        except Exception as e:
            raise RuntimeError(
                "No se pudo importar 'google-cloud-bigquery'. Instala con: pip install google-cloud-bigquery"
            ) from e

# ====== FastAPI ======
app = FastAPI(title="Agente Pandero (Single-File)", version="0.3.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def force_json_charset(request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if ct.startswith("application/json") and "charset" not in ct.lower():
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

# ====== Modelos ======
class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(2048, ge=64, le=4096)
    history: List[ChatMessage] = Field(default_factory=list)

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ====== Normalización (conserva ñ) ======
_VOWEL_MAP = str.maketrans({
    "á":"a","é":"e","í":"i","ó":"o","ú":"u",
    "Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U",
    "ü":"u","Ü":"U",
    # Importante: no tocar ñ / Ñ
})

def _normalize(s: str) -> str:
    """minúsculas + sin tildes en vocales + espacios colapsados (conservando ñ)"""
    s = (s or "").translate(_VOWEL_MAP)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _only_name(s: str) -> str:
    """quita frases de cortesía tipo 'háblame del alumno ...'"""
    t = (s or "").strip()
    t = re.sub(r"^(háblame|hablame|dime|busca|buscar|explícame|explicame)\s+del?\s+", "", t, flags=re.I)
    t = re.sub(r"^(sobre|del?\s+alumno|del?\s+estudiante)\s+", "", t, flags=re.I)
    return t.strip()

# ====== Utils: Gemini ======
def _gemini_configure():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY no configurado en .env")
    lazy_imports()
    genai.configure(api_key=GEMINI_API_KEY)

def check_gemini() -> Dict[str, Any]:
    try:
        if not GEMINI_API_KEY:
            return {"ok": False, "where": "gemini", "msg": "GEMINI_API_KEY no configurado en .env"}
        _gemini_configure()
        # Test generación
        model = genai.GenerativeModel(GEMINI_MODEL)
        r = model.generate_content("Responde 'OK' en una palabra.")
        txt = (getattr(r, "text", "") or "").strip()
        # Test embedding
        dims = None
        if GEMINI_EMBED_MODEL == "embedding-001":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
            payload = {"model": "models/embedding-001", "content": {"parts": [{"text": "ping"}]}}
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding", {}).get("value") or data.get("embedding", {}).get("values")
            dims = len(vec) if vec else None
        else:
            e = genai.embed_content(model=GEMINI_EMBED_MODEL, content="ping")
            dims = len(e["embedding"]["values"])
        return {"ok": True, "where": "gemini", "msg": f"gen='{txt}', emb_model='{GEMINI_EMBED_MODEL}', emb_dims={dims}"}
    except Exception as e:
        return {"ok": False, "where": "gemini", "msg": str(e)}

def embed_text(text: str) -> List[float]:
    """Devuelve el embedding del texto usando el MISMO modelo que la ingesta."""
    if GEMINI_EMBED_MODEL == "embedding-001":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY no configurado")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
        payload = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding", {}).get("value") or data.get("embedding", {}).get("values")
        if not vec:
            raise RuntimeError("Embedding vacío desde embedding-001")
        return vec
    else:
        _gemini_configure()
        e = genai.embed_content(model=GEMINI_EMBED_MODEL, content=text)
        return e["embedding"]["values"]

def generate_answer(question: str, context_text: str, temperature: float, max_tokens: int) -> str:
    """
    Si hay contexto, SIEMPRE responde con lo que haya (aunque sea poco).
    Solo devuelve 'insuficiente' cuando NO hay ningún contexto.
    """
    _gemini_configure()
    model = genai.GenerativeModel(GEMINI_MODEL)

    if not (context_text or "").strip():
        return "El contexto proporcionado es insuficiente para responder tu pregunta."

    target = _only_name(question) or question
    target_norm = _normalize(target)

    system = (
        "Responde en español, claro y profesional. Usa SOLO el CONTEXTO.\n"
        "Si hay algo de contexto, RESPONDE (aunque sea parcial) y dilo.\n"
        "Solo di 'El contexto es insuficiente' si NO hay contexto.\n"
        "Formato:\n"
        "1) Encabezado del objetivo.\n"
        "2) Hallazgos en viñetas (datos concretos, listas/criterios también en viñetas).\n"
        "3) Nota de limitaciones si aplica."
        )

    prompt = (
        f"{system}\n\n"
        f"Objetivo: {target}\n"
        f"Objetivo_norm: {target_norm}\n"
        f"Pregunta: {question}\n\n"
        f"=== CONTEXTO ===\n{context_text}\n=== FIN CONTEXTO ===\n"
        "Sigue el Formato."
        )


    resp = model.generate_content(
        prompt,
        generation_config={"temperature": float(temperature), "max_output_tokens": int(max_tokens)},
    )
    txt = getattr(resp, "text", "") or "No pude generar respuesta."
    return txt.strip()

# ====== Utils: BigQuery ======
def check_bigquery() -> Dict[str, Any]:
    try:
        if not PROJECT_ID:
            return {"ok": False, "where": "bigquery", "msg": "PROJECT_ID no configurado en .env"}
        lazy_imports()
        client = bigquery.Client(project=PROJECT_ID)
        full = f"{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}"
        client.get_table(full)  # NotFound si no existe
        return {"ok": True, "where": "bigquery", "msg": f"tabla OK: {full}"}
    except Exception as e:
        return {"ok": False, "where": "bigquery", "msg": str(e)}

def bq_search(embedding: List[float], limit: int) -> List[Dict[str, Any]]:
    """
    Esquema esperado:
      - embedding: ARRAY<FLOAT64>
      - texto: STRING
      - file_hash: STRING
      - ruta: STRING
    """
    lazy_imports()
    client = bigquery.Client(project=PROJECT_ID)
    sql = f"""
        SELECT
          texto, file_hash, ruta,
          COSINE_DISTANCE(embedding, @embedding) AS distancia
        FROM `{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}`
        ORDER BY distancia ASC
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [
        {
            "texto": r.get("texto"),
            "file_hash": r.get("file_hash"),
            "ruta": r.get("ruta"),
            "distancia": float(r.get("distancia")),
        }
        for r in rows
    ]

# ====== Endpoints ======
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/diag")
def diag():
    g = check_gemini()
    b = check_bigquery()
    return {
        "ok": bool(g.get("ok") and b.get("ok")),
        "checks": [g, b],
        "now": datetime.utcnow().isoformat() + "Z",
        "project_id": PROJECT_ID,
        "dataset": DATASET,
        "table": TABLE_EMBEDDINGS,
        "model": GEMINI_MODEL,
        "embed_model": GEMINI_EMBED_MODEL,
    }

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="'question' no puede estar vacío")

    # Prechecks
    g = check_gemini()
    if not g["ok"]:
        raise HTTPException(status_code=500, detail=f"Gemini no disponible: {g['msg']}")
    b = check_bigquery()
    if not b["ok"]:
        raise HTTPException(status_code=500, detail=f"BigQuery no disponible: {b['msg']}")

    # ====== Construcción de candidatos (prioriza originales con tildes) ======
    q = (payload.question or "").strip()
    candidates: List[str] = []
    if q:
        name_raw  = _only_name(q)              # p.ej. "Jesús Salvatierra de la Cruz"
        name_norm = _normalize(name_raw)
        q_norm    = _normalize(q)

        # Mejor semántica primero:
        if q:         candidates.append(q)          # frase original
        if name_raw:  candidates.append(name_raw)   # solo nombre original
        if q_norm:    candidates.append(q_norm)     # frase normalizada
        if name_norm: candidates.append(name_norm)  # nombre normalizado

    # Añade último mensaje del usuario (si vino)
    last_user = None
    for m in reversed(payload.history or []):
        role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else None)
        if role == "user" and content:
            last_user = content.strip()
            break
    if last_user:
        lu_raw  = _only_name(last_user)
        lu_norm = _normalize(lu_raw)
        lu_full = _normalize(last_user)
        for v in [last_user, lu_raw, lu_full, lu_norm]:
            if v and v not in candidates:
                candidates.append(v)

    # Limpia duplicados y vacíos
    seen, cand_list = set(), []
    for s in candidates:
        s2 = (s or "").strip()
        if s2 and s2 not in seen:
            cand_list.append(s2); seen.add(s2)

    # ====== Recuperación: acumula resultados de todas las variantes ======
    # Pedimos más por variante para aumentar recall y después recortamos.
    per_variant_limit = max(payload.top_k * 3, payload.top_k)
    all_hits: List[Dict[str, Any]] = []
    tried: List[Dict[str, Any]] = []

    for cand in cand_list[:8]:
        try:
            emb = embed_text(cand)
            got = bq_search(emb, per_variant_limit)
            tried.append({"q": cand, "hits": len(got)})
            if got:
                all_hits.extend(got)
        except Exception as e:
            tried.append({"q": cand, "error": str(e)})

    # Deduplicar por (file_hash, ruta, texto) quedándote con la mejor distancia
    uniq: Dict[tuple, Dict[str, Any]] = {}
    for r in all_hits:
        key = (r.get("file_hash"), r.get("ruta"), r.get("texto"))
        d = float(r.get("distancia", 1e9))
        if key not in uniq or d < uniq[key]["distancia"]:
            uniq[key] = r
    chunks = list(uniq.values())

    # ====== Rerank: token match (con ñ), distancia, y longitud ======
    def _contains_all_tokens(text_norm: str, query_norm: str) -> bool:
        toks = [t for t in (query_norm or "").split() if t]
        return bool(toks) and all(t in (text_norm or "") for t in toks)

    target_raw  = _only_name(payload.question) or payload.question
    target_norm = _normalize(target_raw)

    if chunks:
        ranked = []
        for c in chunks:
            t = (c.get("texto") or "")
            t_norm = _normalize(t)

            exact_phrase = 0 if (target_norm and target_norm in t_norm) else 1
            not_contains = 0 if _contains_all_tokens(t_norm, target_norm) else 1
            dist = float(c.get("distancia", 1e9))
            length_key = -len(t)

            ranked.append(((exact_phrase, not_contains, dist, length_key), c))

        ranked.sort(key=lambda x: x[0])
        chunks = [c for _, c in ranked][:payload.top_k]

    # ====== Construcción de contexto ======
    context_text = ""
    if chunks:
        partes = []
        for i, c in enumerate(chunks):
            t = (c.get("texto") or "").strip()
            if t:
                partes.append(f"Fuente {i+1}: {t}")
        context_text = "\n\n".join(partes)

    # ====== Generar respuesta ======
    try:
        answer = generate_answer(
            question=payload.question,
            context_text=context_text,
            temperature=payload.temperature,
            max_tokens=payload.max_output_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Gemini (generación): {e}")

    # ====== Fuentes ======
    sources = [
        {
            "id": c.get("file_hash"),
            "ruta": c.get("ruta"),
            "score": 1 - float(c.get("distancia", 0.0)),
            "preview": (c.get("texto") or "")[:280],
        }
        for c in chunks
    ]

    # ====== Respuesta final ======
    return AskResponse(
        answer=answer,
        sources=sources,
        metadata={
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "retrieved": len(chunks),
            "model": GEMINI_MODEL,
            "embed_model": GEMINI_EMBED_MODEL,
            "retrieval_attempts": tried,
            "server_patch": "ask-retry+merge+rerank-v4"
        },
    )
