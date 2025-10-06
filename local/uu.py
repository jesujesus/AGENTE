import os
import re
import hashlib
import pandas as pd
import fitz
import requests
import tempfile
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import storage, bigquery, documentai

# ==========================
# Cargar variables de entorno
# ==========================
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("❌ PROJECT_ID no está definido en el entorno")

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

LOCATION = os.getenv("LOCATION", "us")
BUCKET_NAME = os.getenv("BUCKET_NAME")
DATASET = os.getenv("DATASET", "vectordb_doc_procesos")
TABLE_EMBEDDINGS = os.getenv("TABLE_EMBEDDINGS", "t_embeddings_prueba")
TABLE_CONTROL = os.getenv("TABLE_CONTROL", "t_files_control_prueba")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 200))
REDUCIR_DIM = os.getenv("REDUCIR_DIM", "False").lower() == "true"
DIM = int(os.getenv("DIM", 768))

# Clientes GCP
storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)
docai_client = documentai.DocumentProcessorServiceClient()

# ==========================
# CREACIÓN/VALIDACIÓN DE TABLAS
# ==========================
def crear_tablas_si_no_existen():
    dataset_ref = bq_client.dataset(DATASET, project=PROJECT_ID)

    try:
        bq_client.get_dataset(dataset_ref)
        print(f"✅ Dataset '{DATASET}' validado.")
    except Exception as e:
        raise RuntimeError(f"❌ ERROR: dataset '{DATASET}' no existe: {e}")

    # Tabla de control
    table_control_id = f"{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}"
    schema_control = [
        bigquery.SchemaField("file_name", "STRING"),
        bigquery.SchemaField("file_hash", "STRING"),
        bigquery.SchemaField("processed_at", "TIMESTAMP"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("ruta", "STRING"),
    ]
    try:
        bq_client.get_table(table_control_id)
        print(f"✅ Tabla de control '{TABLE_CONTROL}' existe.")
    except Exception:
        bq_client.create_table(bigquery.Table(table_control_id, schema=schema_control))
        print(f"🆕 Tabla de control '{TABLE_CONTROL}' creada.")

    # Tabla de embeddings
    table_embeddings_id = f"{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}"
    schema_embeddings = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("texto", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
        bigquery.SchemaField("file_hash", "STRING"),
        bigquery.SchemaField("fuente", "STRING"),
        bigquery.SchemaField("ruta", "STRING"),
        bigquery.SchemaField("processed_at", "TIMESTAMP"),
        bigquery.SchemaField("metadata", "STRING"),
    ]
    try:
        bq_client.get_table(table_embeddings_id)
        print(f"✅ Tabla de embeddings '{TABLE_EMBEDDINGS}' existe.")
    except Exception:
        bq_client.create_table(bigquery.Table(table_embeddings_id, schema=schema_embeddings))
        print(f"🆕 Tabla de embeddings '{TABLE_EMBEDDINGS}' creada.")

# ==========================
# AUXILIARES
# ==========================
def calcular_hash(path_archivo):
    with open(path_archivo, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def validar_archivo(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".pdf": "pdf",
        ".png": "imagen",
        ".jpg": "imagen",
        ".jpeg": "imagen",
        ".txt": "texto",
        ".csv": "tabla",
        ".xlsx": "tabla",
        ".json": "texto",
        ".md": "texto",
        ".docx": "texto",
    }.get(ext)

def chunk_text(text, max_size=500, overlap=50):
    if not text or not text.strip():
        return []
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_size:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    if overlap > 0 and len(chunks) > 1:
        out = []
        for i, ch in enumerate(chunks):
            if i == 0:
                out.append(ch)
            else:
                prev = out[-1]
                tail = prev[-overlap:] if len(prev) >= overlap else prev
                out.append((tail + " " + ch).strip())
        return out
    return chunks

# ==========================
# EXTRACCIÓN DE TEXTO / CHUNKS
# ==========================
def extract_chunks_from_pdf(path, processor_id=None):
    chunks = []
    with fitz.open(path) as doc:
        page_count = len(doc)
        for page_number, page in enumerate(doc, start=1):
            texto = page.get_text("text")
            if texto.strip():
                for ch in chunk_text(texto):
                    chunks.append({
                        "texto": ch,
                        "chunk_type": "paragraph",
                        "page_number": page_number,
                        "page_count": page_count
                    })

            # Imágenes con OCR
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_path = os.path.join(tempfile.gettempdir(), f"page{page_number}_img{img_index}.png")
                    pix.save(img_path)

                    texto_ocr = ""
                    if processor_id:
                        try:
                            texto_ocr = extract_with_docai(img_path, "image/png", processor_id)
                        except Exception as e:
                            print(f"⚠️ Error OCR en página {page_number}, imagen {img_index}: {e}")

                    chunk_type = "image_ocr" if texto_ocr.strip() else "image"
                    chunks.append({
                        "texto": texto_ocr.strip(),
                        "chunk_type": chunk_type,
                        "page_number": page_number,
                        "page_count": page_count
                    })
                except Exception as e:
                    print(f"⚠️ No se pudo procesar imagen en página {page_number}: {e}")
    return chunks

def extract_with_docai(path, mime_type, processor_id):
    with open(path, "rb") as f:
        content = f.read()
    raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
    request = documentai.ProcessRequest(
        name=f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{processor_id}",
        raw_document=raw_document,
    )
    result = docai_client.process_document(request=request)
    return result.document.text

def process_csv(path):
    df = pd.read_csv(path)
    frases = []
    for _, row in df.iterrows():
        frases.append(" | ".join([f"{col}: {row[col]}" for col in df.columns]))
    return frases

# ==========================
# BIGQUERY HELPERS
# ==========================
def archivo_ya_procesado(file_hash):
    query = f"""
        SELECT status
        FROM `{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}`
        WHERE file_hash = @file_hash
        ORDER BY processed_at DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("file_hash", "STRING", file_hash)]
    )
    results = bq_client.query(query, job_config=job_config).result()
    return any(row["status"] == "OK" for row in results)

def registrar_control(file_name, file_hash, status, ruta):
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}"
    row = [{
        "file_name": file_name,
        "file_hash": file_hash,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ruta": ruta
    }]
    errors = bq_client.insert_rows_json(table_id, row)
    if errors:
        print(f"⚠️ Error registrando control: {errors}")

# ==========================
# EMBEDDINGS
# ==========================
def embed_and_insert(chunks, file_name, file_hash, ruta, categoria, proceso, producto, nivel_proceso, mime_type="application/pdf", page_count=None):
    if not GEMINI_API_KEY:
        print("❌ No se generarán embeddings: falta GEMINI_API_KEY.")
        return

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        payload = {
            "requests": [{"model": "models/embedding-001", "content": {"parts": [{"text": ch['texto']} ]}} for ch in batch]
        }

        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            rows = []
            for j, emb_data in enumerate(data.get("embeddings", [])):
                emb = emb_data.get("values")
                if not emb:
                    continue
                if REDUCIR_DIM:
                    emb = emb[:DIM]

                ch = batch[j]
                metadata = {
                    "source_uri": f"gs://{BUCKET_NAME}/{ruta}",
                    "mime_type": mime_type,
                    "page_count": ch.get("page_count", page_count),
                    "page_number": ch.get("page_number"),
                    "chunk_type": ch.get("chunk_type", "paragraph"),
                    "categoria": categoria,
                    "proceso": proceso,
                    "producto": producto,
                    "nivel_proceso": nivel_proceso
                }

                rows.append({
                    "id": f"{file_name}_{i+j}",
                    "texto": ch["texto"],
                    "embedding": emb,
                    "file_hash": file_hash,
                    "fuente": file_name,
                    "ruta": ruta,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": json.dumps(metadata)
                })

            if rows:
                _insert_embeddings(rows, file_name, file_hash)

        except Exception as e:
            print(f"⚠️ Error batch Gemini: {e}")

def _insert_embeddings(rows, file_name, file_hash):
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}"
    errors = bq_client.insert_rows_json(table_id, rows)
    if errors:
        print("⚠️ Errores insertando embeddings en BigQuery:", errors)
    else:
        print(f"✅ Insertados {len(rows)} embeddings de {file_name} (hash={file_hash})")

# ==========================
# PIPELINE PRINCIPAL
# ==========================
def process_file(file_path, file_name, ruta_gcs, processor_id=PROCESSOR_ID):
    file_type = validar_archivo(file_path)
    if not file_type:
        registrar_control(file_name, "N/A", "ERROR", ruta_gcs)
        return

    file_hash = calcular_hash(file_path)
    if archivo_ya_procesado(file_hash):
        print(f"ℹ️ {file_name} ya fue procesado. Saltando…")
        return

    categoria, proceso, producto, nivel_proceso = "general", "general", "general", "general"
    mime_type, page_count = None, None
    chunks = []

    try:
        if file_type == "pdf":
            chunks = extract_chunks_from_pdf(file_path, processor_id)
            mime_type = "application/pdf"

        elif file_type == "imagen" and processor_id:
            mime_type = "image/jpeg" if file_path.endswith(('.jpg', '.jpeg')) else "image/png"
            texto = extract_with_docai(file_path, mime_type, processor_id)
            chunk_type = "image_ocr" if texto.strip() else "image"
            chunks.append({"texto": texto, "chunk_type": chunk_type, "page_number": 1, "page_count": 1})

        elif file_type == "tabla":
            if file_path.endswith(".csv"):
                filas = process_csv(file_path)
                mime_type = "text/csv"
            else:
                df = pd.read_excel(file_path)
                filas = [" | ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            for f in filas:
                chunks.append({"texto": f, "chunk_type": "table", "page_number": 1, "page_count": 1})

        elif file_type == "texto":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()
            mime_type = "text/plain"
            for ch in chunk_text(texto):
                chunks.append({"texto": ch, "chunk_type": "paragraph", "page_number": 1, "page_count": 1})

        if chunks:
            embed_and_insert(chunks, file_name, file_hash, ruta_gcs, categoria, proceso, producto, nivel_proceso, mime_type)
            registrar_control(file_name, file_hash, "OK", ruta_gcs)
        else:
            registrar_control(file_name, file_hash, "SIN_TEXTO", ruta_gcs)

    except Exception as e:
        print(f"❌ Error procesando {file_name}: {e}")
        registrar_control(file_name, file_hash, "ERROR", ruta_gcs)

# ==========================
# CLOUD FUNCTION ENTRYPOINT
# ==========================
def process_gcs_file(event, context):
    file_name = event['name']
    bucket_name = event['bucket']

    if file_name.endswith('/'):
        return

    print(f"▶️ Procesando archivo: {file_name}")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_name))

    try:
        blob.download_to_filename(local_path)
        process_file(local_path, os.path.basename(file_name), file_name)
        print(f"✅ Procesamiento completado: {file_name}")
    except Exception as e:
        print(f"❌ Error al procesar {file_name}: {e}")

# ==========================
# SIMULACIÓN LOCAL
# ==========================
def main_ejecucion_local():
    crear_tablas_si_no_existen()
    bucket = storage_client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs():
        if blob.name.endswith("/"):
            continue
        event = {'name': blob.name, 'bucket': BUCKET_NAME}
        process_gcs_file(event, None)
    print("\n✅ Simulación completada para todo el bucket.")

if __name__ == "__main__":
    main_ejecucion_local()
