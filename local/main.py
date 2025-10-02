import os
import re
import hashlib
import pandas as pd
import fitz
from datetime import datetime
import requests
from google.cloud import storage, bigquery, documentai
import tempfile # para manejo de archivos temporales
from dotenv import load_dotenv

# ==========================
# Cargar variables de entorno (.env)
# ==========================
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("❌ PROJECT_ID no está definido en el entorno")

# Algunos clientes miran esta var:
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

LOCATION = os.getenv("LOCATION", "us")
BUCKET_NAME = os.getenv("BUCKET_NAME")
DATASET = os.getenv("DATASET", "vectordb_doc_procesos")
TABLE_EMBEDDINGS = os.getenv("TABLE_EMBEDDINGS", "t_embeddings_prueba")
TABLE_CONTROL = os.getenv("TABLE_CONTROL", "t_files_control_prueba")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY no está definida. Los embeddings fallarán hasta que la configures.")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"

# Clientes GCP con project explícito
storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)
docai_client = documentai.DocumentProcessorServiceClient()

# ====================================================================
# CREACIÓN/VALIDACIÓN DE INFRA
# ====================================================================
def crear_tablas_si_no_existen():
    """
    Crea (si no existen) las tablas:
      - `{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}`
      - `{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}`
    """
    dataset_ref = bq_client.dataset(DATASET, project=PROJECT_ID)

    # 1) Validar dataset
    try:
        bq_client.get_dataset(dataset_ref)
        print(f"✅ Dataset '{DATASET}' validado.")
    except Exception as e:
        print(f"❌ ERROR: El dataset '{DATASET}' no existe o no hay permisos: {e}")
        raise

    # 2) Tabla de control
    table_control_id = f"{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}"
    schema_control = [
        bigquery.SchemaField("file_name", "STRING"),
        bigquery.SchemaField("file_hash", "STRING"),
        bigquery.SchemaField("processed_at", "TIMESTAMP"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("categoria", "STRING"),
        bigquery.SchemaField("proceso", "STRING"),
        bigquery.SchemaField("producto", "STRING"),
        bigquery.SchemaField("ruta", "STRING"),
    ]
    try:
        bq_client.get_table(table_control_id)
        print(f"✅ Tabla de control '{TABLE_CONTROL}' existe.")
    except Exception:
        table = bigquery.Table(table_control_id, schema=schema_control)
        bq_client.create_table(table)
        print(f"🆕 Tabla de control '{TABLE_CONTROL}' creada.")

    # 3) Tabla de embeddings
    table_embeddings_id = f"{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}"
    schema_embeddings = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("texto", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
        bigquery.SchemaField("file_hash", "STRING"),
        bigquery.SchemaField("fuente", "STRING"),
        bigquery.SchemaField("processed_at", "TIMESTAMP"),
        bigquery.SchemaField("ruta", "STRING"),
        bigquery.SchemaField("categoria", "STRING"),
        bigquery.SchemaField("proceso", "STRING"),
        bigquery.SchemaField("producto", "STRING"),
    ]
    try:
        bq_client.get_table(table_embeddings_id)
        print(f"✅ Tabla de embeddings '{TABLE_EMBEDDINGS}' existe.")
    except Exception:
        table = bigquery.Table(table_embeddings_id, schema=schema_embeddings)
        bq_client.create_table(table)
        print(f"🆕 Tabla de embeddings '{TABLE_EMBEDDINGS}' creada.")

# ==========================
# AUXILIARES
# ==========================
def calcular_hash(path_archivo):
    with open(path_archivo, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def validar_archivo(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    extensiones_validas = {
        ".pdf": "pdf",
        ".png": "imagen",
        ".jpg": "imagen",
        ".jpeg": "imagen",
        ".txt": "texto",
        ".csv": "tabla",
        ".xlsx": "tabla",
        ".json": "texto",
        ".md": "texto",
        ".docx": "texto",  # lo tratará como texto si lo abres tú antes; aquí no se parsea DOCX
    }
    return extensiones_validas.get(ext)

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
# METADATOS
# ==========================
def detectar_producto(ruta, nombre_archivo):
    texto = (ruta + " " + nombre_archivo).lower()
    claves_vehiculos = ["vehiculo", "vehiculos", "automotor", "automotores", "soat",
                        "placa", "carro", "auto", "camion", "moto"]
    claves_viviendas = ["inmueble", "inmuebles", "hipotecario", "casa",
                        "departamento", "vivienda", "condominio", "edificio"]
    for c in claves_vehiculos:
        if c in texto:
            return "vehiculos"
    for c in claves_viviendas:
        if c in texto:
            return "viviendas"
    return "general"

def extraer_metadatos(ruta_gcs, file_name):
    partes = ruta_gcs.split("/")
    categoria = partes[0] if len(partes) > 0 else "desconocido"
    proceso = partes[1] if len(partes) > 1 else "desconocido"
    producto = detectar_producto(ruta_gcs, file_name)
    return categoria, proceso, producto, ruta_gcs

# ==========================
# EXTRACCIÓN DE TEXTO
# ==========================
def extract_text_from_pdf(path):
    texto = ""
    with fitz.open(path) as doc:
        for page in doc:
            texto += page.get_text()
    return texto

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
    return "\n".join(frases)

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
    for row in results:
        if row["status"] == "OK":
            return True
    return False

def registrar_control(file_name, file_hash, status, categoria, proceso, producto, ruta):
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}"
    row = [{
        "file_name": file_name,
        "file_hash": file_hash,
        "processed_at": datetime.utcnow().isoformat(),
        "status": status,
        "categoria": categoria,
        "proceso": proceso,
        "producto": producto,
        "ruta": ruta
    }]
    errors = bq_client.insert_rows_json(table_id, row)
    if errors:
        print(f"⚠️  Error registrando control: {errors}")

def embed_and_insert(chunks, file_name, file_hash, ruta,
                     categoria, proceso, producto, modelo="embedding-001", reducir_dim=True, dim=10):
    if not GEMINI_API_KEY:
        print("❌ No se generarán embeddings: falta GEMINI_API_KEY.")
        return

    rows = []
    for idx, chunk in enumerate(chunks):
        payload = {
            "model": f"models/{modelo}",
            "content": {"parts": [{"text": chunk}]}
        }
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        except Exception as e:
            print(f"⚠️  Error llamando a Gemini: {e}")
            continue

        if not resp.ok:
            print(f"⚠️  Error Gemini HTTP {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        embedding = data.get("embedding", {}).get("values")
        if not embedding:
            print("⚠️  La API no devolvió embedding:", data)
            continue

        if reducir_dim:
            embedding = embedding[:dim]

        rows.append({
            "id": f"{file_name}_{idx}",
            "texto": chunk,
            "embedding": embedding,
            "file_hash": file_hash,
            "fuente": file_name,
            "processed_at": datetime.utcnow().isoformat(),
            "ruta": ruta,
            "categoria": categoria,
            "proceso": proceso,
            "producto": producto
        })

    if rows:
        table_id = f"{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}"
        errors = bq_client.insert_rows_json(table_id, rows)
        if errors:
            print("⚠️  Errores insertando embeddings en BigQuery:", errors)
        else:
            print(f"✅ Insertados {len(rows)} embeddings de {file_name}")

# ==========================
# PIPELINE PRINCIPAL
# ==========================
def process_file(file_path, file_name, ruta_gcs, processor_id=PROCESSOR_ID):
    """
    Procesa archivo según tipo detectado: extrae texto, chunkea, genera embeddings y registra.
    """
    # Asegurar infra antes de cualquier consulta
    #crear_tablas_si_no_existen()

    file_type = validar_archivo(file_path)
    if not file_type:
        print(f"⚠️  {file_name}: formato no soportado")
        registrar_control(file_name, "N/A", "ERROR", "desconocido", "desconocido", "general", ruta_gcs)
        return

    file_hash = calcular_hash(file_path)
    categoria, proceso, producto, ruta = extraer_metadatos(ruta_gcs, file_name)

    if archivo_ya_procesado(file_hash):
        print(f"ℹ️  {file_name} ya fue procesado. Saltando…")
        return

    texto = ""
    try:
        # --- EXTRACCIÓN DE TEXTO ---
        if file_type == "pdf":
            texto = extract_text_from_pdf(file_path)
            if len(texto.strip()) < 50 and processor_id:
                print("ℹ️  Texto PDF muy corto. Intentando con Document AI…")
                texto = extract_with_docai(file_path, "application/pdf", processor_id)

        elif file_type == "imagen" and processor_id:
            print("ℹ️  Usando Document AI para OCR de imagen…")
            mime_type = "image/jpeg" if file_path.endswith(('.jpg', '.jpeg')) else "image/png"
            texto = extract_with_docai(file_path, mime_type, processor_id)

        elif file_type == "tabla":
            if file_path.endswith(".csv"):
                texto = process_csv(file_path)
            else:  # XLSX
                df = pd.read_excel(file_path)
                frases = []
                for _, row in df.iterrows():
                    frases.append(" | ".join([f"{col}: {row[col]}" for col in df.columns]))
                texto = "\n".join(frases)

        elif file_type == "texto":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()

        # --- CHUNKING + EMBEDDINGS ---
        if texto.strip():
            chunks = chunk_text(texto, max_size=500, overlap=50)
            # 🚧 En pruebas, solo procesa los primeros 20 chunks
            chunks = chunks[:20]


            embed_and_insert(chunks, file_name, file_hash, ruta, categoria, proceso, producto)
            registrar_control(file_name, file_hash, "OK", categoria, proceso, producto, ruta)
        else:
            print(f"⚠️  {file_name}: no se extrajo texto válido")
            registrar_control(file_name, file_hash, "SIN_TEXTO", categoria, proceso, producto, ruta)

    except Exception as e:
        print(f"❌ Error procesando {file_name}: {e}")
        registrar_control(file_name, file_hash, "ERROR", categoria, proceso, producto, ruta)

# ==========================
# GCF ENTRYPOINT (GCS TRIGGER)
# ==========================
def process_gcs_file(data, context):
    """
    Cloud Function disparada por un evento de Cloud Storage.
    """
    """
    try:
        crear_tablas_si_no_existen()
    except Exception as e:
        print(f"❌ Falló la creación/validación de tablas de BigQuery: {e}")
        return
        """

    file_name = data['name']
    bucket_name = data['bucket']
    ruta_gcs = file_name

    if file_name.endswith('/'):
        print(f"ℹ️  Archivo saltado: {file_name} es un directorio.")
        return

    print(f"▶️  Iniciando procesamiento del archivo: {file_name} del bucket: {bucket_name}")

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    import tempfile
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_name))


    try:
        blob.download_to_filename(local_path)
        print(f"📥 Archivo descargado a: {local_path}")
    except Exception as e:
        print(f"❌ Error al descargar {file_name} de GCS: {e}")
        return

    try:
        process_file(local_path, os.path.basename(file_name), ruta_gcs)
    except Exception as e:
        print(f"❌ Error fatal en el pipeline para {file_name}: {e}")

    print(f"✅ Procesamiento de {file_name} completado.")


""""""""""""""""""""""""""""""""""""""""""""""""""""""
######################################
##""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""
# ==========================
# SIMULACIÓN DE EJECUCIÓN LOCAL
# ==========================
##
"""

def main_ejecucion_local():
    """
    Función que simula la ejecución local pero procesando TODOS los archivos del bucket.
    """
    try:
        crear_tablas_si_no_existen()
    except Exception as e:
        print(f"❌ Falló la creación/validación de tablas de BigQuery: {e}")
        return

    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()

    for blob in blobs:
        if blob.name.endswith("/"):
            # saltamos carpetas virtuales
            continue

        data_simulada = {
            'name': blob.name,
            'bucket': BUCKET_NAME
        }
        contexto_simulado = None

        print(f"\n🚀 Iniciando simulación de procesamiento para {blob.name}...")
        process_gcs_file(data_simulada, contexto_simulado)

    print("\n✅ Simulación de ejecución completada para todo el bucket.")

if __name__ == "__main__":
    # Asegúrate de que las variables de entorno están cargadas (ya lo haces)
    # y llama a la función de simulación.
    
    # 🛑 Asegúrate de que tu variable GOOGLE_APPLICATION_CREDENTIALS esté establecida en PowerShell 🛑
    # (Esto lo haces con $env:GOOGLE_APPLICATION_CREDENTIALS="...")
    
    main_ejecucion_local()