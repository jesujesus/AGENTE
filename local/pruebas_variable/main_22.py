import os
import re
import hashlib
import pandas as pd
import fitz 
from datetime import datetime
import requests
from google.cloud import storage, bigquery, documentai

from dotenv import load_dotenv

# 👇 Fuerza a que busque el .env en la carpeta raíz del proyecto
load_dotenv(dotenv_path="C:/Users/Admin/Desktop/local/.env")

# ==========================
# CONFIGURACIÓN INICIAL (NOMBRES ACTUALIZADOS)
# ==========================
PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("❌ PROJECT_ID no está definido en el entorno")

print("DEBUG PROJECT_ID:", PROJECT_ID)


LOCATION = os.environ.get("LOCATION", "us")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# DATASET que ya existe (Nombre del primer script)
DATASET = os.environ.get("DATASET", "vectordb_doc_procesos")
# Nuevas Tablas de prueba (Nombres del primer script)
TABLE_EMBEDDINGS = os.environ.get("TABLE_EMBEDDINGS", "t_embeddings_prueba")
TABLE_CONTROL = os.environ.get("TABLE_CONTROL", "t_files_control_prueba")
PROCESSOR_ID = os.environ.get("PROCESSOR_ID")


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"


#storage_client = storage.Client()
#bq_client = bigquery.Client()
#docai_client = documentai.DocumentProcessorServiceClient()
storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)
docai_client = documentai.DocumentProcessorServiceClient()

# ====================================================================
# FUNCIONES PARA CREACIÓN DE INFRAESTRUCTURA
# ====================================================================

def crear_tablas_si_no_existen():
    """
    Crea las dos tablas en BigQuery si no existen.
    Usa los nombres: 't_files_control_prueba' y 't_embeddings_prueba'.
    """
    dataset_ref = bq_client.dataset(DATASET, project=PROJECT_ID)
    
    # 1. Validar Dataset
    try:
        bq_client.get_dataset(dataset_ref)
        print(f"Dataset '{DATASET}' validado: existe.")
    except Exception as e:
        print(f"ERROR: El Dataset '{DATASET}' no existe o hay un problema de permisos. {e}")
        raise # Detiene la ejecución si el Dataset base no existe/es inaccesible

    # 2. Esquema para t_files_control_prueba
    schema_control = [
        bigquery.SchemaField("file_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("file_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("processed_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("categoria", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("proceso", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("producto", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ruta", "STRING", mode="NULLABLE"),
    ]

    # 3. Crear Tabla de Control (t_files_control_prueba)
    table_control_id = f"{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}"
    table_control = bigquery.Table(table_control_id, schema=schema_control)
    try:
        bq_client.get_table(table_control)
        print(f"Tabla '{TABLE_CONTROL}' ya existe.")
    except Exception:
        bq_client.create_table(table_control)
        print(f"Tabla '{TABLE_CONTROL}' creada exitosamente.")

    # 4. Esquema para t_embeddings_prueba
    schema_embeddings = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("texto", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"), # Array de FLOAT para el vector
        bigquery.SchemaField("file_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("fuente", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("processed_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("ruta", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("categoria", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("proceso", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("producto", "STRING", mode="NULLABLE"),
    ]
    
    # 5. Crear Tabla de Embeddings (t_embeddings_prueba)
    table_embeddings_id = f"{PROJECT_ID}.{DATASET}.{TABLE_EMBEDDINGS}"
    table_embeddings = bigquery.Table(table_embeddings_id, schema=schema_embeddings)
    try:
        bq_client.get_table(table_embeddings)
        print(f"Tabla '{TABLE_EMBEDDINGS}' ya existe.")
    except Exception:
        bq_client.create_table(table_embeddings)
        print(f"Tabla '{TABLE_EMBEDDINGS}' creada exitosamente.")


# ==========================
# FUNCIONES AUXILIARES
# ==========================

def calcular_hash(path_archivo):
    """Genera un hash SHA-256 del archivo para control de duplicados."""
    with open(path_archivo, "rb") as f:
        contenido = f.read()
        return hashlib.sha256(contenido).hexdigest()


def validar_archivo(file_path):
    """Clasifica el archivo según extensión y decide si procesarlo."""
    ext = os.path.splitext(file_path)[1].lower()
    extensiones_validas = {
        ".pdf": "pdf",
        ".png": "imagen",
        ".jpg": "imagen",
        ".jpeg": "imagen",
        ".txt": "texto",
        ".csv": "tabla",
        ".xlsx": "tabla",
        ".json": "texto"
    }
    return extensiones_validas.get(ext, None)


def chunk_text(text, max_size=500, overlap=50):
    """Divide texto en fragmentos respetando frases y agregando solapamiento."""
    
    if not text or not text.strip():
        return []

    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[\.\?\!])\s+|\n+', text)

    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    
    if overlap > 0 and len(chunks) > 1:
        # Lógica de solapamiento
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                prev_chunk = overlapped_chunks[-1]
                overlap_text = prev_chunk[-overlap:]
                # Intenta asegurar que el solapamiento sea palabra completa
                if len(overlap_text.strip()) > 0 and overlap_text[0].isalnum():
                    overlap_text = overlap_text.split()[-1] + " " + overlap_text[overlap_text.rfind(' ')+1:]
                
                overlapped_chunks.append(overlap_text.strip() + " " + chunk)
        return overlapped_chunks

    return chunks


# ==========================
# METADATA EXTRA
# ==========================

def detectar_producto(ruta, nombre_archivo):
    """Clasifica el archivo como vehiculos, viviendas o general."""
    
    texto = (ruta + " " + nombre_archivo).lower()
    claves_vehiculos = ["vehiculo", "vehiculos", "automotor", "automotores", "soat",
                         "placa", "carro", "auto", "camion", "moto"]
    claves_viviendas = ["inmueble", "inmuebles", "hipotecario", "casa",
                         "departamento", "vivienda", "condominio", "edificio"]

    for clave in claves_vehiculos:
        if clave in texto:
            return "vehiculos"
    for clave in claves_viviendas:
        if clave in texto:
            return "viviendas"
    return "general"


def extraer_metadatos(ruta_gcs, file_name):
    """Extrae categoria, proceso, producto y ruta desde la ruta GCS."""
    
    partes = ruta_gcs.split("/")
    categoria = partes[0] if len(partes) > 0 else "desconocido"
    proceso = partes[1] if len(partes) > 1 else "desconocido"
    producto = detectar_producto(ruta_gcs, file_name)
    return categoria, proceso, producto, ruta_gcs


# ==========================
# FUNCIONES DE TEXTO
# ==========================

def extract_text_from_pdf(path):
    """Extrae texto simple de PDF con PyMuPDF."""
    
    texto = ""
    with fitz.open(path) as doc:
        for page in doc:
            texto += page.get_text()
    return texto


def extract_with_docai(path, mime_type, processor_id):
    """Usa Document AI para OCR o tablas."""
    
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
    """Transforma CSV en frases legibles para embeddings."""
    
    df = pd.read_csv(path)
    frases = []
    for _, row in df.iterrows():
        frases.append(" | ".join([f"{col}: {row[col]}" for col in df.columns]))
    return "\n".join(frases)


# ==========================
# BIGQUERY HELPERS
# ==========================

def archivo_ya_procesado(file_hash):
    """Revisa en BigQuery si el archivo ya fue procesado."""
    
    query = f"""
        SELECT status FROM `{PROJECT_ID}.{DATASET}.{TABLE_CONTROL}`
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
    """Inserta registro en la tabla de control con metadata extendida."""
    
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
         print(f" Error registrando control: {errors}")


def embed_and_insert(chunks, file_name, file_hash, ruta,
                     categoria, proceso, producto, modelo="embedding-001", reducir_dim=False, dim=128):
    """Genera embeddings con Gemini y los guarda en BigQuery con metadata completa."""
    
    rows = []
    for idx, chunk in enumerate(chunks):
        payload = {
            "model": f"models/{modelo}",
            "content": {"parts": [{"text": chunk}]}
        }

        resp = requests.post(GEMINI_URL, json=payload, timeout=15)
        if not resp.ok:
            print(f" Error en Gemini HTTP {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        embedding = data.get("embedding", {}).get("values")

        if not embedding:
            print(" La API no devolvió embedding:", data)
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
            print(" Errores en BigQuery:", errors)
        else:
            print(f" Insertados {len(rows)} embeddings de {file_name}")


# ==========================
# PIPELINE PRINCIPAL
# ==========================

def process_file(file_path, file_name, ruta_gcs, processor_id=PROCESSOR_ID):
    """Procesa archivo según tipo detectado: extrae texto, chunkea, genera embeddings y registra."""
    
    file_type = validar_archivo(file_path)
    if not file_type:
        print(f" {file_name}: formato no soportado")
        registrar_control(file_name, "N/A", "ERROR", "desconocido", "desconocido", "general", ruta_gcs)
        return

    file_hash = calcular_hash(file_path)
    categoria, proceso, producto, ruta = extraer_metadatos(ruta_gcs, file_name)

    if archivo_ya_procesado(file_hash):
        print(f" {file_name} ya fue procesado. Saltando...")
        return

    texto = ""

    try:
        # --- EXTRACCIÓN DE TEXTO ---
        if file_type == "pdf":
            texto = extract_text_from_pdf(file_path)
            if len(texto.strip()) < 50 and processor_id:
                print(" Intentando con Document AI...")
                texto = extract_with_docai(file_path, "application/pdf", processor_id)

        elif file_type == "imagen" and processor_id:
            print(" Usando Document AI para OCR...")
            mime_type = "image/jpeg" if file_path.endswith(('.jpg', '.jpeg')) else "image/png"
            texto = extract_with_docai(file_path, mime_type, processor_id)

        elif file_type == "tabla":
            if file_path.endswith(".csv"):
                texto = process_csv(file_path)
            else: # Asume XLSX
                df = pd.read_excel(file_path)
                frases = []
                for _, row in df.iterrows():
                    frases.append(" | ".join([f"{col}: {row[col]}" for col in df.columns]))
                texto = "\n".join(frases)

        elif file_type == "texto":
            with open(file_path, "r", encoding="utf-8") as f:
                texto = f.read()
        
        # --- CHUNKING Y EMBEDDING ---
        if texto.strip():
            chunks = chunk_text(texto, max_size=500, overlap=50)
            embed_and_insert(chunks, file_name, file_hash, ruta,
                             categoria, proceso, producto)
            registrar_control(file_name, file_hash, "OK", categoria, proceso, producto, ruta)
        else:
            print(f" {file_name}: no se extrajo texto válido")
            registrar_control(file_name, file_hash, "SIN_TEXTO", categoria, proceso, producto, ruta)

    except Exception as e:
        print(f" Error procesando {file_name}: {e}")
        registrar_control(file_name, file_hash, "ERROR", categoria, proceso, producto, ruta)


# ==========================
# FUNCIÓN PARA CLOUD FUNCTION 
# ==========================

def process_gcs_file(data, context):
    """
    Cloud Function disparada por un evento de Cloud Storage.
    """
    
    # --- 1. CREACIÓN DE INFRAESTRUCTURA ---
    try:
        crear_tablas_si_no_existen()
    except Exception as e:
        print(f"Falló la creación/validación de tablas de BigQuery: {e}")
        return # Si la DB falla, la función termina

    # --- 2. CONFIGURACIÓN DEL EVENTO ---
    file_name = data['name']
    bucket_name = data['bucket']
    ruta_gcs = file_name
    
    
    if file_name.endswith('/'):
        print(f" Archivo saltado: {file_name} es un directorio.")
        return
    
    print(f" Iniciando procesamiento del archivo: {file_name} del bucket: {bucket_name}")

    
    # --- 3. DESCARGA DEL ARCHIVO ---
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    local_path = f"/tmp/{os.path.basename(file_name)}" 
    
    try:
        blob.download_to_filename(local_path)
        print(f" Archivo descargado a: {local_path}")
    except Exception as e:
        print(f" Error al descargar {file_name} de GCS: {e}")
        return 

    
    # --- 4. EJECUCIÓN DEL PIPELINE ---
    try:
        process_file(local_path, os.path.basename(file_name), ruta_gcs)
    except Exception as e:
        print(f" Error fatal en el pipeline para {file_name}: {e}")
        
    print(f" Procesamiento de {file_name} completado.")