import os
from dotenv import load_dotenv
from main_2 import process_file
# from script_vectorizacion import process_gcs_file  # <- Se usará solo en GCP

# ================================================================
# 🚩 Cargar variables de entorno
# ================================================================
load_dotenv()

# ================================================================
# 🚩 Ejecución local (simulación)
# ================================================================
if __name__ == "__main__":
    print("🔍 Iniciando prueba local...")

# 👇 Aquí agregas los archivos manualmente
    archivos = [
        "alumno.pdf",
        "image(1).png",
        "funcional (1).md",
        "PROCESO DE APRENDIZAJE.docx"
    ]

    for file_name in archivos:
        file_path = f"archivos-prueba/{file_name}"   # ruta local
        ruta_gcs = f"archivos-prueba/{file_name}"   # simulamos ruta en GCS
        print(f"\n➡️ Procesando: {file_name}")
        process_file(file_path, file_name, ruta_gcs)

    print("\n✅ Procesamiento finalizado en local.")



# ================================================================
# 🚩 Entrada para Cloud Functions (comentada en local)
# ================================================================
"""
from script_vectorizacion import process_gcs_file

def entry_point_for_gcf(data, context):
    return process_gcs_file(data, context)
"""


"""

# ================================================================
carpeta_local = "archivos-prueba"

print("🔍 Iniciando prueba local con múltiples archivos...")

for file_name in os.listdir(carpeta_local):
    file_path = os.path.join(carpeta_local, file_name)
    if os.path.isfile(file_path):
        print(f"\n➡️ Procesando: {file_name}")
        ruta_gcs = f"{carpeta_local}/{file_name}"  # simulamos ruta en GCS
        process_file(file_path, file_name, ruta_gcs)

print("✅ Procesamiento de todos los archivos finalizado.")
# """