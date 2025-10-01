import os
from google.cloud import storage
from dotenv import load_dotenv

# Cargar variables desde .env si existe
load_dotenv()

# Si no tienes .env, puedes poner el PROJECT_ID directamente:
PROJECT_ID = os.getenv("PROJECT_ID", "steel-autonomy-472502-f3")

# Crear cliente de Storage con project explícito
client = storage.Client(project=PROJECT_ID)

buckets = list(client.list_buckets())
print("Buckets disponibles:", [b.name for b in buckets])