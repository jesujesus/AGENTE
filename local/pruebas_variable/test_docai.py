import os
from dotenv import load_dotenv
from google.cloud import documentai

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "steel-autonomy-472502-f3")
LOCATION = os.getenv("LOCATION", "us")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")  # Asegúrate de tenerlo en tu .env

client = documentai.DocumentProcessorServiceClient()

name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
print("✅ Conexión exitosa a Document AI processor:")
print("   ", name)
