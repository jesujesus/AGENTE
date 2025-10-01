import os
from dotenv import load_dotenv

load_dotenv()

print("PROJECT_ID:", os.getenv("PROJECT_ID"))
print("LOCATION:", os.getenv("LOCATION"))
print("BUCKET_NAME:", os.getenv("BUCKET_NAME"))
print("DATASET:", os.getenv("DATASET"))
print("TABLE_EMBEDDINGS:", os.getenv("TABLE_EMBEDDINGS"))
print("TABLE_CONTROL:", os.getenv("TABLE_CONTROL"))
print("PROCESSOR_ID:", os.getenv("PROCESSOR_ID"))
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
print("GOOGLE_APPLICATION_CREDENTIALS:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
