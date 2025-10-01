# test_gemini_embed.py
import os, requests
from dotenv import load_dotenv
load_dotenv("C:/Users/Admin/Desktop/local/.env")

key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={key}"
payload = {"model":"models/embedding-001","content":{"parts":[{"text":"hola mundo"}]}}
r = requests.post(url, json=payload, timeout=30)
print(r.status_code, r.text[:200])
