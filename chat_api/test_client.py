
import requests

url = "http://127.0.0.1:8000/chat"
data = {"pregunta": "¿Qué dice el documento alumno.pdf?"}

resp = requests.post(url, json=data)

print("Código de estado:", resp.status_code)
print("Respuesta JSON:", resp.json())
