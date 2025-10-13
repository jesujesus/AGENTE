import streamlit as st
import requests
from typing import Dict, Any, List

# =========================
# Config básica
# =========================
st.set_page_config(page_title="Agente Pandero IA", layout="centered")

# Ajusta aquí si cambiaste el puerto del backend
API_URL = "http://127.0.0.1:8080/ask"   # FastAPI en 8080
#API_URL = "https://prize-nano-outstanding-handhelds.trycloudflare.com/ask"
SIMULATION_MODE = False                  # True = sin backend, respuesta mock

st.title("🤖 Agente Pandero IA (MVP)")


# =========================
# Estado de sesión (historial)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}, ...]

# =========================
# Funciones auxiliares
# =========================
def mock_api_response(question: str) -> Dict[str, Any]:
    return {
        "respuesta": f"Demo: recibí tu pregunta «{question}». (SIMULACIÓN)",
        "referencias": []
    }

def call_backend(question: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {
        "question": question,
        "top_k": 3,
        "temperature": 0.3,
        "max_output_tokens": 2048,
        "session_id": "frontend-session-1",
        "history": history[-6:],
    }
    resp = requests.post(API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    # Backend real
    if "answer" in data:
        return {"answer": data.get("answer", ""), "sources": data.get("sources", [])}
    # Mock antiguo
    if "respuesta" in data:
        return {"answer": data.get("respuesta", ""), "sources": data.get("referencias", [])}
    return {"answer": "No se encontró respuesta.", "sources": []}

# =========================
# UI mínima
# =========================
with st.form("ask_form", clear_on_submit=False):
    question = st.text_input("Tu pregunta:", placeholder="Ej: ¿Qué documentos necesito para ...?")
    submitted = st.form_submit_button("Enviar")

if submitted:
    if not question:
        st.warning("Escribe una pregunta.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            raw = mock_api_response(question) if SIMULATION_MODE else call_backend(question, st.session_state.messages)
            data = normalize_response(raw)
            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
        except requests.exceptions.ConnectionError:
            st.error(f"No se pudo conectar a {API_URL}. ¿El backend está corriendo?")
        except requests.exceptions.HTTPError as e:
            st.error(f"Error HTTP: {e}")
            try:
                st.json(raw)  # si llegó JSON de error
            except Exception:
                pass
        except Exception as e:
            st.error(f"Error inesperado: {e}")

# Mostrar conversación
st.divider()
st.subheader("Conversación")
for m in st.session_state.messages:
    speaker = "Tú" if m["role"] == "user" else "Agente"
    st.markdown(f"**{speaker}:** {m['content']}")

# Controles rápidos
col1, col2 = st.columns(2)
with col1:
    if st.button("Limpiar chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.toggle("Modo simulación", value=SIMULATION_MODE, key="sim_flag", help="Si se activa, no llama al backend.")
    # Nota: si cambias este toggle, re-lanza la app o úsalo para setear SIMULATION_MODE global.
