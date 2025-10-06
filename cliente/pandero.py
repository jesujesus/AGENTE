import streamlit as st
import requests
import time
from typing import Dict, Any, List

# --- Configuración de la Interfaz ---
st.set_page_config(
    page_title="Agente Pandero IA",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Definir el color azul principal de Pandero (ejemplo)
PANDERO_BLUE = "#005BAA" # Azul oscuro profesional
PANDERO_LIGHT_BLUE = "#E0F2FF" # Azul muy claro para fondos

# Aplicar estilo CSS para el color principal y la fuente
st.markdown(f"""
    <style>
        .stButton>button {{
            background-color: {PANDERO_BLUE};
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background-color: #004D99; /* Azul ligeramente más oscuro al pasar el ratón */
            transform: translateY(-1px);
        }}
        .stTextInput>div>div>input {{
            border: 2px solid {PANDERO_BLUE};
            border-radius: 10px;
            padding: 10px;
        }}
        /* Estilo para el título */
        .title-text {{
            color: {PANDERO_BLUE};
            font-weight: 800;
        }}
    </style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown('<h1 class="title-text">🤖 Agente Pandero IA</h1>', unsafe_allow_html=True)
st.markdown("Escribe tu pregunta para consultar información sobre productos financieros.")
st.markdown("---")

# --- Lógica de la Aplicación (Modo SIMULACIÓN) ---

# Bandera para activar/desactivar la simulación
SIMULATION_MODE = True # Por defecto: SIMULACIÓN ACTIVA
API_URL = "http://localhost:8000/ask"

def mock_api_response(question: str) -> Dict[str, Any]:
    """Simula una respuesta del backend sin hacer una llamada de red."""
    time.sleep(1.5) # Simula un poco de latencia
    
    return {
        "respuesta": f"¡Hola! Gracias por preguntar sobre **'{question}'**. Como ejemplo, el financiamiento de Pandero para autos nuevos tiene un proceso de evaluación que dura aproximadamente 48 horas y requiere DNI, comprobante de ingresos y solicitud firmada. (Respuesta en modo SIMULACIÓN)",
        "referencias": [
            {
                "titulo": "Procedimiento de Evaluación Crediticia - 2024",
                "uri": "https://pandero.com/docs/evaluacion.pdf"
            },
            {
                "titulo": "Requisitos Legales",
                "uri": "https://pandero.com/requisitos.html"
            }
        ]
    }

# Campo de entrada de texto para la pregunta del usuario
question = st.text_input(
    "**Haz tu pregunta:**",
    placeholder="Ej: ¿Qué documentos necesito para un crédito de vehículo usado?"
)

# Botón para enviar la consulta
if st.button("Enviar Consulta ➡️", type="primary"):
    if not question:
        st.warning("Por favor, ingresa una pregunta para continuar.")
    else:
        # Indicador de carga
        with st.spinner('Consultando al Agente Pandero...'):
            data = None
            
            if SIMULATION_MODE:
                data = mock_api_response(question)
            else:
                # Lógica para la API real (se ejecuta solo si SIMULATION_MODE = False)
                try:
                    resp = requests.post(API_URL, json={"question": question})
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error(f"⚠️ Error de conexión: No se pudo conectar al servidor en {API_URL}.")
                    st.caption("Asegúrate de que tu aplicación FastAPI esté corriendo en el puerto 8000.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Error del servidor (HTTP): {e}")
                except Exception as e:
                    st.error(f"🚨 Ocurrió un error inesperado: {e}")
            
            # --- Renderizar la Respuesta ---
            if data:
                st.subheader("✅ Respuesta del Agente:")
                
                respuesta = data.get("respuesta", "No se encontró respuesta en el formato esperado.")
                st.markdown(f'<div style="background-color: {PANDERO_LIGHT_BLUE}; padding: 15px; border-radius: 10px; border-left: 5px solid {PANDERO_BLUE};">{respuesta}</div>', unsafe_allow_html=True)
                
                referencias = data.get("referencias")
                if referencias:
                    st.subheader("📚 Referencias/Fuentes:")
                    st.json(referencias)
                else:
                    st.caption("No se proporcionaron fuentes para esta respuesta.")

# Pie de página o instrucciones
st.markdown("---")
if SIMULATION_MODE:
    st.caption("🟢 **MODO SIMULACIÓN ACTIVO.** Ejecuta el backend y cambia `SIMULATION_MODE = False` para la conexión real.")
else:
    st.caption(f"🔴 Conectando a la API real en: `{API_URL}`")
