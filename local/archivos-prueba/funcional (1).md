📄 Requerimiento: Bot de Telegram para Recojo de Equipos (Logibot)
🎯 Objetivo del proyecto

Desarrollar un bot conversacional en Telegram que coordine automáticamente la recolección de equipos de clientes (ej. módems), reemplazando la interacción manual actual. El bot debe simular la naturalidad de una conversación humana y asegurar la correcta captura de datos necesarios para la logística.

🧩 Contexto

Actualmente, un equipo de 3-4 personas coordina manualmente vía WhatsApp.

Se dispone de CSVs con registros de conversaciones previas, que contienen distintos flujos, errores comunes y casos especiales.

Queremos usar estos datos como dataset de entrenamiento/análisis para mejorar la comprensión del bot y cubrir la mayor variedad de casos reales.

📋 Requerimientos funcionales

Plataforma inicial: Telegram Bot API (con proyección futura a WhatsApp Business API).

Flujo principal:

Saludo / presentación.

Captura de datos en etapas:

Dirección.

Turno disponible (mañana / tarde / noche).

Persona que entregará.

Teléfono de contacto.

Tipo/cantidad de equipos (opcional).

Confirmación de datos al usuario.

Registro final en base de datos.

Casos especiales a manejar:

Cliente ya entregó equipo.

Cliente rechaza entrega.

Cliente responde fuera de contexto (“no sé”, “luego”, insultos).

Cliente hace preguntas adicionales no relacionadas.

Persistencia:

Guardar datos en SQLite (mínimo) o BD escalable.

Exportación a CSV/Excel.

Integración futura con backend vía API REST.

📊 Requerimientos de IA

Uso de dataset (CSVs):

Analizar conversaciones reales para identificar patrones de respuesta.

Entrenar/testear un modelo que permita entender entradas ambiguas o fuera del flujo.
c
Detectar intenciones y clasificarlas (ej. confirmación, rechazo, fuera de contexto).

IA en el bot:

No solo flujo rígido → debe tolerar lenguaje natural.

Reconocimiento de variaciones en dirección, nombres y teléfonos.

Proponer respuestas automáticas en casos ambiguos, con opción de repreguntar.

Métrica esperada:

% de conversaciones completadas sin intervención humana.

% de datos capturados correctamente frente a dataset de referencia.

🚀 Entregables

Prototipo funcional en Telegram.

Diseño de flujo conversacional y casos de prueba.

Script que use los CSVs para entrenar/validar el modelo.

Informe de resultados del análisis de los datos y métricas de exactitud.

Documentación técnica (instalación, uso, dataset).

📌 Alcance de esta primera fase

Versión mínima (MVP) en Telegram.

Manejo de casos comunes + 3 casos especiales (ya entregó, rechazo, fuera de contexto).

Persistencia local en SQLite.

Uso de CSVs como base para análisis de intents y validación.