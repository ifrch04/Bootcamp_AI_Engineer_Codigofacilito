"""
Clase 4 - Snippet 03: JSON Mode
Obtener respuestas en formato JSON usando response_format.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"

# --- JSON Mode ---
response = client.chat.completions.create(
    model=MODEL,
    temperature=0.2,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": (
                "Extrae los datos de contacto del texto del usuario. "
                "Responde en JSON con los campos: nombre, email, telefono, ciudad."
            ),
        },
        {
            "role": "user",
            "content": (
                "Hola, soy Carlos Méndez. Me puedes contactar en "
                "carlos@ejemplo.com o al 555-1234. Vivo en Guadalajara."
            ),
        },
    ],
)

# El contenido es un string JSON que debemos parsear manualmente
contenido = response.choices[0].message.content
print("Respuesta cruda:")
print(contenido)
print(response.choices[0].message)
print()

# Parsear el JSON
datos = json.loads(contenido)
print("Datos parseados:")
for campo, valor in datos.items():
    print(f"  {campo}: {valor}")
