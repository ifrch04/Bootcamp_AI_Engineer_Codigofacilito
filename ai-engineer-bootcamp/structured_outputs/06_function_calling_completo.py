"""
Clase 4 - Snippet 06: Function Calling Completo
Flujo completo: modelo solicita función → ejecutamos → modelo responde.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"


class ConceptoFactura(BaseModel):
    descripcion: str = Field(description="Descripción del concepto facturado.")
    cantidad: int = Field(description="Cantidad del concepto.")
    precio_unitario: float = Field(description="Precio unitario del concepto.")
    total: float = Field(description="Total del concepto (cantidad x precio_unitario).")
class Factura(BaseModel):
    emisor: str = Field(description="Nombre del emisor de la factura.")
    receptor: str = Field(description="Nombre del receptor de la factura.")
    conceptos: list[ConceptoFactura] = Field(description="Lista de conceptos facturados.")
    total: float = Field(description="Total de la factura.")
    fecha: str = Field(description="Fecha de emisión de la factura en formato YYYY-MM-DD.")

# --- Definición de la tool ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_producto",
            "description": "Busca productos en el catálogo por nombre o categoría y el precio",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Término de búsqueda del producto.",
                    },
                    "precio": {
                        "type": "number",
                        "description": "Precio máximo en pesos mexicanos.",
                    },
                    "categoria": {
                        "type": "string",
                        "enum": ["electrónica", "ropa", "hogar", "deportes"],
                        "description": "Categoría del producto.",
                    }
                    
                },
                "required": ["query"],
            },
        },
    }
]


# --- Función simulada ---
def buscar_producto(query: str, categoria: str | None = None, precio: float | None = None) -> list[dict]:
    """Simula una búsqueda en base de datos."""
    productos = [
        {"nombre": "Laptop Pro 15", "precio": 24999, "categoria": "electrónica"},
        {"nombre": "Laptop Air 13", "precio": 18999, "categoria": "electrónica"},
        {"nombre": "Laptop Gamer X", "precio": 32999, "categoria": "electrónica"},
        {"nombre": "Manzana", "precio": 32999, "categoria": "comida"},
        {"nombre": "Naranja", "precio": 32999, "categoria": "comida"},
    ]
    consulta = []
    for producto in productos:
        if producto["categoria"] == categoria:
            consulta.append(producto)
        if producto["precio"] == precio:
            consulta.append(producto)
    return consulta


# === PASO 1: Enviar mensaje + tools al modelo ===
print("=== Paso 1: Enviando mensaje al modelo ===")

input_order = input("Por favor indica el producto que estás buscando...")

messages = [
    {"role": "user", "content": input_order},
]



response = client.chat.completions.create(
    model=MODEL,
    temperature=0.2,
    messages=messages,
    tools=tools,
)

mensaje_asistente = response.choices[0].message
print(f"El modelo quiere llamar una función: {mensaje_asistente.tool_calls is not None}")

print(mensaje_asistente.tool_calls)
tool_call = mensaje_asistente.tool_calls[0]
print(tool_call.function.arguments)
# === PASO 2: Ejecutar la función localmente ===
if mensaje_asistente.tool_calls:
    tool_call = mensaje_asistente.tool_calls[0]
    nombre_funcion = tool_call.function.name
    argumentos = json.loads(tool_call.function.arguments)

    print(f"\n=== Paso 2: Ejecutando '{nombre_funcion}' con args: {argumentos} ===")

    # Ejecutar la función
    resultado = buscar_producto(**argumentos)
    print(f"Resultado: {json.dumps(resultado, ensure_ascii=False)}")

    # === PASO 3: Enviar resultado al modelo ===
    print("\n=== Paso 3: Enviando resultado al modelo ===")
    messages.append(mensaje_asistente)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(resultado, ensure_ascii=False),
        }
    )

    response_final = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    print(f"\nRespuesta final del modelo:")
    print(response_final.choices[0].message.content)
