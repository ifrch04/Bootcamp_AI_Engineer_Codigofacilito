"""
Clase 4 - Actividad: Extractor de Recetas Completo
4 partes: contrato, extracción, textos difíciles y function calling.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from ollama import chat

load_dotenv()
#
#client = OpenAI(
#    base_url="https://api.groq.com/openai/v1",
#    api_key=os.environ.get("GROQ_API_KEY"),
#)
#MODEL = "openai/gpt-oss-120b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # requerido pero ignorado
)

MODEL = "qwen3-coder:latest" 



# =============================================
# PARTE 1: Contrato Pydantic
# =============================================

class Ingrediente(BaseModel):
    nombre: str = Field(description="Nombre del ingrediente.")
    cantidad: float = Field(description="Cantidad numérica del ingrediente.")
    unidad: str = Field(description="Unidad de medida: piezas, gramos, ml, cucharadas, etc.")


class Receta(BaseModel):
    titulo: str = Field(description="Nombre del platillo.")
    tiempo_minutos: int = Field(description="Tiempo total de preparación en minutos.")
    dificultad: str = Field(description="Nivel de dificultad: fácil, media o difícil.")
    ingredientes: list[Ingrediente] = Field(description="Lista de ingredientes.")
    pasos: list[str] = Field(description="Pasos de preparación en orden.")
    porciones: int = Field(default=None, description="Número de porciones que rinde la receta.")
    costo_preparacion: float = Field(default=None, description="Costo estimado de preparación en moneda local.")

print("=" * 60)
print("  PARTE 1: Contrato Pydantic")
print("=" * 60)
print(json.dumps(Receta.model_json_schema(), indent=2, ensure_ascii=False))


# =============================================
# PARTE 2: Extracción del texto base
# =============================================

SYSTEM_PROMPT = (
    "Eres un chef experto. Extrae la receta del texto del usuario "
    "en el formato estructurado solicitado. Convierte cantidades "
    "informales (como 'un cuarto de kilo' o 'al gusto') a números "
    "precisos usando tu mejor estimación. IMPORTANTE: Usa exactamente "
    "los nombres de campo del schema (nombre, cantidad, unidad)."
    "Por favor también calcula el costo con ingredientes de calidad y las porciones basándote en los ingredientes y cantidades, usando precios promedio locales."
)

texto_chilaquiles = """
mira para los chilaquiles verdes necesitas como medio kilo de totopos
o si quieres los haces con tortillas pero es más trabajo, también unos
3 tomates verdes, un diente de ajo, cebolla al gusto, crema y queso
fresco para servir. Ah y chile serrano como 2 o 3 dependiendo qué tan
picoso te guste. Primero hierves los tomates con el chile y el ajo como
15 minutos, luego los licuas con un poco de agua. En una sartén con
aceite fríes la salsa como 5 minutos, le pones sal, y ya al final
echas los totopos y mezclas bien. Se sirven con crema, queso y cebolla.
En total como en 30 minutos los tienes listos, es bastante fácil.
"""


def extraer_receta(texto: str) -> Receta:
    """Extrae una receta de un texto libre usando Structured Outputs."""
    response = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,
        response_format=Receta,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto},
        ],
    )
    return response.choices[0].message.parsed


def imprimir_receta(receta: Receta) -> None:
    """Imprime una receta formateada."""
    print(f"\n  {receta.titulo.upper()}")
    print(f"  Tiempo: {receta.tiempo_minutos} min | Dificultad: {receta.dificultad} | Costo: ${receta.costo_preparacion:.2f} | Porciones: {receta.porciones}")
    print("-" * 50)
    print("  Ingredientes:")
    for ing in receta.ingredientes:
        print(f"    - {ing.cantidad} {ing.unidad} de {ing.nombre}")
    print("  Pasos:")
    for i, paso in enumerate(receta.pasos, 1):
        print(f"    {i}. {paso}")
    print()


print("\n" + "=" * 60)
print("  PARTE 2: Extracción del texto base")
print("=" * 60)

receta_chilaquiles = extraer_receta(texto_chilaquiles)
imprimir_receta(receta_chilaquiles)


# =============================================
# PARTE 3: Textos difíciles
# =============================================

texto_mole = """
para un buen mole poblano necesitas un cuarto de kilo de chile ancho
y como doscientos gramos de chile mulato, también unos cien gramos de
chocolate de mesa, una tablilla pues, medio pan bolillo del día anterior
para espesar, un puñito de ajonjolí tostado, pasas como dos cucharadas,
almendras peladas otro tanto, un plátano macho bien maduro, unas cinco
especias: clavo, pimienta, canela en raja, comino y anís en poca
cantidad de cada una. Primero tuestas los chiles en el comal sin
quemarlos, los remojas en agua caliente media hora, mientras tanto fríes
el pan, el plátano, las almendras y las pasas por separado. Ya que
están suaves los chiles los licuas con todo lo frito, las especias y un
poco del agua donde remojaste. Cuelas la salsa y la fríes en manteca
como cuarenta y cinco minutos moviendo seguido para que no se pegue, al
final agregas el chocolate y dejas que se derrita. Es un platillo
difícil que toma como dos horas y media.
"""

texto_tacos_pastor = """
ok los tacos al pastor llevan lo siguiente te voy diciendo, compras
como un kilo de carne de cerdo en bistec finito y la marinas con una
salsa que haces de 4 chiles guajillo desvenados remojados en agua
caliente y licuados con un poco de vinagre de piña como 3 cucharadas,
achiote una cucharada, orégano y comino al gusto y sal. Eso lo dejas
marinando mínimo 2 horas. Luego vas poniendo la carne en un sartén
bien caliente con un poco de aceite, la vas cortando chiquita como
en los tacos de la calle. Aparte cortas piña en rodajas y las asas
en el mismo sartén hasta que se doren. Calientas tortillas, pones la
carne, la piña picada encima, cebolla picada y cilantro, y salsa
verde al gusto. Con el kilo te salen como unos 15 tacos buenos,
en una hora ya los tienes.
"""

print("=" * 60)
print("  PARTE 3: Textos difíciles")
print("=" * 60)

print("\n--- Texto con cantidades en letras ---")
receta_mole = extraer_receta(texto_mole)
imprimir_receta(receta_mole)

print("--- Texto con pasos mezclados con ingredientes ---")
receta_tacos = extraer_receta(texto_tacos_pastor)
imprimir_receta(receta_tacos)


# =============================================
# PARTE 4 (Bonus): Function Calling
# =============================================

print("=" * 60)
print("  PARTE 4 (Bonus): Function Calling")
print("=" * 60)

# Almacén en memoria de recetas extraídas
recetas_guardadas: list[Receta] = [
    receta_chilaquiles,
    receta_mole,
    receta_tacos,
]

# Definición de tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "extraer_receta",
            "description": "Extrae una receta estructurada de un texto libre con instrucciones de cocina.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texto": {
                        "type": "string",
                        "description": "Texto libre que contiene una receta de cocina.",
                    },
                },
                "required": ["texto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_recetas",
            "description": "Busca en las recetas guardadas por nombre o ingrediente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "termino": {
                        "type": "string",
                        "description": "Término de búsqueda (nombre del platillo o ingrediente).",
                    },
                },
                "required": ["termino"],
            },
        },
    },
]


def ejecutar_buscar_recetas(termino: str) -> str:
    """Busca en recetas guardadas por nombre o ingrediente."""
    resultados = []
    termino_lower = termino.lower()
    for receta in recetas_guardadas:
        if termino_lower in receta.titulo.lower():
            resultados.append(receta.model_dump())
            continue
        for ing in receta.ingredientes:
            if termino_lower in ing.nombre.lower():
                resultados.append(receta.model_dump())
                break
    if not resultados:
        return json.dumps({"mensaje": f"No se encontraron recetas con '{termino}'"}, ensure_ascii=False)
    return json.dumps(resultados, ensure_ascii=False)


def ejecutar_extraer_receta(texto: str) -> str:
    """Extrae una receta y la guarda en el almacén."""
    receta = extraer_receta(texto)
    recetas_guardadas.append(receta)
    return receta.model_dump_json()


def procesar_mensaje(mensaje_usuario: str) -> str:
    """Procesa un mensaje del usuario usando function calling."""
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente de cocina. Puedes extraer recetas de textos "
                "o buscar en las recetas que ya tienes guardadas. Usa las "
                "herramientas disponibles según lo que el usuario necesite."
            ),
        },
        {"role": "user", "content": mensaje_usuario},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    mensaje_asistente = response.choices[0].message

    # Si no hay tool calls, retornar la respuesta directa
    if not mensaje_asistente.tool_calls:
        return mensaje_asistente.content

    # Ejecutar la tool call
    tool_call = mensaje_asistente.tool_calls[0]
    nombre = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"  [Tool call: {nombre}({json.dumps(args, ensure_ascii=False)[:80]}...)]")

    if nombre == "extraer_receta":
        resultado = ejecutar_extraer_receta(**args)
    elif nombre == "buscar_recetas":
        resultado = ejecutar_buscar_recetas(**args)
    else:
        resultado = json.dumps({"error": f"Función desconocida: {nombre}"})

    # Enviar resultado al modelo
    messages.append(mensaje_asistente)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": resultado,
        }
    )

    response_final = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    return response_final.choices[0].message.content


# --- Probar function calling ---
print("\n--- Prueba: Buscar recetas con chile ---")
respuesta = procesar_mensaje("¿Qué recetas tienes que lleven chile?")
print(f"\n{respuesta}")

print("\n--- Prueba: Buscar recetas de tacos ---")
respuesta = procesar_mensaje("¿Tienes alguna receta de tacos?")
print(f"\n{respuesta}")

print(f"\nTotal de recetas guardadas: {len(recetas_guardadas)}")
