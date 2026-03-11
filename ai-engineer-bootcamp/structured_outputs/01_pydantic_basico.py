"""
Clase 4 - Snippet 01: Pydantic Básico
Validación de datos con Pydantic: modelos, tipos y errores.
"""

from pydantic import BaseModel, EmailStr, ValidationError


class Contacto(BaseModel):
    nombre: str
    email: EmailStr
    edad: int | None = None


contacto = Contacto(
    nombre="Carlos Méndez",
    email="carlos@ejemplo.com",
    edad=30,
)

print("Contacto válido:")
print(contacto)
print()

print("JSON:")
print(contacto.model_dump_json(indent=2))
print()

contacto_sin_edad = Contacto(
    nombre="Ana López",
    email="ana@ejemplo.com",
)

print("Contacto sin edad:")
print(contacto_sin_edad)
print()

print("Intentando crear contacto con datos inválidos...")
try:
    contacto_invalido = Contacto(
        nombre="Pedro",
        email="correo@correo.com",
        edad=30,
    )

    print(contacto_invalido.model_dump_json(indent=2))
except ValidationError as e:
    print(f"Error de validación:\n{e}")
    error_msg =  str(e)
    if 'value is not a valid email address' in error_msg:
        print("Tu correo no es válido, vuelvelo a mandar")
