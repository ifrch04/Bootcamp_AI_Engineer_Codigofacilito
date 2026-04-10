# API Reference

## Authentication

Todas las rutas de la API requieren autenticación mediante un Bearer token en el header `Authorization`.

```
Authorization: Bearer <tu_token>
```

Para obtener un token, usa el endpoint `POST /auth/login` con tus credenciales.
Los tokens expiran después de 24 horas. Usa `POST /auth/refresh` para renovar un token antes de que expire.

## Endpoints

### GET /users

Lista todos los usuarios registrados. Soporta paginación.

**Query parameters:**
- `page` (int, default: 1) — Número de página
- `per_page` (int, default: 20, max: 100) — Resultados por página
- `sort` (string, default: "created_at") — Campo para ordenar: `name`, `email`, `created_at`

**Respuesta exitosa (200):**
```json
{
  "data": [{"id": 1, "name": "Ana", "email": "ana@example.com", "createdAt": "2026-01-15T10:00:00Z"}],
  "page": 1,
  "total_pages": 5
}
```

### POST /users

Crea un nuevo usuario.

**Body (JSON):**
```json
{
  "name": "string (requerido)",
  "email": "string (requerido, único)",
  "metadata": "object (opcional)"
}
```

**Respuesta exitosa (201):** Devuelve el usuario creado con su `id` asignado.

**Errores comunes:**
- `400` — Falta `name` o `email`
- `409` — El email ya está registrado

### GET /users/{id}

Obtiene un usuario por su ID.

**Path parameters:**
- `id` (int, requerido) — ID del usuario

**Respuesta exitosa (200):** Devuelve el objeto del usuario completo.

**Errores comunes:**
- `404` — El usuario no existe

### DELETE /users/{id}

Elimina un usuario por su ID. Esta operación es irreversible.

**Path parameters:**
- `id` (int, requerido) — ID del usuario

**Respuesta exitosa (204):** Sin cuerpo.

**Errores comunes:**
- `404` — El usuario no existe
- `403` — No tienes permisos para eliminar este usuario

## Rate Limiting

Todas las rutas están sujetas a un rate limit de 100 requests por minuto por API key.
Si excedes el límite, recibirás un error `429 Too Many Requests` con el header `Retry-After` indicando cuántos segundos esperar.
