# Changelog

Historial de cambios de la API, ordenado del más reciente al más antiguo.

## v2.3.0 — 2026-03-15

- Nuevo endpoint `POST /users/bulk` para crear múltiples usuarios en una sola request (max 50)
- Fix en el rate limiter que dejaba pasar requests después de alcanzar el límite en ventanas de tiempo concurrentes
- Nuevo campo `last_login` en la respuesta de `GET /users/{id}`
- Mejora en los mensajes de error de validación: ahora incluyen el campo específico que falló

## v2.2.0 — 2026-02-01

- **Breaking change:** `GET /users` ahora devuelve `createdAt` en formato ISO 8601 en vez de epoch timestamp
- Nuevo campo opcional `metadata` (objeto JSON libre) en el body de `POST /users`
- Soporte para ordenar resultados con el query parameter `sort` en `GET /users`
- Fix: `DELETE /users/{id}` ahora devuelve 404 en vez de 500 cuando el usuario no existe

## v2.1.0 — 2026-01-10

- Nuevo endpoint `POST /auth/refresh` para renovar tokens sin re-autenticarse
- Aumento del rate limit de 60 a 100 requests por minuto
- **Breaking change:** el campo `user_name` fue renombrado a `name` en todos los endpoints
- Mejora de performance: `GET /users` con paginación ahora responde en <100ms para datasets grandes
- Fix: el header `Retry-After` en respuestas 429 ahora reporta segundos correctamente (antes reportaba milisegundos)
