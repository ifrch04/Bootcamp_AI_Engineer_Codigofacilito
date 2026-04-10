# Troubleshooting

Guía para resolver los errores más comunes de la API.

## Error 401 Unauthorized

**Causa:** Tu token de autenticación expiró, es inválido, o no lo estás enviando.

**Solución:**
1. Verifica que estás incluyendo el header `Authorization: Bearer <token>` en tu request
2. Si el token expiró, renuévalo con `POST /auth/refresh`
3. Si el refresh también falla, vuelve a autenticarte con `POST /auth/login`

## Error 403 Forbidden

**Causa:** Tu token es válido pero no tienes permisos para esta operación.

**Solución:**
1. Verifica que tu usuario tiene el rol necesario (admin, editor, viewer)
2. Contacta al administrador para solicitar permisos adicionales

## Error 429 Too Many Requests

**Causa:** Excediste el rate limit de 100 requests por minuto.

**Solución:**
1. Lee el header `Retry-After` en la respuesta para saber cuántos segundos esperar
2. Implementa exponential backoff en tu cliente
3. Si necesitas más capacidad, contacta a soporte para aumentar tu límite
4. Revisa tu código por loops que hagan requests innecesarios

## Error 500 Internal Server Error

**Causa:** Error inesperado en el servidor. No es tu culpa.

**Solución:**
1. Espera unos minutos y reintenta — puede ser un problema temporal
2. Si persiste, revisa la página de status en `status.example.com`
3. Reporta el error incluyendo el `request_id` del header de respuesta

## Database connection timeout

**Causa:** El servidor no puede conectarse a la base de datos.

**Solución:**
1. Verifica que `DATABASE_URL` está bien configurado en tu archivo `.env`
2. Confirma que el servidor de base de datos está corriendo: `pg_isready -h localhost -p 5432`
3. Revisa que no hay un firewall bloqueando la conexión
4. Si usas Docker, verifica que el contenedor de la DB está activo: `docker ps | grep postgres`
