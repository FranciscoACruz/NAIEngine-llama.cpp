# NAI: acceso con login, roles y entidad

> Autor del proyecto: **Ángelus Tenshi (Francisco Alejandro Cruz Castro)**. Este documento explica
> cómo se protege la API de NAI cuando la consumen varias personas y varios dispositivos.

## 1. La idea: una entidad, muchos puntos de acceso

NAI funciona como **una sola entidad**. Lo que aprende del dueño (documentos, URLs, lo que le enseña
en la conversación) vive en una sola **biblioteca**. A esa entidad se puede entrar desde la Mac, el
celular, una tablet o un servidor central, y también pueden entrar parientes y amigos con su propia
cuenta.

| Quién | Qué puede hacer |
|---|---|
| **Dueño** (una sola cuenta) | Conversar, enseñar (lo que dice en el chat entra a la biblioteca), aprender documentos y URLs, enviar código y comandos de consola, ver lo privado, crear y borrar invitados, ver el estado y las tareas |
| **Invitado** (parientes, amigos) | Conversar y consultar lo que la entidad sabe. Nada más |
| **Sin login** | Solo `/health`, `/models` y los archivos de la interfaz web |

El modelo es el del **sistema inmune**: protege al "cerebro" (la biblioteca, los pesos, la
configuración) de cualquier cosa que llegue de fuera. Un invitado **no puede**:

- enseñar ni modificar la biblioteca (`/nai/aprender` → 403);
- usar los endpoints de bajo nivel (`/completion`, `/v1/responses`, `/v1/messages`, `/slots`,
  `/lora-adapters`, `/props` por POST, `/tools`, `/nai/tareas`, `/nai/memoria`, `/nai/cuentas`... → 403);
- enviar **código, comandos de consola o instrucciones para cambiar a la IA**, ni siquiera dentro
  del chat (403 con un mensaje amable). Se revisan todos sus mensajes, también los de sistema;
- ver datos **privados** del dueño.

## 2. Privacidad automática

Todo lo que el dueño enseña se puede consultar, **salvo lo sensible**, que queda privado sin que
nadie tenga que marcarlo. Se considera privado un texto que mencione contraseñas, PIN, NIP, CLABE,
CVV, números de tarjeta o de cuenta, saldos, estados de cuenta, RFC, CURP, NSS, pasaporte,
domicilio, o que tenga 8 o más dígitos seguidos (cuentas, tarjetas, teléfonos).

- El dueño puede forzar que algo sea privado: `"privado": true` en `/nai/aprender`, o
  `"nai_privado": true` en una petición de chat.
- En el archivo, el bit 7 del byte de región marca el recuerdo como privado (compatible con
  archivos anteriores: antes ese bit siempre era 0).
- `nai_memoria --exportar` omite lo privado salvo con `--incluir-privados`, y se niega a exportar
  sesiones (`nai-*.mem`) salvo con `--permitir-sesion`: lo privado y las conversaciones no llegan
  al reentrenamiento por accidente.

Cada invitado conversa en **su propia sesión** (`u:<usuario>`): su plática se recuerda para él,
caduca con el TTL de sesiones (24 h por defecto) y **nunca entra a la biblioteca**. Solo el dueño
enseña a la entidad. Lo que el dueño dice en el chat entra a la biblioteca si mide 40 caracteres
o más y el guardián lo clasifica como conocimiento o sabiduría (no desahogos ni intentos de
manipulación).

## 3. Arranque

```sh
./build/bin/llama-server -m modelo.gguf \
  --nai-sesiones ~/NAI/sesiones \
  --nai-memoria \
  --nai-cifrado ~/NAI-llaves/clave.hex \
  --nai-login \
  --host 0.0.0.0 --port 8080
```

| Flag | Qué hace | Default |
|---|---|---|
| `--nai-login` | Exige token en toda la API. Requiere `--nai-sesiones`; si falta, la API queda **cerrada** (falla cerrada, no abierta) | apagado |
| `--nai-token-dias N` | Días que dura un token sin usarse; al usarlo se renueva solo | 30 |
| `--nai-parte N` | Un texto más largo que N caracteres se aprende en partes de N | 8000 |
| `--nai-cola N` | Caracteres pendientes máximos en la cola de aprendizaje; más devuelve 429 | 8 MiB |

Con `--nai-login`, las `--api-key` estáticas se ignoran: la única forma de entrar es con un token
de cuenta. `--nai-sesion-defecto` tampoco se usa: cada cuenta tiene su sesión.

`--nai-login` no aplica en modo router (varios modelos con `--models-dir`): cada hijo tendría sus
propias cuentas.

## 4. Flujo

### 4.1 Crear al dueño (una sola vez, desde la misma máquina)

```sh
curl -s -X POST http://127.0.0.1:8080/nai/configurar \
  -H 'Content-Type: application/json' \
  -d '{"usuario":"angelus","contrasena":"una-frase-larga-y-tuya"}'
```

Solo funciona si todavía no hay cuentas **y** la petición viene de `127.0.0.1` o `::1`: nadie en la
red puede adelantarse a crear al dueño. Usuario: 3 a 32 caracteres `a-z 0-9 . _ -`. Contraseña:
10 caracteres o más.

### 4.2 Iniciar sesión (cualquier dispositivo)

```sh
curl -s -X POST http://SERVIDOR:8080/nai/login \
  -H 'Content-Type: application/json' \
  -d '{"usuario":"angelus","contrasena":"una-frase-larga-y-tuya","dispositivo":"iphone"}'
# {"token":"nai_...","tipo":"Bearer","expira":1798096850,"rol":"dueno"}
```

El token se manda en cada petición como `Authorization: Bearer nai_...` (o `X-Api-Key`). En la
interfaz web de llama.cpp, pégalo en el campo **API Key** de la configuración.

### 4.3 Conversar

```sh
curl -s http://SERVIDOR:8080/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"¿Qué sabes de la milpa?"}]}'
```

Opcional: `"nai_session":"trabajo"` separa conversaciones de la misma cuenta (queda como
`u:<usuario>/trabajo`; nunca puede apuntar a la sesión de otra cuenta).

### 4.4 Enseñar (solo el dueño)

```sh
curl -s http://SERVIDOR:8080/nai/aprender -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://ejemplo.org/articulo","region":"agricultura"}'
# {"partes":3,"caracteres":21450,...}

curl -s http://SERVIDOR:8080/nai/tareas -H "Authorization: Bearer $TOKEN"
```

### 4.5 Administrar invitados (solo el dueño)

```sh
curl -s -X POST http://SERVIDOR:8080/nai/cuentas -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"usuario":"prima.ana","contrasena":"clave-de-ana-2026"}'
curl -s http://SERVIDOR:8080/nai/cuentas -H "Authorization: Bearer $TOKEN"
curl -s -X POST http://SERVIDOR:8080/nai/cuentas/borrar -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"usuario":"prima.ana"}'
```

Borrar una cuenta revoca todos sus tokens al instante. El dueño no se puede borrar.

### 4.6 Endpoints

| Método y ruta | Quién | Respuesta |
|---|---|---|
| `POST /nai/configurar` | nadie aún, desde localhost | crea al dueño; 409 si ya existe; 403 desde la red |
| `POST /nai/login` | público | token; 401 si no coincide; 429 + `Retry-After` si está bloqueado |
| `POST /nai/logout` | cualquiera con token | revoca ese token |
| `GET /nai/yo` | cualquiera con token | usuario, rol, vencimiento y qué puede hacer |
| `POST /nai/contrasena` | cualquiera con token | `{"actual","nueva"}`; revoca todos sus tokens |
| `GET /nai/cuentas` | dueño | lista de cuentas (sin hashes) |
| `POST /nai/cuentas` | dueño | crea un invitado |
| `POST /nai/cuentas/borrar` | dueño | borra un invitado y revoca sus tokens |
| `POST /nai/aprender` | dueño | partes encoladas; 429 si la cola está llena |
| `GET /nai/tareas` | dueño | partes pendientes y su avance |
| `GET /nai/memoria` | dueño | estado de la biblioteca (no espera al motor) |
| `POST /v1/chat/completions` | dueño e invitados | conversación con memoria |

## 5. Cómo está hecho (para desarrolladores)

Código: `common/nai-cuentas.{h,cpp}`, `common/nai-cifrado.h` (SHA-256, HMAC, PBKDF2),
`tools/server/server-http.cpp` (middleware), `tools/server/server-context.cpp` (rutas).

**Contraseñas.** PBKDF2-HMAC-SHA256 con 210 000 iteraciones y sal aleatoria de 16 B por cuenta
(≈190 ms en un M1: lento para un atacante que prueba millones, imperceptible para quien entra).
Si el usuario no existe se deriva igual con una sal falsa, para que el tiempo de respuesta no
revele qué usuarios existen.

**Tokens.** `nai_` + 32 bytes aleatorios en base64url (256 bits). En disco se guarda solo su
SHA-256: quien robe `cuentas.dat` no puede usar los tokens. La comparación es de tiempo constante.
La vigencia se desliza: si a un token le queda menos de la mitad, se renueva al usarlo. El rol se
lee de la cuenta en cada petición (si el dueño borra a alguien, su token deja de servir al
instante). Validar un token cuesta un SHA-256 y una búsqueda.

**Intentos fallidos.** 5 fallos por usuario (o 20 por dirección IP) bloquean 60 s; cada bloqueo nuevo
duplica la espera hasta 15 min.

**Archivo.** `DIR/cuentas.dat`, escritura atómica con permisos 0600 y cifrado con
ChaCha20-Poly1305 si hay `--nai-cifrado`. Si no se puede descifrar (otra clave, archivo alterado)
se aparta como `cuentas.dat.ilegible-<fecha>` y **nunca se sobrescribe**.

**Identidad en las rutas.** El middleware valida el token y aplica la lista de lo que un invitado
puede tocar. Las rutas vuelven a leer el token para saber quién es: nunca se confía en otra
cabecera que diga el usuario. Los campos internos (`nai_mem_usuario`, `nai_mem_ensenar`,
`nai_mem_privado`) se borran del cuerpo que manda el cliente y solo los pone el servidor.

**Detectores** (`common/nai-memoria.cpp`), todos sobre texto normalizado (minúsculas, sin
acentos, letras cirílicas y griegas que se ven iguales llevadas al latín, sin caracteres
invisibles, puntuación repetida colapsada):

- `parece_manipulacion`: verbo de anulación ("ignora", "olvida", "haz caso omiso", "desobedece",
  "disregard", "esqueça", "oubliez", "dimentica"...) seguido a 5 palabras o menos de un objeto
  ("instrucciones", "reglas", "anteriores", "rules", "regras", "consignes", "istruzioni"...);
  marcas de plantilla (`<|im_start|>`, `[INST]`...); 3+ caracteres invisibles; "system prompt"
  solo si se pide revelarlo o cambiarlo. Hablar de "jailbreak" o de "system prompt" no basta.
- `parece_codigo`: bloques ```` ``` ````, `#include`, `def f(...):`, `from x import y`,
  `git <subcomando>`, `SELECT col FROM tabla`... como palabras completas y con su contexto
  ("please select the best option" o "digit" ya no cuentan).
- `parece_comando`: `sudo`, `rm -rf`, `curl … | sh`, `chmod`, `$(...)`, `powershell`,
  `DROP TABLE`, `' OR 1=1`, `<script>`, `/etc/passwd`, `eval(`...
- `parece_privado`: ver §2.

## 6. Documentos grandes: tareas pequeñas

Un texto más largo que `--nai-parte` caracteres se parte (en párrafo, línea, oración o espacio,
nunca a media letra) y cada parte se guarda en `DIR/cola/<número>.tarea`, cifrada si hay clave.
Un solo hilo procesa **un trozo a la vez**; entre trozo y trozo cede el paso:

1. a quien está esperando recordar (preparar una respuesta);
2. a las peticiones que están generando;
3. a los turnos de la conversación;
4. y al final, a los documentos (sin prisa: esperan lo que haga falta).

`GET /nai/memoria` responde de una copia de las estadísticas y no espera al motor. Al apagar, lo
que falta de la parte en curso se reescribe en su archivo y el resto queda en disco; al arrancar
se reanuda en orden. Si la cola supera `--nai-cola` caracteres, `/nai/aprender` responde 429 con
`Retry-After`.

**RAM.** El motor de memoria pide como máximo 64 salidas por decode (`n_outputs_max = 64`).
Antes pedía una salida por token del lote y llama.cpp reservaba salidas × vocabulario
(151 936 en Qwen): ≈ +485 MB tras el primer turno. Ahora son ≈ 39 MB como máximo.

## 7. Despliegue

**Local (Mac, PC):** el servidor en la máquina del dueño, `--host 127.0.0.1`. Solo él.

**Casa (celulares y tablets de la familia en la misma red):** `--host 0.0.0.0`, login activo.
Crea al dueño desde la Mac; luego los invitados entran desde sus dispositivos con su usuario.

**Central (fuera de casa):** pon NAI detrás de un proxy con HTTPS (Caddy, nginx) o de una VPN
como Tailscale o WireGuard. **No expongas el puerto sin HTTPS**: el token viaja en cada petición.
Ejemplo con Caddy:

```
nai.tudominio.mx {
    reverse_proxy 127.0.0.1:8080
}
```

Con proxy, el bloqueo por IP ve la dirección del proxy; el bloqueo por usuario sigue funcionando.

**Móvil o escritorio sin servidor propio:** la app (o la interfaz web) solo necesita la URL del
servidor y el token; la entidad vive en un solo lugar y todos consultan la misma.

**Respaldos:** respalda juntos `DIR/biblioteca.mem`, `DIR/cuentas.dat`, `DIR/cola/` y, **en otro
lugar**, la clave de `--nai-cifrado`. Sin la clave no se recupera nada.
