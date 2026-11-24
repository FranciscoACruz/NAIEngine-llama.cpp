# Prompt para preparar material de aprendizaje para NAI

Este prompt sirve para pedirle a otras IA (ChatGPT, Gemini, Copilot u otras) que preparen archivos que NAI pueda aprender con `POST /nai/aprender`. Cada archivo sale listo para cargarse: un tema, una región, párrafos que se entienden solos.

## 1. El prompt

Copia el bloque completo. Solo cambia las dos primeras líneas: el tema y la región.

```
TEMA: [escribe aquí el tema, por ejemplo: la enseñanza del amor en Rumi]
REGIÓN: [una de: conciencia, ética, espiritualidad, empatía, psique, filosofía, lógica, matemáticas, física, química, biología, ingeniería, mecánica, aerodinámica, ciberseguridad, supervivencia, agricultura, arte, medicina tradicional, finanzas personales, cocina, compras, deportes, trámites, entretenimiento, general]

Vas a preparar material de aprendizaje para NAI, una inteligencia artificial que aprende por fragmentos. Cada párrafo que escribas se guardará como un recuerdo independiente, así que el formato importa tanto como el contenido.

CONTENIDO
- Explica el TEMA con profundidad, priorizando la comprensión filosófica y espiritual sobre la puramente técnica o situacional: primero el sentido (para qué, por qué importa a una vida humana), después los datos.
- Si el tema toca tradiciones de sabiduría, cosmogonías o creencias, preséntalas con respeto y sin jerarquizarlas: ninguna es "la correcta". Cuando sea útil, muestra cómo distintas culturas han entendido lo mismo.
- Incluye, cuando existan, enseñanzas de vida: lo que las personas han aprendido viviendo, no solo lo que se estudia.

HONESTIDAD (obligatorio)
- No inventes citas. Si no estás seguro de que alguien dijo algo con esas palabras, no lo pongas entre comillas: parafrasea y di "se atribuye a" o "según la tradición".
- Distingue hechos, interpretaciones y leyendas ("según el mito...", "los historiadores creen...").
- Si un dato es dudoso o discutido, dilo en el mismo párrafo.
- No copies textos largos de libros: explica con tus palabras.

FORMATO (obligatorio)
- Solo párrafos de prosa, separados por una línea en blanco. Sin títulos, sin listas, sin tablas, sin negritas, sin emojis.
- Cada párrafo de 2 a 4 oraciones y menos de 600 caracteres.
- Cada párrafo debe entenderse solo, sin los demás: nombra siempre el sujeto. Escribe "Rumi enseñaba que..." y nunca "Él decía que..." ni "Como vimos antes...".
- Entre 15 y 30 párrafos por archivo.
- Español claro y natural.

NATURALEZA
Al final, después del último párrafo, escribe una sola línea así:
NATURALEZA: sabiduria
o
NATURALEZA: conocimiento
Usa "sabiduria" si el texto es sobre todo reflexión filosófica, espiritual o enseñanza de vida; "conocimiento" si es sobre todo información académica, técnica o práctica.

ENTREGA
Entrega un solo bloque de código con el texto, y antes del bloque el nombre del archivo con este formato exacto:
REGIÓN--naturaleza--tema-corto.md
Ejemplo: espiritualidad--sabiduria--rumi-el-amor.md
(nombre en minúsculas, sin espacios, guiones en lugar de espacios; la región conserva sus acentos).
```

## 2. Cargar los archivos en NAI

Guarda los archivos en una carpeta solo para ellos (no en la raíz del repo, donde el `README.md` se aprendería como material) y, con el servidor corriendo con `--nai-memoria`, ejecuta:

```console
mkdir -p ~/nai-aprendizaje
cd ~/nai-aprendizaje
for f in *--*.md; do
  region="${f%%--*}"
  grep -v '^NATURALEZA:' "$f" > /tmp/nai-envio.md
  curl -s http://localhost:8080/nai/aprender -H "Content-Type: application/json" -d "{\"archivo\":\"$(base64 -i /tmp/nai-envio.md)\",\"nombre\":\"$f\",\"region\":\"$region\"}"
  echo "  <- $f"
done
```

- Solo se envían los archivos con el formato `región--naturaleza--tema.md` (`*--*.md`).
- La región se toma del nombre del archivo (lo que va antes del primer `--`).
- `/tmp` ya existe en macOS y Linux; ahí se deja una copia temporal de cada archivo.
- La línea `NATURALEZA:` se quita antes de enviar, para que no se guarde como si fuera un recuerdo.
- En Linux, cambia `base64 -i archivo` por `base64 -w0 archivo`.
- Para ver el avance: `curl http://localhost:8080/nai/memoria`.

## 3. Antes de cargar

1. **Revisa cada archivo.** Las IA inventan citas y atribuciones aunque se les pida que no lo hagan, sobre todo con maestros espirituales. Lo que NAI aprende termina en su reentrenamiento: un error no revisado se vuelve parte de NAI.
2. **No pidas a ninguna IA que genere desahogos.** Un desahogo inventado no enseña nada verdadero sobre las personas. Esa naturaleza debe venir de conversaciones y textos reales; el guardián la reconoce solo.
3. **Mezcla fuentes.** Si NAI aprende solo de textos generados por IA, hereda la voz de esas IA y todo empieza a sonar igual. Pide el mismo tema a dos o tres IA distintas y combínalo con textos originales: libros, testimonios, notas propias.

## 4. Pendiente

La línea `NATURALEZA:` hoy sirve para revisar la clasificación a mano. Cuando `/nai/aprender` acepte el campo `"naturaleza"` (README §9.5), el script la enviará directamente y no dependerá del guardián.
