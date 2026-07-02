"""Genera una explicación en lenguaje natural de la clasificación usando Claude."""
import anthropic

CLAUDE_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """Eres un ingeniero experto en mantenimiento de plantas solares fotovoltaicas,
especializado en interpretar termografías infrarrojas de módulos solares.

Vas a recibir el resultado de un clasificador de Deep Learning que analizó una imagen térmica
de un módulo o celda solar. Tu tarea es explicarle al usuario, en español, en 3-5 oraciones:
1. Qué significa la clase detectada en términos físicos/eléctricos.
2. Qué tan confiable es la predicción dado el nivel de confianza.
3. Qué acción recomendarías tomar (inspección, mantenimiento, ninguna acción si está sano, etc.).

Sé directo y concreto. No repitas el nombre de la clase como si fuera toda tu respuesta."""


def get_explanation(resultado_clasificacion: dict) -> str:
    client = anthropic.Anthropic()

    clase = resultado_clasificacion["clase_predicha"]
    confianza = resultado_clasificacion["confianza"]
    top3 = resultado_clasificacion["probabilidades"][:3]

    contexto = (
        f"Clase detectada: {clase['es']} ({clase['en']})\n"
        f"Confianza: {confianza * 100:.1f}%\n"
        f"Top 3 probabilidades: "
        + ", ".join(f"{p['es']} ({p['prob'] * 100:.1f}%)" for p in top3)
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": contexto}],
    )

    return next(block.text for block in response.content if block.type == "text")
