"""
vectorize_memory.py – Vectoriza sesiones de chat usando LLM para curar relevancia.

Uso:
    python vectorize_memory.py

Lee archivos .jsonl de data/chat_sessions/, usa el LLM (OpenRouter) para
decidir qué es relevante, vectoriza solo los resúmenes curados en ChromaDB,
y borra los archivos procesados.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorize")

SESSIONS_DIR = Path("data/chat_sessions")

CURATION_PROMPT = """\
Eres un curador de memoria para una IA llamada MIA. Te doy un fragmento de conversación.
Decide si contiene información RELEVANTE para recordar a largo plazo.

Información relevante:
- Datos personales del usuario (nombre, gustos, trabajo, relaciones)
- Opiniones fuertes o preferencias expresadas
- Eventos importantes mencionados
- Promesas o compromisos hechos por MIA o el usuario
- Información técnica útil compartida
- Contexto emocional significativo

NO es relevante:
- Saludos genéricos ("hola", "qué tal", "bye")
- Bromas sin contexto duradero
- Preguntas ya respondidas sin valor futuro
- Repeticiones de información ya conocida
- Comandos técnicos del sistema
- Mensajes de debug o error

Responde SOLO con una de estas dos opciones:
- "GUARDAR: <resumen conciso de lo relevante>" si hay algo que recordar
- "DESCARTAR" si no hay nada relevante

Conversación:
{conversation}"""


def load_config():
    """Carga config.yaml y retorna MIAConfig."""
    from mia.config import load_config as _load
    return _load()


def create_llm_client(config):
    """Crea cliente OpenAI para curación."""
    import os
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai no instalado. pip install openai")
        sys.exit(1)

    api_key = config.llm.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("No hay API key de OpenRouter. Configúrala en config.yaml o OPENROUTER_API_KEY")
        sys.exit(1)

    base_url = config.llm.base_url or "https://openrouter.ai/api/v1"
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/Jamonoid/MIA-AI",
            "X-Title": "MIA - AI Memory Vectorizer",
        },
    )
    return client


def curate_with_llm(client, model: str, conversation_text: str) -> str | None:
    """Pregunta al LLM si el fragmento es relevante. Retorna resumen o None."""
    prompt = CURATION_PROMPT.format(conversation=conversation_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()

        if result.upper().startswith("GUARDAR:"):
            summary = result[len("GUARDAR:"):].strip()
            return summary
        return None

    except Exception as exc:
        logger.warning("Error LLM curación: %s", exc)
        return None


def create_rag(config):
    """Crea instancia de RAGMemory."""
    from mia.rag_memory import RAGMemory
    return RAGMemory(config.rag)


def main():
    """Punto de entrada principal."""
    if not SESSIONS_DIR.exists() or not list(SESSIONS_DIR.glob("*.jsonl")):
        logger.info("No hay sesiones pendientes en %s", SESSIONS_DIR)
        return

    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    logger.info("Encontradas %d sesiones para procesar", len(session_files))

    # Cargar config y módulos
    config = load_config()
    client = create_llm_client(config)
    rag = create_rag(config)

    if not rag.enabled:
        logger.error("RAG no está habilitado en config.yaml")
        sys.exit(1)

    stats = {"processed": 0, "saved": 0, "discarded": 0, "errors": 0}

    for session_file in session_files:
        logger.info("─── Procesando: %s ───", session_file.name)

        try:
            messages = []
            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
        except Exception as exc:
            logger.error("Error leyendo %s: %s", session_file.name, exc)
            stats["errors"] += 1
            continue

        if not messages:
            logger.info("  Sesión vacía, borrando...")
            session_file.unlink()
            continue

        # Agrupar en pares user/assistant (ventana de 4 mensajes = 2 turnos)
        window_size = 4
        for i in range(0, len(messages), window_size):
            window = messages[i:i + window_size]
            conversation_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in window
            )

            stats["processed"] += 1

            # Curar con LLM
            summary = curate_with_llm(client, config.llm.model_name, conversation_text)

            if summary:
                # Vectorizar el resumen
                rag.ingest(conversation_text, summary)
                stats["saved"] += 1
                logger.info("  ✅ GUARDADO: %s", summary[:80])
            else:
                stats["discarded"] += 1
                logger.info("  ⏭️  Descartado")

            # Rate limiting mínimo
            time.sleep(0.3)

        # Borrar archivo procesado
        session_file.unlink()
        logger.info("  🗑️  Archivo borrado: %s", session_file.name)

    # Resumen
    logger.info("═══════════════════════════════════════")
    logger.info("Fragmentos procesados: %d", stats["processed"])
    logger.info("Memorias guardadas:    %d", stats["saved"])
    logger.info("Descartados:           %d", stats["discarded"])
    logger.info("Errores:               %d", stats["errors"])
    logger.info("═══════════════════════════════════════")


if __name__ == "__main__":
    main()
