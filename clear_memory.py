"""
clear_memory.py – Borra toda la memoria vectorizada de MIA.

Uso:
    python clear_memory.py          # Borra solo ChromaDB
    python clear_memory.py --all    # Borra ChromaDB + sesiones pendientes
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clear_memory")

SESSIONS_DIR = Path("data/chat_sessions")


def main():
    """Punto de entrada principal."""
    clear_sessions = "--all" in sys.argv

    # Confirmar
    print("⚠️  Esto va a borrar TODA la memoria vectorizada de MIA.")
    if clear_sessions:
        print("   También se borrarán las sesiones de chat pendientes.")
    print()
    confirm = input("¿Continuar? (s/N): ").strip().lower()
    if confirm not in ("s", "si", "sí", "y", "yes"):
        print("Cancelado.")
        return

    # Cargar config
    from mia.config import load_config
    config = load_config()

    # Borrar ChromaDB
    from mia.rag_memory import RAGMemory
    rag = RAGMemory(config.rag)
    if rag.enabled:
        count = rag.clear()
        logger.info("ChromaDB: %d documentos eliminados ✓", count)
    else:
        logger.warning("RAG no está habilitado en config.yaml")

    # Borrar sesiones pendientes
    if clear_sessions and SESSIONS_DIR.exists():
        files = list(SESSIONS_DIR.glob("*.jsonl"))
        for f in files:
            f.unlink()
        logger.info("Sesiones pendientes: %d archivos borrados ✓", len(files))

    print("\n✅ Memoria limpiada.")


if __name__ == "__main__":
    main()
