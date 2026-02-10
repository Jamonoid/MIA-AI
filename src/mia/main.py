"""
main.py – Punto de entrada de MIA.

Carga configuración, inicializa pipeline y ejecuta el loop principal.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """Configura logging global."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silenciar loggers ruidosos
    for noisy in ["httpx", "httpcore", "urllib3", "chromadb", "onnxruntime"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    """Punto de entrada principal."""
    setup_logging("INFO")
    logger = logging.getLogger("mia")

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║             MIA – AI                 ║")
    logger.info("║              v0.1.0                  ║")
    logger.info("╚══════════════════════════════════════╝")

    # Cargar config
    from .config import load_config

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml no encontrado en %s", Path.cwd())
        sys.exit(1)

    config = load_config(config_path)
    logger.info("Config cargado ✓")

    # Crear y cargar pipeline
    from .pipeline import MIAPipeline

    pipeline = MIAPipeline(config)

    try:
        pipeline.load()
    except Exception as exc:
        logger.error("Error cargando módulos: %s", exc)
        logger.error(
            "Verifica que los modelos están en las rutas correctas "
            "según config.yaml"
        )
        sys.exit(1)

    # Manejar Ctrl+C
    loop = asyncio.new_event_loop()

    def _signal_handler() -> None:
        logger.info("Señal de interrupción recibida...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    # Ejecutar
    try:
        loop.run_until_complete(pipeline.run())
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
        loop.run_until_complete(pipeline.shutdown())
    finally:
        loop.close()
        logger.info("Hasta luego 👋")


if __name__ == "__main__":
    main()
