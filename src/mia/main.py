"""
main.py – Punto de entrada de MIA.

Carga configuración, inicializa pipeline y ejecuta el loop principal.
"""

from __future__ import annotations

import asyncio
import logging
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
    # Cargar variables de entorno (.env)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv no instalado, variables se leen del entorno

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

    # Ejecutar
    async def _run() -> None:
        try:
            await pipeline.run()
        except KeyboardInterrupt:
            pass
        except asyncio.CancelledError:
            pass
        finally:
            await pipeline.shutdown()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    finally:
        logger.info("Hasta luego 👋")


if __name__ == "__main__":
    main()
