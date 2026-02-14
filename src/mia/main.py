"""
main.py – Punto de entrada de MIA (Discord-only).

Carga configuración, inicializa pipeline y ejecuta el bot de Discord.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """Configura logging global (consola + archivo)."""
    from datetime import datetime

    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s"
    datefmt = "%H:%M:%S"

    # Crear directorio de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"mia_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler (con timestamps completos)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    logging.getLogger("mia").info("Log file: %s", log_file)

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
        pass

    setup_logging("INFO")
    logger = logging.getLogger("mia")

    # Verificar token de Discord
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.error(
            "DISCORD_BOT_TOKEN no encontrado. "
            "Crea un archivo .env con DISCORD_BOT_TOKEN=tu_token"
        )
        sys.exit(1)

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║         MIA – Discord Bot            ║")
    logger.info("║              v0.2.0                  ║")
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
        logger.error("Error cargando módulos: %s", exc, exc_info=True)
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
