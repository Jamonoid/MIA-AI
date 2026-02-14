"""
tts_filter.py – Filtra texto antes de enviarlo al TTS.

Remueve contenido que no debería ser leído en voz alta:
asteriscos (*negrita*, *acción*), paréntesis, brackets, etc.
NO afecta el historial del LLM ni los subtítulos del chat.
"""

from __future__ import annotations

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


def tts_filter(text: str) -> str:
    """Aplica todos los filtros de texto para TTS.

    Args:
        text: Texto crudo del LLM.

    Returns:
        Texto limpio para sintetizar.
    """
    if not text or not text.strip():
        return text

    result = text

    # 1. Remover contenido entre asteriscos (*acción*, **negrita**)
    result = _filter_asterisks(result)

    # 2. Remover contenido entre paréntesis (acotaciones)
    result = _filter_nested(result, "(", ")")

    # 3. Remover contenido entre brackets [acciones]
    result = _filter_nested(result, "[", "]")

    # 4. Remover contenido entre angle brackets <metadata>
    result = _filter_nested(result, "<", ">")

    # 5. Remover caracteres especiales no pronunciables
    result = _remove_special_characters(result)

    # 6. Limpiar espacios múltiples
    result = re.sub(r"\s+", " ", result).strip()

    if result != text:
        logger.debug("TTS filter: '%s' → '%s'", text[:60], result[:60])

    return result


def _filter_asterisks(text: str) -> str:
    """Remueve texto encerrado en asteriscos (*, **, ***, etc.)."""
    filtered = re.sub(r"\*{1,}((?!\*).)*?\*{1,}", "", text)
    return re.sub(r"\s+", " ", filtered).strip()


def _filter_nested(text: str, left: str, right: str) -> str:
    """Remueve texto encerrado en símbolos anidados (paréntesis, brackets, etc.)."""
    if not text:
        return text

    result = []
    depth = 0
    for char in text:
        if char == left:
            depth += 1
        elif char == right:
            if depth > 0:
                depth -= 1
        else:
            if depth == 0:
                result.append(char)

    return "".join(result)


def _remove_special_characters(text: str) -> str:
    """Remueve caracteres no pronunciables (conserva letras, números, puntuación, espacios)."""
    normalized = unicodedata.normalize("NFKC", text)

    def is_valid(char: str) -> bool:
        cat = unicodedata.category(char)
        return (
            cat.startswith("L")   # Letras
            or cat.startswith("N")  # Números
            or cat.startswith("P")  # Puntuación
            or char.isspace()
        )

    return "".join(c for c in normalized if is_valid(c))
