@echo off
title MIA - Vectorizar Memoria
cd /d "%~dp0"
echo.
echo ══════════════════════════════════════
echo   Vectorizando sesiones de chat...
echo ══════════════════════════════════════
echo.
.venv\Scripts\python vectorize_memory.py
echo.
pause
