@echo off
title MIA - Borrar Memoria
cd /d "%~dp0"
echo.
.venv\Scripts\python clear_memory.py --all
echo.
pause
