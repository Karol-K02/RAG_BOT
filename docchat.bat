@echo off
rem Starts DocChat without a console window. Run from anywhere: uses this file's folder.
cd /d "%~dp0"
if not defined LLM_BASE_URL set LLM_BASE_URL=http://localhost:1234/v1
if not defined LLM_MODEL set LLM_MODEL=qwen/qwen3.5-9b
start "" ".venv\Scripts\pythonw.exe" app.py
