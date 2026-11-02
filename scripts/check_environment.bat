@echo off
setlocal
set "ROOT=%~dp0.."
if not defined ADAPTIVEAOI_PYTHON set "ADAPTIVEAOI_PYTHON=python"
if not defined ADAPTIVEAOI_BACKEND_PORT set "ADAPTIVEAOI_BACKEND_PORT=8000"

echo [1/6] Python
"%ADAPTIVEAOI_PYTHON%" --version || echo Python check failed. Set ADAPTIVEAOI_PYTHON.
echo [2/6] Node.js (not required by the current Streamlit frontend)
where node >nul 2>nul && node --version || echo Node.js not found; this is expected for the current Streamlit UI.
echo [3/6] Integration configuration
if exist "%ROOT%\configs\system_integration.yaml" (echo Found) else echo Missing
echo [4/6] Default memory bank
if exist "%ROOT%\outputs\checkpoints\mvtec\bottle\memory_bank.pt" (echo Found) else echo Missing
echo [5/6] Streamlit application
if exist "%ROOT%\app\系统总览.py" (echo Found) else echo Missing
echo [6/6] Backend health
curl -s http://127.0.0.1:%ADAPTIVEAOI_BACKEND_PORT%/api/health || echo Backend is not running yet.
