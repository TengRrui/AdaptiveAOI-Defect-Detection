@echo off
setlocal
set "ROOT=%~dp0.."
if not defined ADAPTIVEAOI_PYTHON set "ADAPTIVEAOI_PYTHON=python"
if not defined ADAPTIVEAOI_BACKEND_PORT set "ADAPTIVEAOI_BACKEND_PORT=8000"
if not exist "%ROOT%\configs\system_integration.yaml" (
  echo Integration configuration is missing: %ROOT%\configs\system_integration.yaml
  exit /b 1
)
"%ADAPTIVEAOI_PYTHON%" --version || (echo Python was not found. Set ADAPTIVEAOI_PYTHON.& exit /b 1)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%ADAPTIVEAOI_BACKEND_PORT% .*LISTENING"') do (
  echo Port %ADAPTIVEAOI_BACKEND_PORT% is already in use by PID %%P. Stop it or change ADAPTIVEAOI_BACKEND_PORT.
  exit /b 1
)
cd /d "%ROOT%"
"%ADAPTIVEAOI_PYTHON%" scripts\run_system_api.py
