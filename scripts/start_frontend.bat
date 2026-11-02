@echo off
setlocal
set "ROOT=%~dp0.."
if not defined ADAPTIVEAOI_PYTHON set "ADAPTIVEAOI_PYTHON=python"
if not defined ADAPTIVEAOI_FRONTEND_PORT set "ADAPTIVEAOI_FRONTEND_PORT=8501"
if not exist "%ROOT%\app\系统总览.py" (
  echo Streamlit application is missing: %ROOT%\app\系统总览.py
  exit /b 1
)
"%ADAPTIVEAOI_PYTHON%" -m streamlit --version || (echo Streamlit is unavailable in ADAPTIVEAOI_PYTHON.& exit /b 1)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%ADAPTIVEAOI_FRONTEND_PORT% .*LISTENING"') do (
  echo Port %ADAPTIVEAOI_FRONTEND_PORT% is already in use by PID %%P. Stop it or choose another Streamlit port.
  exit /b 1
)
cd /d "%ROOT%"
"%ADAPTIVEAOI_PYTHON%" -m streamlit run app\系统总览.py --server.address 127.0.0.1 --server.port %ADAPTIVEAOI_FRONTEND_PORT% --server.headless true --browser.gatherUsageStats false
