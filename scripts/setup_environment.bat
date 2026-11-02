@echo off
setlocal
set "ROOT=%~dp0.."
if not defined ADAPTIVEAOI_PYTHON set "ADAPTIVEAOI_PYTHON=python"

echo [AdaptiveAOI] Project root: %ROOT%
"%ADAPTIVEAOI_PYTHON%" --version || goto :python_missing
if not exist "%ROOT%\requirements.txt" goto :requirements_missing

echo Installing declared Python dependencies only. No datasets or model weights will be downloaded by this script.
"%ADAPTIVEAOI_PYTHON%" -m pip install -r "%ROOT%\requirements.txt" || goto :install_failed
echo Environment setup completed.
exit /b 0

:python_missing
echo Python was not found. Set ADAPTIVEAOI_PYTHON to the adaptive_aoi interpreter.
exit /b 1
:requirements_missing
echo requirements.txt is missing: %ROOT%\requirements.txt
exit /b 1
:install_failed
echo Dependency installation failed. Review the pip output above.
exit /b 1
