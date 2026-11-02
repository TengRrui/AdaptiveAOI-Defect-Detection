@echo off
setlocal
set "LAUNCHER_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER_ROOT%start_adaptiveaoi.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo AdaptiveAOI startup failed. See the System outputs\system_runtime\logs\launcher directory for details.
  pause
)
exit /b %EXIT_CODE%
