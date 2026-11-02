@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launchers\start.ps1" %*
exit /b %ERRORLEVEL%
