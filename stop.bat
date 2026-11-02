@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launchers\stop.ps1" -NoDialog
exit /b %ERRORLEVEL%
