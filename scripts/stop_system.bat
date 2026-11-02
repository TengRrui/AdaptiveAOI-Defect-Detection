@echo off
setlocal
set "ROOT=%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\launchers\stop.ps1" -NoDialog
exit /b %ERRORLEVEL%
