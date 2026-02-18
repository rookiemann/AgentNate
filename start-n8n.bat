@echo off
cd /d "%~dp0"

REM Add portable Python to PATH
set PATH=%~dp0python;%~dp0python\Scripts;%PATH%

REM Enable native Python runner for Code node
set N8N_RUNNERS_ENABLED=true

echo Starting portable n8n (Node 24.12.0 + n8n 1.123.5 + Python 3.14.2)
echo.
echo Editor → http://localhost:5678
echo Python  → %~dp0python\python.exe
echo Close this window to stop n8n
echo.
node\node.exe node_modules\n8n\bin\n8n
pause