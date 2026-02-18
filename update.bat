@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo                    AgentNate Updater v1.0
echo  ============================================================
echo.

REM Handle help command
for %%a in (%*) do (
    if /i "%%a"=="--help" goto :showhelp
    if /i "%%a"=="-h" goto :showhelp
)

REM Verify prerequisites
if not exist "%~dp0python\python.exe" (
    echo  ERROR: Python not found. Run install.bat first.
    exit /b 1
)
if not exist "%~dp0node\node.exe" (
    echo  ERROR: Node.js not found. Run install.bat first.
    exit /b 1
)

set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0node;%PATH%"

REM Parse flags
set "UPDATE_ALL=1"
set "UPDATE_PYTHON=0"
set "UPDATE_NODE=0"
set "UPDATE_PLAYWRIGHT=0"
set "UPDATE_CUDA=0"
for %%a in (%*) do (
    if /i "%%a"=="--python" (set "UPDATE_PYTHON=1" & set "UPDATE_ALL=0")
    if /i "%%a"=="--node" (set "UPDATE_NODE=1" & set "UPDATE_ALL=0")
    if /i "%%a"=="--playwright" (set "UPDATE_PLAYWRIGHT=1" & set "UPDATE_ALL=0")
    if /i "%%a"=="--cuda" set "UPDATE_CUDA=1"
)

REM If --cuda is the only flag, also set UPDATE_ALL=0
if "%UPDATE_CUDA%"=="1" if "%UPDATE_PYTHON%"=="0" if "%UPDATE_NODE%"=="0" if "%UPDATE_PLAYWRIGHT%"=="0" set "UPDATE_ALL=0"

REM ============================================================
REM  Update Python packages
REM ============================================================
if "%UPDATE_ALL%"=="1" set "UPDATE_PYTHON=1"
if "%UPDATE_PYTHON%"=="1" (
    echo  [Python] Updating packages from requirements.txt...
    REM Filter out llama-cpp-python (no pre-built wheel on PyPI for Python 3.14)
    type "%~dp0requirements.txt" | findstr /V /I "llama-cpp-python" > "%~dp0python\_req_filtered.txt"
    "%~dp0python\python.exe" -m pip install -r "%~dp0python\_req_filtered.txt" --upgrade --no-warn-script-location
    del "%~dp0python\_req_filtered.txt" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo           FAILED
        exit /b 1
    )
    echo %DATE% %TIME% > "%~dp0python\.packages-installed"
    echo           OK
    echo.
)

REM ============================================================
REM  Update n8n
REM ============================================================
if "%UPDATE_ALL%"=="1" set "UPDATE_NODE=1"
if "%UPDATE_NODE%"=="1" (
    echo  [n8n] Updating via npm...
    "%~dp0node\npm.cmd" update --prefix "%~dp0."
    if !ERRORLEVEL! NEQ 0 (
        echo        FAILED
        exit /b 1
    )
    echo        OK
    echo.
)

REM ============================================================
REM  Update Playwright
REM ============================================================
if "%UPDATE_ALL%"=="1" set "UPDATE_PLAYWRIGHT=1"
if "%UPDATE_PLAYWRIGHT%"=="1" (
    echo  [Playwright] Updating Chromium browser...
    set "PLAYWRIGHT_BROWSERS_PATH=%~dp0python\.playwright-browsers"
    "%~dp0python\python.exe" -m playwright install chromium
    if !ERRORLEVEL! NEQ 0 (
        echo               FAILED ^(non-critical^)
    ) else (
        echo %DATE% %TIME% > "%~dp0python\.playwright-installed"
        echo               OK
    )
    echo.
)

REM ============================================================
REM  CUDA llama-cpp-python wheel (optional)
REM ============================================================
if "%UPDATE_CUDA%"=="1" (
    echo  [CUDA] Reinstalling llama-cpp-python GPU wheel...
    "%~dp0python\python.exe" -m pip uninstall llama-cpp-python -y >nul 2>&1
    "%~dp0python\python.exe" -m pip install "https://github.com/rookiemann/llama-cpp-python-py314-cuda131-wheel/releases/download/v0.3.16-cuda13.1-py3.14/llama_cpp_python-0.3.16-cp314-cp314-win_amd64.whl" --force-reinstall --no-deps --no-warn-script-location
    if !ERRORLEVEL! NEQ 0 (
        echo         FAILED
        exit /b 1
    )
    echo %DATE% %TIME% > "%~dp0python\.llama-cuda-installed"
    echo         OK
    echo.
)

echo.
echo  Update complete!
echo.
endlocal
goto :eof

:showhelp
echo.
echo  AgentNate Updater
echo  =================
echo.
echo  Usage:  update.bat [options]
echo.
echo  With no options, updates everything (Python packages, n8n, Playwright).
echo.
echo  Options:
echo    --python      Update Python packages only
echo    --node        Update n8n only
echo    --playwright  Update Playwright Chromium only
echo    --cuda        Reinstall CUDA llama-cpp-python wheel
echo    --help, -h    Show this help
echo.
echo  Examples:
echo    update.bat                  Update all components
echo    update.bat --python         Update Python packages only
echo    update.bat --python --cuda  Update packages + reinstall CUDA wheel
echo    update.bat --node           Update n8n to latest compatible version
echo.
endlocal
