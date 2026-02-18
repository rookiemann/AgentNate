@echo off
setlocal
cd /d "%~dp0"

REM ============================================================
REM  AgentNate Launcher
REM  Auto-installs dependencies if needed, then starts the app
REM ============================================================

REM Handle help command
for %%a in (%*) do (
    if /i "%%a"=="--help" goto :showhelp
    if /i "%%a"=="-h" goto :showhelp
    if /i "%%a"=="help" goto :showhelp
)

REM Check if all dependencies are installed
set "NEEDS_INSTALL=0"
if not exist "%~dp0python\python.exe" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Scripts\pip.exe" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\.packages-installed" set "NEEDS_INSTALL=1"
if not exist "%~dp0node\node.exe" set "NEEDS_INSTALL=1"
if not exist "%~dp0node_modules\n8n\bin\n8n" set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="1" (
    echo.
    echo  AgentNate dependencies not found. Running installer...
    echo.
    call "%~dp0install.bat" %*
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo  Installation failed. Please check the output above.
        pause
        exit /b 1
    )
    echo.
)

REM Set up PATH for portable runtimes
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0node;%PATH%"
set "PLAYWRIGHT_BROWSERS_PATH=%~dp0python\.playwright-browsers"

REM Parse mode argument (default: browser)
set "MODE=browser"
for %%a in (%*) do (
    if /i "%%a"=="--server" set "MODE=server"
    if /i "%%a"=="--desktop" set "MODE=desktop"
    if /i "%%a"=="--browser" set "MODE=browser"
)

echo.
echo  Starting AgentNate (%MODE% mode)...
echo  Python: %~dp0python\python.exe
echo  Node:   %~dp0node\node.exe
echo.

"%~dp0python\python.exe" "%~dp0run.py" --mode %MODE%
goto :eof

:showhelp
echo.
echo  AgentNate Launcher
echo  ==================
echo.
echo  Usage:  launcher.bat [options]
echo.
echo  Modes:
echo    --browser     Open in default browser (default)
echo    --server      API server only (no UI auto-open)
echo    --desktop     Desktop window (PyWebView)
echo.
echo  Install options (passed to install.bat if needed):
echo    --no-llama    Skip llama-cpp-python CUDA wheel install
echo.
echo  Other:
echo    --help, -h    Show this help
echo.
echo  The launcher automatically runs install.bat on first use.
echo  To force reinstall, delete the python\ and node\ directories.
echo.
endlocal
