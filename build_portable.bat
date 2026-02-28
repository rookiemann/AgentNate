@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM  AgentNate Portable Distribution Builder
REM
REM  Creates self-contained archives:
REM    1. AgentNate-Portable.7z  -- Full app, unzip-click-use
REM    2. AgentNate-vLLM.7z      -- Optional vLLM environment
REM
REM  What's INCLUDED in the main archive:
REM    - All source code (backend, UI, orchestrator, providers)
REM    - Portable Python 3.14.2 + all packages (~1.2 GB)
REM    - Portable Node.js 24 + n8n workflow engine (~1.1 GB)
REM
REM  What's EXCLUDED:
REM    - modules/ (ComfyUI, TTS, Music -- installable from the UI)
REM    - envs/ (vLLM -- separate archive)
REM    - .git directories, __pycache__, logs, temp files
REM    - API keys / user settings
REM ============================================================

echo.
echo  ============================================================
echo       AgentNate Portable Distribution Builder
echo  ============================================================
echo.

REM --- Configuration ---
set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
set "RELEASE_DIR=%~dp0releases"
set "MAIN_ARCHIVE=%RELEASE_DIR%\AgentNate-Portable.7z"
set "VLLM_ARCHIVE=%RELEASE_DIR%\AgentNate-vLLM.7z"
set "COMP_LEVEL=3"

REM Check 7-Zip
if not exist "%SEVENZIP%" (
    echo  ERROR: 7-Zip not found at "%SEVENZIP%"
    echo  Install 7-Zip or update the path in this script.
    exit /b 1
)

REM Create releases directory
if not exist "%RELEASE_DIR%" mkdir "%RELEASE_DIR%"

REM Parse flags
set "SKIP_MAIN=0"
set "SKIP_VLLM=0"
for %%a in (%*) do (
    if /i "%%a"=="--main-only" set "SKIP_VLLM=1"
    if /i "%%a"=="--vllm-only" set "SKIP_MAIN=1"
    if /i "%%a"=="--help" goto :showhelp
    if /i "%%a"=="-h" goto :showhelp
)

if "%SKIP_MAIN%"=="1" goto :build_vllm

REM ============================================================
REM  STEP 1: Pre-flight checks
REM ============================================================
echo  [1/3] Pre-flight checks...

set "READY=1"
if not exist "python\python.exe" (
    echo        WARNING: python\python.exe not found
    set "READY=0"
)
if not exist "node\node.exe" (
    echo        WARNING: node\node.exe not found
    set "READY=0"
)
if not exist "node_modules\n8n\bin\n8n" (
    echo        WARNING: n8n not found
    set "READY=0"
)
if "%READY%"=="1" echo        All components found
echo        OK

REM Delete old main archive if it exists
if exist "%MAIN_ARCHIVE%" (
    echo  [---] Removing old archive...
    del "%MAIN_ARCHIVE%"
)

REM ============================================================
REM  STEP 2: Archive core app
REM ============================================================
echo  [2/3] Archiving core app (source + Python + Node + n8n)...
echo        ~2.3 GB uncompressed...

REM Write an exclusion list file for 7z (avoids batch variable expansion issues)
> "%RELEASE_DIR%\_excludes.txt" (
echo .git
echo __pycache__
echo *.pyc
echo *.pyo
echo .claude
echo *.log
echo *.pid
echo Thumbs.db
echo Desktop.ini
echo .DS_Store
)

"%SEVENZIP%" a -t7z -mx=%COMP_LEVEL% -mmt=on "%MAIN_ARCHIVE%" backend core data manual orchestrator providers python node node_modules requirements scripts settings tests ui workers workflows config.py gpu_utils.py inference_worker.py install.bat launcher.bat LICENSE package.json package-lock.json README.md requirements.txt run.py start-n8n.bat update.bat .gitignore CODEBASE_OVERVIEW.md CONTRIBUTING.md -xr@"%RELEASE_DIR%\_excludes.txt"

if !ERRORLEVEL! NEQ 0 (
    del "%RELEASE_DIR%\_excludes.txt" 2>nul
    echo  ERROR: Failed to archive core app!
    exit /b 1
)
del "%RELEASE_DIR%\_excludes.txt" 2>nul
echo        OK
echo.

REM ============================================================
REM  STEP 3: Finalize
REM ============================================================
echo  [3/3] Finalizing...
echo        Modules excluded (ComfyUI, TTS, Music -- installable from UI)
echo        settings.json excluded (app auto-creates clean defaults on first run)
echo        OK
echo.

REM ============================================================
REM  Build vLLM side archive
REM ============================================================
:build_vllm
if "%SKIP_VLLM%"=="1" goto :done

echo  [vLLM] Building AgentNate-vLLM.7z ...

if not exist "envs\vllm" (
    echo         SKIP - envs\vllm directory not found
    goto :done
)

if exist "%VLLM_ARCHIVE%" del "%VLLM_ARCHIVE%"

"%SEVENZIP%" a -t7z -mx=%COMP_LEVEL% -mmt=on "%VLLM_ARCHIVE%" envs\vllm -xr@"%RELEASE_DIR%\_excludes.txt"

if !ERRORLEVEL! NEQ 0 (
    echo  WARNING: vLLM archive had issues
) else (
    echo         OK
)

:done
echo.
echo  ============================================================
echo       Build Complete!
echo  ============================================================
echo.

REM Show archive sizes
if exist "%MAIN_ARCHIVE%" (
    for %%F in ("%MAIN_ARCHIVE%") do (
        set "SIZE=%%~zF"
        set /a "SIZE_MB=!SIZE! / 1048576"
        set /a "SIZE_GB=!SIZE_MB! / 1024"
        echo  AgentNate-Portable.7z:  !SIZE_MB! MB  (~!SIZE_GB! GB)
    )
)
if exist "%VLLM_ARCHIVE%" (
    for %%F in ("%VLLM_ARCHIVE%") do (
        set "SIZE=%%~zF"
        set /a "SIZE_MB=!SIZE! / 1048576"
        echo  AgentNate-vLLM.7z:     !SIZE_MB! MB
    )
)

echo.
echo  How to use:
echo    1. Extract AgentNate-Portable.7z to any folder
echo    2. Double-click launcher.bat
echo    3. App opens in browser at http://localhost:8000
echo.
echo  First-time setup:
echo    - Settings tab: set your models directory
echo    - Add API keys for OpenRouter/Serper if desired
echo    - Install modules from their UI tabs:
echo        llama.cpp, vLLM, ComfyUI, TTS, Music
echo.
endlocal
exit /b 0

:showhelp
echo.
echo  Usage:  build_portable.bat [options]
echo.
echo  Options:
echo    --main-only   Build only the main portable archive
echo    --vllm-only   Build only the vLLM side archive
echo    --help, -h    Show this help
echo.
endlocal
exit /b 0
