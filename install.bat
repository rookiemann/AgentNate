@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo                   AgentNate Installer v1.0
echo           Portable Python 3.14.2 + Node.js 24.12.0
echo  ============================================================
echo.

REM ============================================================
REM  Version Configuration
REM ============================================================
set "PYTHON_VERSION=3.14.2"
set "PYTHON_URL=https://www.python.org/ftp/python/3.14.2/python-3.14.2-embed-amd64.zip"
set "PYTHON_PTH_ZIP=python314.zip"
set "NODE_VERSION=24.12.0"
set "NODE_URL=https://nodejs.org/dist/v24.12.0/node-v24.12.0-win-x64.zip"
set "NODE_EXTRACT_DIR=node-v24.12.0-win-x64"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "LLAMA_WHEEL=https://github.com/rookiemann/llama-cpp-python-py314-cuda131-wheel/releases/download/v0.3.16-cuda13.1-py3.14/llama_cpp_python-0.3.16-cp314-cp314-win_amd64.whl"

REM Parse flags
set "SKIP_LLAMA=0"
for %%a in (%*) do (
    if /i "%%a"=="--no-llama" set "SKIP_LLAMA=1"
)

set "STAGES_TOTAL=7"
if "%SKIP_LLAMA%"=="1" set "STAGES_TOTAL=6"

echo  Components to install:
echo    - Python %PYTHON_VERSION% (embedded distribution)
echo    - pip (package manager)
echo    - Python packages (requirements.txt)
echo    - Playwright Chromium (browser automation)
echo    - Node.js %NODE_VERSION% (portable binary)
echo    - n8n (workflow engine)
if "%SKIP_LLAMA%"=="0" echo    - llama-cpp-python CUDA wheel (GPU acceleration)
echo.

REM ============================================================
REM  STAGE 1: Download and Extract Python Embedded
REM ============================================================
echo [1/%STAGES_TOTAL%] Python %PYTHON_VERSION% embedded...
if exist "%~dp0python\python.exe" (
    echo          SKIP - already installed
    goto :stage2
)
echo          Downloading from python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP%\agentnate-python-embed.zip'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download Python %PYTHON_VERSION%
    echo          URL: %PYTHON_URL%
    exit /b 1
)
echo          Extracting to python\ ...
if not exist "%~dp0python" mkdir "%~dp0python"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%TEMP%\agentnate-python-embed.zip' -DestinationPath '%~dp0python' -Force"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not extract Python archive
    exit /b 1
)
del "%TEMP%\agentnate-python-embed.zip" 2>nul
if not exist "%~dp0python\python.exe" (
    echo          FAILED: python.exe not found after extraction
    exit /b 1
)
echo          OK

:stage2
REM ============================================================
REM  STAGE 2: Configure Python for site-packages + Install pip
REM ============================================================
echo [2/%STAGES_TOTAL%] Python configuration + pip...
if exist "%~dp0python\Scripts\pip.exe" (
    echo          SKIP - pip already installed
    goto :stage3
)

REM Write python314._pth to enable site-packages and import site
echo          Configuring python314._pth...
(
    echo %PYTHON_PTH_ZIP%
    echo .
    echo Lib
    echo Lib\site-packages
    echo DLLs
    echo import site
) > "%~dp0python\python314._pth"

REM Create required directories
if not exist "%~dp0python\Lib\site-packages" mkdir "%~dp0python\Lib\site-packages"
if not exist "%~dp0python\Scripts" mkdir "%~dp0python\Scripts"

REM Download and run get-pip.py
echo          Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%TEMP%\get-pip.py'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download get-pip.py
    exit /b 1
)
echo          Installing pip...
"%~dp0python\python.exe" "%TEMP%\get-pip.py" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: pip installation error
    exit /b 1
)
del "%TEMP%\get-pip.py" 2>nul
if not exist "%~dp0python\Scripts\pip.exe" (
    echo          FAILED: pip.exe not found after installation
    exit /b 1
)
echo          OK

:stage3
REM ============================================================
REM  STAGE 3: Install Python Packages
REM ============================================================
echo [3/%STAGES_TOTAL%] Python packages (requirements.txt)...
if exist "%~dp0python\.packages-installed" (
    echo          SKIP - already installed ^(delete python\.packages-installed to force^)
    goto :stage4
)
echo          Installing packages (this may take several minutes)...
REM Filter out llama-cpp-python — no pre-built wheel for Python 3.14 on PyPI.
REM It is installed separately from a pre-built CUDA wheel in stage 7.
type "%~dp0requirements.txt" | findstr /V /I "llama-cpp-python" > "%~dp0python\_req_filtered.txt"
"%~dp0python\python.exe" -m pip install -r "%~dp0python\_req_filtered.txt" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    del "%~dp0python\_req_filtered.txt" 2>nul
    echo          FAILED: pip install error - check output above
    exit /b 1
)
del "%~dp0python\_req_filtered.txt" 2>nul
echo %DATE% %TIME% > "%~dp0python\.packages-installed"
echo          OK

:stage4
REM ============================================================
REM  STAGE 4: Install Playwright Chromium
REM ============================================================
echo [4/%STAGES_TOTAL%] Playwright Chromium browser...
if exist "%~dp0python\.playwright-installed" (
    echo          SKIP - already installed ^(delete python\.playwright-installed to force^)
    goto :stage5
)
echo          Installing Chromium browser (~150MB download)...
set "PLAYWRIGHT_BROWSERS_PATH=%~dp0python\.playwright-browsers"
"%~dp0python\python.exe" -m playwright install chromium
if !ERRORLEVEL! NEQ 0 (
    echo          WARNING: Playwright install failed (non-critical, browser tools won't work)
    echo          You can retry later: python\python.exe -m playwright install chromium
    goto :stage5
)
echo %DATE% %TIME% > "%~dp0python\.playwright-installed"
echo          OK

:stage5
REM ============================================================
REM  STAGE 5: Download and Extract Node.js
REM ============================================================
echo [5/%STAGES_TOTAL%] Node.js %NODE_VERSION%...
if exist "%~dp0node\node.exe" (
    echo          SKIP - already installed
    goto :stage6
)
echo          Downloading from nodejs.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%NODE_URL%' -OutFile '%TEMP%\agentnate-node-portable.zip'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download Node.js %NODE_VERSION%
    echo          URL: %NODE_URL%
    exit /b 1
)
echo          Extracting...
if exist "%TEMP%\agentnate-node-extract" rmdir /s /q "%TEMP%\agentnate-node-extract"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%TEMP%\agentnate-node-portable.zip' -DestinationPath '%TEMP%\agentnate-node-extract' -Force"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not extract Node.js archive
    exit /b 1
)
echo          Moving to node\ ...
if not exist "%~dp0node" mkdir "%~dp0node"
REM Node.js ZIP extracts to a subdirectory, move contents up
robocopy "%TEMP%\agentnate-node-extract\%NODE_EXTRACT_DIR%" "%~dp0node" /E /MOVE >nul 2>&1
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 (
    echo          FAILED: Could not move Node.js files (robocopy error !RC!)
    exit /b 1
)
rmdir /s /q "%TEMP%\agentnate-node-extract" 2>nul
del "%TEMP%\agentnate-node-portable.zip" 2>nul
if not exist "%~dp0node\node.exe" (
    echo          FAILED: node.exe not found after extraction
    exit /b 1
)
echo          OK

:stage6
REM ============================================================
REM  STAGE 6: Install n8n via npm
REM ============================================================
echo [6/%STAGES_TOTAL%] n8n workflow engine...
if exist "%~dp0node_modules\n8n\bin\n8n" (
    echo          SKIP - already installed
    goto :stage7
)
echo          Installing via npm (this may take several minutes)...
set "PATH=%~dp0node;%PATH%"
"%~dp0node\npm.cmd" install --prefix "%~dp0."
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: npm install error - check output above
    exit /b 1
)
if not exist "%~dp0node_modules\n8n\bin\n8n" (
    echo          FAILED: n8n not found after npm install
    exit /b 1
)
echo          OK

:stage7
REM ============================================================
REM  STAGE 7: llama-cpp-python CUDA wheel (pre-built for Py3.14)
REM ============================================================
if "%SKIP_LLAMA%"=="1" goto :done
echo [7/%STAGES_TOTAL%] llama-cpp-python CUDA wheel...
if exist "%~dp0python\.llama-cuda-installed" (
    echo          SKIP - already installed ^(delete python\.llama-cuda-installed to force^)
    goto :done
)
echo          Installing pre-built CUDA wheel from GitHub...
"%~dp0python\python.exe" -m pip install "%LLAMA_WHEEL%" --no-deps --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          WARNING: llama-cpp-python wheel install failed ^(non-critical^)
    echo          Local LLM inference via llama.cpp won't be available.
    echo          You can retry: python\python.exe -m pip install "%LLAMA_WHEEL%"
    goto :done
)
echo %DATE% %TIME% > "%~dp0python\.llama-cuda-installed"
echo          OK

:done
echo.
echo  ============================================================
echo                   Installation Complete!
echo  ============================================================
echo.
echo  To start AgentNate:
echo    launcher.bat                  (auto-detect mode)
echo    launcher.bat --server         (API server only)
echo    launcher.bat --browser        (open in browser)
echo    launcher.bat --desktop        (desktop window)
echo.
echo  Or manually:
echo    python\python.exe run.py --mode browser
echo.
endlocal
