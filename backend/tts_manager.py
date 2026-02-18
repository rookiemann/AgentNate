"""
TTS Module Manager

Manages the portable TTS server lifecycle:
- Download/clone the portable TTS server from GitHub
- Bootstrap (Python, Git, FFmpeg, venvs, models)
- Start/stop the TTS API gateway server
- Proxy requests to the TTS API
- Clean shutdown of all processes
"""

import os
import re
import asyncio
import logging
import subprocess
import signal
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("TTSManager")

REPO_URL = "https://github.com/rookiemann/portable-tts-server"
DEFAULT_API_PORT = 8100
API_STARTUP_TIMEOUT = 120  # TTS models can be slow to load
PROXY_TIMEOUT = 30

# Model setup — mirrors MODEL_SETUP from the TTS manager GUI
# Each entry: env = venv name, weights_repo = HF repo (None = auto-download on first use)
MODEL_SETUP = {
    "bark":       {"display": "Bark",            "desc": "Expressive TTS - laughter, music, emotions",   "env": "coqui_env",      "weights_repo": "suno/bark",                "weights_dir": "bark",        "weights_size": "~5GB"},
    "chatterbox": {"display": "Chatterbox",      "desc": "Emotion control, voice cloning",               "env": "chatterbox_env", "weights_repo": "ResembleAI/chatterbox",    "weights_dir": "chatterbox",  "weights_size": "~2GB"},
    "dia":        {"display": "Dia 1.6B",        "desc": "Dialogue TTS with [S1]/[S2] speaker tags",     "env": "unified_env",    "weights_repo": "nari-labs/Dia-1.6B-0626",  "weights_dir": "dia",         "weights_size": "~6GB"},
    "f5":         {"display": "F5-TTS",          "desc": "Diffusion TTS, reference audio cloning",       "env": "f5tts_env",      "weights_repo": "SWivid/F5-TTS",            "weights_dir": "f5-tts",      "weights_size": "~1.5GB"},
    "fish":       {"display": "Fish Speech 1.5", "desc": "Fast TTS with voice cloning",                  "env": "unified_env",    "weights_repo": "fishaudio/fish-speech-1.5","weights_dir": "fish-speech", "weights_size": "~1GB"},
    "higgs":      {"display": "Higgs Audio 3B",  "desc": "Boson AI ChatML (CPU supported)",              "env": "higgs_env",      "weights_repo": None,                       "weights_dir": None,          "weights_size": None},
    "kokoro":     {"display": "Kokoro 82M",      "desc": "Lightweight, fast, 54 built-in voices",        "env": "unified_env",    "weights_repo": "hexgrad/Kokoro-82M",       "weights_dir": "kokoro",      "weights_size": "~300MB"},
    "qwen":       {"display": "Qwen Omni 7B",    "desc": "Multimodal with speech output",                "env": "qwen3_env",      "weights_repo": None,                       "weights_dir": None,          "weights_size": None},
    "vibevoice":  {"display": "VibeVoice",       "desc": "Speaker-conditioned TTS",                      "env": "vibevoice_env",  "weights_repo": None,                       "weights_dir": None,          "weights_size": None},
    "xtts":       {"display": "XTTS v2",         "desc": "Multilingual voice cloning, 58 built-in voices","env": "coqui_env",     "weights_repo": "coqui/XTTS-v2",            "weights_dir": "xtts-v2",     "weights_size": "~1.8GB"},
}

# Env -> display name mapping
ENV_DISPLAY = {
    "coqui_env": "Coqui (XTTS + Bark)",
    "unified_env": "Unified (Kokoro + Fish + Dia)",
    "chatterbox_env": "Chatterbox",
    "f5tts_env": "F5-TTS",
    "qwen3_env": "Qwen Omni",
    "vibevoice_env": "VibeVoice",
    "higgs_env": "Higgs Audio",
}

# Debug log file for TTS subprocess output
_debug_log_path: Optional[Path] = None


def _get_debug_log() -> Optional[Path]:
    """Get the debug log path."""
    global _debug_log_path
    if _debug_log_path is None:
        base = Path(__file__).parent.parent
        instances_dir = base / ".n8n-instances"
        if instances_dir.is_dir():
            _debug_log_path = instances_dir / "tts-debug.log"
        else:
            _debug_log_path = base / "tts-debug.log"
    return _debug_log_path


def _log_to_file(message: str):
    """Append a timestamped message to the TTS debug log."""
    import datetime
    try:
        path = _get_debug_log()
        if path:
            with open(path, "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def _log(msg: str, level: str = "info"):
    """Log to both Python logger and debug file."""
    getattr(logger, level, logger.info)(msg)
    prefix = {"error": "ERROR", "warning": "WARN", "debug": "DEBUG"}.get(level, "INFO")
    _log_to_file(f"[{prefix}] {msg}")


def _pipe_reader(pipe, label: str):
    """Read a subprocess pipe line-by-line and log each line. Runs in a thread."""
    try:
        for raw_line in iter(pipe.readline, b''):
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                _log_to_file(f"[{label}] {line}")
    except Exception as e:
        _log_to_file(f"[{label}] Pipe reader error: {e}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


class TTSManager:
    """Manages the portable TTS server lifecycle."""

    def __init__(self, modules_dir: Path, process_registry=None):
        self.modules_dir = modules_dir
        self.module_dir = modules_dir / "tts"
        self.api_port = DEFAULT_API_PORT
        self.api_process: Optional[subprocess.Popen] = None
        self.process_registry = process_registry
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # Shared HTTP client for proxy calls
        self._http_client: Optional[httpx.AsyncClient] = None
        # TTL cache for get_status()
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_time: float = 0.0
        self._STATUS_CACHE_TTL: float = 5.0
        self._status_lock: Optional[asyncio.Lock] = None
        # TTL cache for is_api_running
        self._api_running_cache: Optional[bool] = None
        self._api_running_cache_time: float = 0.0
        self._API_RUNNING_CACHE_TTL: float = 3.0

        # Truncate the debug log on init
        try:
            log_path = _get_debug_log()
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("")
                _log(f"TTS debug log initialized: {log_path}")
        except Exception:
            pass

        _log(f"TTS manager init: modules_dir={modules_dir}, module_dir={self.module_dir}")

    # ======================== Status Checks ========================

    def is_module_downloaded(self) -> bool:
        """Check if the TTS server repo has been cloned."""
        result = (self.module_dir / "tts_api_server.py").is_file()
        _log(f"is_module_downloaded: {result}", "debug")
        return result

    def is_bootstrapped(self) -> bool:
        """Check if Python embedded has been downloaded."""
        path = self.module_dir / "python_embedded" / "python.exe"
        result = path.is_file()
        _log(f"is_bootstrapped: {result}", "debug")
        return result

    def is_installed(self) -> bool:
        """Check if TTS gateway requirements are installed (bootstrap completed)."""
        # Check for key gateway packages in python_embedded site-packages
        sp = self.module_dir / "python_embedded" / "Lib" / "site-packages"
        result = (sp / "fastapi").is_dir() and (sp / "uvicorn").is_dir()
        _log(f"is_installed: {result}", "debug")
        return result

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=PROXY_TIMEOUT,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    async def _close_client(self):
        """Close the shared HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def is_api_running(self) -> bool:
        """Health check on the TTS API gateway (cached for 3s)."""
        now = time.time()
        if self._api_running_cache is not None and (now - self._api_running_cache_time) < self._API_RUNNING_CACHE_TTL:
            return self._api_running_cache

        # Fast path: if we know the process isn't running, skip HTTP
        if self.api_process is not None and self.api_process.poll() is not None:
            self._api_running_cache = False
            self._api_running_cache_time = time.time()
            return False

        url = f"http://127.0.0.1:{self.api_port}/health"
        try:
            client = await self._get_client()
            r = await client.get(url, timeout=3)
            ok = r.status_code == 200
            self._api_running_cache = ok
            self._api_running_cache_time = time.time()
            return ok
        except Exception:
            self._api_running_cache = False
            self._api_running_cache_time = time.time()
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get combined status of the TTS module (cached with 5s TTL, stampede-safe)."""
        now = time.time()
        if self._status_cache and (now - self._status_cache_time) < self._STATUS_CACHE_TTL:
            return self._status_cache

        # Lazy-init lock
        if self._status_lock is None:
            self._status_lock = asyncio.Lock()

        async with self._status_lock:
            now = time.time()
            if self._status_cache and (now - self._status_cache_time) < self._STATUS_CACHE_TTL:
                return self._status_cache

            status = {
                "module_downloaded": self.is_module_downloaded(),
                "bootstrapped": self.is_bootstrapped(),
                "installed": self.is_installed(),
                "api_running": False,
                "api_port": self.api_port,
                "workers": [],
                "models": [],
                "devices": [],
            }

            if await self.is_api_running():
                status["api_running"] = True
                try:
                    workers = await self.proxy("GET", "/api/workers")
                    status["workers"] = workers.get("workers", [])
                except Exception as e:
                    _log(f"Failed to fetch workers: {e}", "warning")

                try:
                    models = await self.proxy("GET", "/api/models")
                    status["models"] = models.get("models", [])
                except Exception as e:
                    _log(f"Failed to fetch models: {e}", "warning")

                try:
                    devices = await self.proxy("GET", "/api/devices")
                    status["devices"] = devices.get("devices", [])
                except Exception as e:
                    _log(f"Failed to fetch devices: {e}", "warning")

            self._status_cache = status
            self._status_cache_time = time.time()
            return status

    # ======================== Module Lifecycle ========================

    async def download_module(self) -> Dict[str, Any]:
        """Clone the portable TTS server from GitHub."""
        if self.is_module_downloaded():
            _log("Module already downloaded, skipping clone")
            return {"success": True, "message": "Module already downloaded"}

        self.modules_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Cloning TTS server: {REPO_URL} -> {self.module_dir}")

        try:
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", REPO_URL, str(self.module_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

            stdout_text = stdout.decode().strip()
            stderr_text = stderr.decode().strip()

            if stdout_text:
                _log(f"[GIT STDOUT] {stdout_text}")
            if stderr_text:
                _log(f"[GIT STDERR] {stderr_text}")

            if process.returncode == 0:
                _log("TTS module downloaded successfully")
                return {"success": True, "message": "Module downloaded successfully"}
            else:
                _log(f"Git clone failed (rc={process.returncode}): {stderr_text}", "error")
                return {"success": False, "error": stderr_text}
        except asyncio.TimeoutError:
            _log("Git clone timed out after 120 seconds", "error")
            return {"success": False, "error": "Git clone timed out after 120 seconds"}
        except Exception as e:
            _log(f"Git clone exception: {type(e).__name__}: {e}", "error")
            return {"success": False, "error": str(e)}

    async def update_module(self) -> Dict[str, Any]:
        """Update the TTS server via git pull."""
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded yet"}

        _log("Updating TTS module via git pull...")

        try:
            process = await asyncio.create_subprocess_exec(
                "git", "-C", str(self.module_dir), "pull",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

            stdout_text = stdout.decode().strip()
            stderr_text = stderr.decode().strip()
            _log(f"[GIT PULL] stdout={stdout_text}, stderr={stderr_text}")

            if process.returncode == 0:
                _log(f"Module updated: {stdout_text}")
                return {"success": True, "message": stdout_text}
            else:
                return {"success": False, "error": stderr_text}
        except Exception as e:
            _log(f"Git pull exception: {e}", "error")
            return {"success": False, "error": str(e)}

    def _generate_bootstrap_script(self, sd: str) -> str:
        """
        Generate a clean sequential bootstrap .bat script.
        Each step is a simple CALL :label, avoiding nested if() blocks
        that break under headless cmd /c execution.
        """
        # sd = script dir with trailing backslash, e.g. E:\AgentNate\modules\tts\
        return f'''@echo off
rem ============================================
rem  TTS Module - Headless Bootstrap
rem  Generated by AgentNate TTSManager
rem ============================================

set "SD={sd}"
set "PYTHON_DIR=%SD%python_embedded"
set "PYTHON_EXE=%PYTHON_DIR%\\python.exe"
set "GIT_DIR=%SD%git_portable"
set "GIT_EXE=%GIT_DIR%\\cmd\\git.exe"
set "FFMPEG_DIR=%SD%ffmpeg"
set "RUBBERBAND_DIR=%SD%rubberband"
set "ESPEAK_DIR=%SD%espeak_ng"

rem ---- Step 1: Download embedded Python ----
if exist "%PYTHON_EXE%" (
    echo [OK] Python already installed.
    goto :step_pip
)
echo [1/8] Downloading Python 3.10.11 embedded...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile '%SD%python_embedded.zip'"
if not exist "%SD%python_embedded.zip" echo ERROR: Python download failed. && exit /b 1

echo [1/8] Extracting Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%python_embedded.zip' -DestinationPath '%PYTHON_DIR%' -Force"
del "%SD%python_embedded.zip" 2>nul
if not exist "%PYTHON_EXE%" echo ERROR: Python extraction failed. && exit /b 1

echo [1/8] Configuring Python pth file...
if not exist "%PYTHON_DIR%\\Lib\\site-packages" mkdir "%PYTHON_DIR%\\Lib\\site-packages"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$pth = Get-ChildItem '%PYTHON_DIR%\\python*._pth' | Select-Object -First 1; if ($pth) {{ $zip = (Get-ChildItem '%PYTHON_DIR%\\python*.zip' | Select-Object -First 1).Name; if (-not $zip) {{ $zip = 'python310.zip' }}; @($zip, '.', 'Lib', 'Lib\\site-packages', '', 'import site') | Set-Content -Path $pth.FullName -Encoding ASCII; Write-Host 'Configured:' $pth.Name }}"

rem ---- Step 2: Install pip ----
:step_pip
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] pip already available.
    goto :step_git
)
echo [2/8] Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\\get-pip.py'"
if not exist "%PYTHON_DIR%\\get-pip.py" echo ERROR: get-pip download failed. && exit /b 1
echo [2/8] Installing pip...
"%PYTHON_EXE%" "%PYTHON_DIR%\\get-pip.py"
del "%PYTHON_DIR%\\get-pip.py" 2>nul
"%PYTHON_EXE%" -m pip install --upgrade pip 2>nul

rem ---- Step 3: Download portable Git ----
:step_git
if exist "%GIT_EXE%" (
    echo [OK] Portable Git already installed.
    goto :step_ffmpeg
)
echo [3/8] Downloading portable Git 2.47.1...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip' -OutFile '%SD%git_portable.zip'"
if not exist "%SD%git_portable.zip" (
    echo WARNING: Git download failed. Some features may not work.
    goto :step_ffmpeg
)
echo [3/8] Extracting Git...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%git_portable.zip' -DestinationPath '%GIT_DIR%' -Force"
del "%SD%git_portable.zip" 2>nul

rem ---- Step 4: Download portable FFmpeg ----
:step_ffmpeg
if exist "%FFMPEG_DIR%\\bin\\ffmpeg.exe" (
    echo [OK] FFmpeg already installed.
    goto :step_rubberband
)
echo [4/8] Downloading portable FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%SD%ffmpeg_portable.zip'"
if not exist "%SD%ffmpeg_portable.zip" (
    echo WARNING: FFmpeg download failed.
    goto :step_rubberband
)
echo [4/8] Extracting FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%ffmpeg_portable.zip' -DestinationPath '%FFMPEG_DIR%' -Force"
del "%SD%ffmpeg_portable.zip" 2>nul

rem ---- Step 5: Download Rubberband ----
:step_rubberband
if exist "%RUBBERBAND_DIR%\\rubberband.exe" (
    echo [OK] Rubberband already installed.
    goto :step_espeak
)
echo [5/8] Downloading Rubberband 4.0.0...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip' -OutFile '%SD%rubberband_portable.zip'"
if not exist "%SD%rubberband_portable.zip" (
    echo WARNING: Rubberband download failed.
    goto :step_espeak
)
echo [5/8] Extracting Rubberband...
if not exist "%RUBBERBAND_DIR%" mkdir "%RUBBERBAND_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%rubberband_portable.zip' -DestinationPath '%RUBBERBAND_DIR%' -Force"
del "%SD%rubberband_portable.zip" 2>nul

rem ---- Step 6: Download eSpeak NG ----
:step_espeak
if exist "%ESPEAK_DIR%\\espeak-ng.exe" goto :step_path
rem Check inside subdirectory too
for /d %%D in ("%ESPEAK_DIR%\\eSpeak*") do if exist "%%D\\espeak-ng.exe" goto :step_path
echo [6/8] Downloading eSpeak NG 1.52...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi' -OutFile '%SD%espeak_ng.msi'"
if not exist "%SD%espeak_ng.msi" (
    echo WARNING: eSpeak NG download failed. Kokoro TTS will not work.
    goto :step_path
)
echo [6/8] Extracting eSpeak NG...
if not exist "%ESPEAK_DIR%" mkdir "%ESPEAK_DIR%"
start /wait msiexec /a "%SD%espeak_ng.msi" /qn TARGETDIR="%ESPEAK_DIR%"
del "%SD%espeak_ng.msi" 2>nul

rem ---- Step 7: Set up PATH and install requirements ----
:step_path
if exist "%GIT_EXE%" set "PATH=%GIT_DIR%\\cmd;%PATH%"
if exist "%FFMPEG_DIR%\\bin\\ffmpeg.exe" set "PATH=%FFMPEG_DIR%\\bin;%PATH%"
for /d %%D in ("%FFMPEG_DIR%\\ffmpeg-*") do if exist "%%D\\bin\\ffmpeg.exe" set "PATH=%%D\\bin;%PATH%"
for /d %%D in ("%RUBBERBAND_DIR%\\rubberband-*") do if exist "%%D\\rubberband.exe" set "PATH=%%D;%PATH%"
if exist "%RUBBERBAND_DIR%\\rubberband.exe" set "PATH=%RUBBERBAND_DIR%;%PATH%"
if exist "%ESPEAK_DIR%\\espeak-ng.exe" set "PATH=%ESPEAK_DIR%;%PATH%"
for /d %%D in ("%ESPEAK_DIR%\\eSpeak*") do if exist "%%D\\espeak-ng.exe" set "PATH=%%D;%PATH%"

echo [7/8] Installing requirements...
"%PYTHON_EXE%" -m pip install -r "%SD%requirements.txt" --quiet 2>nul
if errorlevel 1 "%PYTHON_EXE%" -m pip install -r "%SD%requirements.txt"

rem ---- Step 8: Install Whisper ----
echo [8/8] Installing Whisper...
"%PYTHON_EXE%" -m pip install openai-whisper --no-deps --quiet 2>nul

echo.
echo [DONE] Bootstrap complete.
'''

    async def bootstrap(self) -> Dict[str, Any]:
        """
        Run the bootstrap process: download Python, Git, FFmpeg, create venvs, install deps.
        Generates a clean sequential bootstrap script (avoids complex nested if-blocks
        in install.bat that break under headless cmd /c execution).
        """
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded. Download it first."}

        if self.is_installed():
            _log("Already fully installed, skipping bootstrap")
            return {"success": True, "message": "Already installed"}

        _log("Running TTS bootstrap (headless)...")

        # Generate a purpose-built headless script instead of truncating install.bat.
        # The original install.bat uses enabledelayedexpansion + nested if() blocks
        # with !VAR! syntax that causes "... was unexpected at this time" errors
        # when run via cmd /c with CREATE_NO_WINDOW + stdin=DEVNULL.
        bootstrap_script = self.module_dir / "_bootstrap_headless.bat"
        try:
            sd = str(self.module_dir).rstrip("\\") + "\\"
            content = self._generate_bootstrap_script(sd)
            with open(bootstrap_script, "w", encoding="utf-8") as f:
                f.write(content)
            _log(f"Created bootstrap script: {bootstrap_script} ({len(content)} chars)")
        except Exception as e:
            _log(f"Failed to create bootstrap script: {e}", "error")
            return {"success": False, "error": f"Failed to create bootstrap script: {e}"}

        try:
            _log(f"Executing: cmd /c {bootstrap_script}")
            process = await asyncio.create_subprocess_exec(
                "cmd", "/c", str(bootstrap_script),
                cwd=str(self.module_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            async def _read_stream(stream, label):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        _log_to_file(f"[BOOTSTRAP {label}] {text}")

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        _read_stream(process.stdout, "OUT"),
                        _read_stream(process.stderr, "ERR"),
                        process.wait(),
                    ),
                    timeout=1800,  # 30 minutes — TTS installs many venvs + models
                )
            except asyncio.TimeoutError:
                _log("Bootstrap timed out after 30 minutes", "error")
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    bootstrap_script.unlink(missing_ok=True)
                except Exception:
                    pass
                return {"success": False, "error": "Bootstrap timed out after 30 minutes"}

            # Clean up temp script
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass

            _log(f"Bootstrap process exited with code {process.returncode}")

            if process.returncode == 0 or self.is_installed():
                _log("Bootstrap completed successfully")
                return {"success": True, "message": "Bootstrap completed"}
            else:
                _log(f"Bootstrap failed (rc={process.returncode})", "error")
                return {"success": False, "error": f"Bootstrap exited with code {process.returncode}. Check tts-debug.log for details."}
        except asyncio.TimeoutError:
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": "Bootstrap timed out"}
        except Exception as e:
            _log(f"Bootstrap exception: {type(e).__name__}: {e}", "error")
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": str(e)}

    # ======================== Model Management ========================

    def get_model_info(self) -> list:
        """Get info about all TTS models: env installed, weights downloaded."""
        models = []
        venvs_dir = self.module_dir / "venvs"
        models_dir = self.module_dir / "tts_models"

        for model_id, info in MODEL_SETUP.items():
            env_name = info["env"]
            # Check if the venv exists and has the python executable
            venv_python = venvs_dir / env_name / "Scripts" / "python.exe"
            env_installed = venv_python.is_file()

            # Check if weights are downloaded
            weights_downloaded = False
            if info["weights_dir"]:
                weights_path = models_dir / info["weights_dir"]
                try:
                    weights_downloaded = weights_path.is_dir() and any(weights_path.iterdir())
                except (OSError, StopIteration):
                    weights_downloaded = False
            else:
                # Models with no weights_repo auto-download on first use
                weights_downloaded = None  # N/A

            # Which other models share this env?
            shared_with = [m for m, i in MODEL_SETUP.items() if i["env"] == env_name and m != model_id]

            models.append({
                "id": model_id,
                "display": info["display"],
                "desc": info["desc"],
                "env_name": env_name,
                "env_display": ENV_DISPLAY.get(env_name, env_name),
                "env_installed": env_installed,
                "weights_repo": info["weights_repo"],
                "weights_dir": info["weights_dir"],
                "weights_size": info["weights_size"],
                "weights_downloaded": weights_downloaded,
                "shared_env_with": shared_with,
            })

        return models

    async def install_model_env(self, env_name: str) -> Dict[str, Any]:
        """Install a model's virtual environment (create venv + pip install packages)."""
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded"}
        if not self.is_bootstrapped():
            return {"success": False, "error": "Module not bootstrapped. Run bootstrap first."}

        if env_name not in ENV_DISPLAY:
            return {"success": False, "error": f"Unknown environment: {env_name}"}

        venvs_dir = self.module_dir / "venvs"
        venv_path = venvs_dir / env_name
        python_embedded = self.module_dir / "python_embedded" / "python.exe"

        if not python_embedded.is_file():
            return {"success": False, "error": "Embedded Python not found"}

        # Check if already installed
        venv_python = venv_path / "Scripts" / "python.exe"
        if venv_python.is_file():
            return {"success": True, "message": f"Environment {env_name} already installed"}

        _log(f"Installing environment: {env_name}")

        # Load the install config from the TTS module
        try:
            import importlib.util
            init_path = self.module_dir / "install_configs" / "__init__.py"
            if not init_path.is_file():
                return {"success": False, "error": "install_configs not found in TTS module"}

            # We need to load the config module from the TTS directory
            # Use subprocess to avoid import path conflicts
            config_script = f'''
import sys, os, json
sys.path.insert(0, r"{self.module_dir}")
from install_configs import ALL_CONFIGS
cfg = ALL_CONFIGS.get("{env_name}")
if not cfg:
    print(json.dumps({{"error": "Config not found"}}))
    sys.exit(1)
steps = cfg.get_install_steps()
print(json.dumps({{"steps": steps, "name": cfg.display_name}}))
'''
            result = subprocess.run(
                [str(python_embedded), "-c", config_script],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.module_dir),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            if result.returncode != 0:
                _log(f"Failed to load install config: {result.stderr}", "error")
                return {"success": False, "error": f"Failed to load config: {result.stderr[:300]}"}

            config_data = __import__("json").loads(result.stdout.strip())
            if "error" in config_data:
                return {"success": False, "error": config_data["error"]}

            steps = config_data["steps"]
            display_name = config_data.get("name", env_name)

        except Exception as e:
            _log(f"Error loading install config: {e}", "error")
            return {"success": False, "error": str(e)}

        # Step 1: Create the venv using virtualenv (embedded Python lacks stdlib venv)
        _log(f"Creating venv at {venv_path}...")
        try:
            venvs_dir.mkdir(parents=True, exist_ok=True)

            # Ensure virtualenv is installed in embedded Python
            _log("Ensuring virtualenv is available...")
            process = await asyncio.create_subprocess_exec(
                str(python_embedded), "-m", "pip", "install", "virtualenv", "--quiet",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            await asyncio.wait_for(process.communicate(), timeout=120)

            # Create the venv with virtualenv
            process = await asyncio.create_subprocess_exec(
                str(python_embedded), "-m", "virtualenv", str(venv_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            if process.returncode != 0:
                err = stderr.decode().strip()
                _log(f"Venv creation failed: {err}", "error")
                return {"success": False, "error": f"Venv creation failed: {err[:300]}"}
            _log("Venv created successfully")
        except Exception as e:
            _log(f"Venv creation error: {e}", "error")
            return {"success": False, "error": str(e)}

        # Step 2: Run install steps (pip installs + git clones)
        venv_python_exe = venv_path / "Scripts" / "python.exe"
        git_exe = self.module_dir / "git_portable" / "cmd" / "git.exe"
        git_cmd = str(git_exe) if git_exe.is_file() else "git"

        for i, step in enumerate(steps):
            step_desc = step.get("description", f"Step {i+1}")
            _log(f"[{i+1}/{len(steps)}] {step_desc}")

            if step["type"] == "pip":
                args = step["args"]
                cmd = [str(venv_python_exe), "-m", "pip"] + args
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(self.module_dir),
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    )

                    async def _log_stream(stream, label):
                        while True:
                            line = await stream.readline()
                            if not line:
                                break
                            text = line.decode("utf-8", errors="replace").rstrip()
                            if text:
                                _log_to_file(f"[ENV-INSTALL {label}] {text}")

                    await asyncio.wait_for(
                        asyncio.gather(
                            _log_stream(process.stdout, "OUT"),
                            _log_stream(process.stderr, "ERR"),
                            process.wait(),
                        ),
                        timeout=600,  # 10 min per pip step
                    )

                    if process.returncode != 0:
                        _log(f"Pip install failed at step: {step_desc}", "error")
                        return {"success": False, "error": f"Failed at: {step_desc}. Check tts-debug.log."}

                except asyncio.TimeoutError:
                    return {"success": False, "error": f"Timeout at: {step_desc}"}
                except Exception as e:
                    return {"success": False, "error": f"Error at {step_desc}: {e}"}

            elif step["type"] == "git_clone":
                url = step["url"]
                repo_name = url.split("/")[-1].replace(".git", "")
                repos_dir = venv_path / "repos"
                repos_dir.mkdir(exist_ok=True)
                clone_path = repos_dir / repo_name

                try:
                    if clone_path.is_dir():
                        _log(f"Repo {repo_name} already exists, pulling...")
                        cmd = [git_cmd, "-C", str(clone_path), "pull"]
                    else:
                        cmd = [git_cmd, "clone", url, str(clone_path)]

                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    )
                    await asyncio.wait_for(process.communicate(), timeout=300)

                    # Editable install if requested
                    if step.get("editable") and (clone_path / "setup.py").is_file() or (clone_path / "pyproject.toml").is_file():
                        _log(f"Installing {repo_name} in editable mode...")
                        process = await asyncio.create_subprocess_exec(
                            str(venv_python_exe), "-m", "pip", "install", "-e", str(clone_path),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=str(self.module_dir),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                        )
                        await asyncio.wait_for(process.communicate(), timeout=300)

                except Exception as e:
                    _log(f"Git clone warning: {e}", "warning")
                    # Don't fail the whole install for a git clone issue

        _log(f"Environment {display_name} installed successfully!")
        return {"success": True, "message": f"Environment {display_name} installed"}

    async def download_model_weights(self, model_id: str) -> Dict[str, Any]:
        """Download model weights from HuggingFace."""
        if model_id not in MODEL_SETUP:
            return {"success": False, "error": f"Unknown model: {model_id}"}

        info = MODEL_SETUP[model_id]
        if not info["weights_repo"]:
            return {"success": True, "message": f"{info['display']} weights auto-download on first use"}

        models_dir = self.module_dir / "tts_models"
        weights_dir = models_dir / info["weights_dir"]

        # Check if already downloaded
        try:
            if weights_dir.is_dir() and any(weights_dir.iterdir()):
                return {"success": True, "message": f"{info['display']} weights already downloaded"}
        except (OSError, StopIteration):
            pass

        # Find a venv with huggingface_hub installed
        venvs_dir = self.module_dir / "venvs"
        hf_python = None
        env_name = info["env"]

        async def _has_huggingface_hub(python_path: str) -> bool:
            """Check if huggingface_hub is importable in the given python."""
            try:
                proc = await asyncio.create_subprocess_exec(
                    python_path, "-c", "import huggingface_hub",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
                await asyncio.wait_for(proc.communicate(), timeout=15)
                return proc.returncode == 0
            except Exception:
                return False

        # Prefer the model's own env
        own_env = venvs_dir / env_name / "Scripts" / "python.exe"
        if own_env.is_file() and await _has_huggingface_hub(str(own_env)):
            hf_python = str(own_env)
        else:
            # Try other envs
            for try_env in ["unified_env", "coqui_env", "chatterbox_env", "f5tts_env"]:
                p = venvs_dir / try_env / "Scripts" / "python.exe"
                if p.is_file() and await _has_huggingface_hub(str(p)):
                    hf_python = str(p)
                    break

        if not hf_python:
            return {"success": False, "error": f"No environment with huggingface_hub found. Install {ENV_DISPLAY.get(env_name, env_name)} first, then retry the download."}

        _log(f"Downloading {info['display']} weights from {info['weights_repo']}...")

        download_script = (
            f'import os\n'
            f'os.environ["HF_HOME"] = r"{models_dir}"\n'
            f'os.environ["HUGGINGFACE_HUB_CACHE"] = r"{models_dir / "hub"}"\n'
            f'from huggingface_hub import snapshot_download\n'
            f'snapshot_download(\n'
            f'    repo_id="{info["weights_repo"]}",\n'
            f'    local_dir=r"{weights_dir}",\n'
            f'    local_dir_use_symlinks=False\n'
            f')\n'
            f'print("Download complete!")\n'
        )

        try:
            process = await asyncio.create_subprocess_exec(
                hf_python, "-c", download_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.module_dir),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            async def _log_stream(stream, label):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        _log_to_file(f"[WEIGHTS {label}] {text}")

            await asyncio.wait_for(
                asyncio.gather(
                    _log_stream(process.stdout, "OUT"),
                    _log_stream(process.stderr, "ERR"),
                    process.wait(),
                ),
                timeout=1800,  # 30 min for large models
            )

            if process.returncode != 0:
                _log(f"Weights download failed for {model_id}", "error")
                return {"success": False, "error": f"Download failed. Check tts-debug.log."}

            _log(f"Weights downloaded for {info['display']}")
            return {"success": True, "message": f"{info['display']} weights downloaded to {weights_dir}"}

        except asyncio.TimeoutError:
            return {"success": False, "error": "Download timed out after 30 minutes"}
        except Exception as e:
            _log(f"Weights download error: {e}", "error")
            return {"success": False, "error": str(e)}

    # ======================== API Server Lifecycle ========================

    async def start_api_server(self) -> Dict[str, Any]:
        """Start the TTS API gateway server."""
        _log("start_api_server() called")

        if await self.is_api_running():
            _log("API server already running, skipping")
            return {"success": True, "message": "API server already running"}

        if not self.is_bootstrapped():
            _log("Not bootstrapped, cannot start API server", "error")
            return {"success": False, "error": "Module not bootstrapped. Run bootstrap first."}

        python_exe = self.module_dir / "python_embedded" / "python.exe"
        server_script = self.module_dir / "tts_api_server.py"

        if not python_exe.is_file():
            _log(f"Python not found at {python_exe}", "error")
            return {"success": False, "error": f"Python not found at {python_exe}"}

        if not server_script.is_file():
            _log(f"tts_api_server.py not found at {server_script}", "error")
            return {"success": False, "error": f"tts_api_server.py not found at {server_script}"}

        cmd = [str(python_exe), str(server_script), "--port", str(self.api_port)]
        _log(f"Starting TTS API server: {' '.join(cmd)}")
        _log(f"  CWD: {self.module_dir}")

        try:
            self.api_process = subprocess.Popen(
                cmd,
                cwd=str(self.module_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            _log(f"TTS API server process started, PID={self.api_process.pid}")

            # Start background threads to read subprocess output
            self._stdout_thread = threading.Thread(
                target=_pipe_reader,
                args=(self.api_process.stdout, "TTS-STDOUT"),
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=_pipe_reader,
                args=(self.api_process.stderr, "TTS-STDERR"),
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            if self.process_registry:
                self.process_registry.register(self.api_port, self.api_process.pid, "tts_api")

            # Wait for API to become ready
            for i in range(API_STARTUP_TIMEOUT):
                await asyncio.sleep(1)

                if self.api_process.poll() is not None:
                    _log(f"TTS API server process DIED after {i+1}s with code {self.api_process.returncode}", "error")
                    return {
                        "success": False,
                        "error": f"Process exited with code {self.api_process.returncode} after {i+1}s. Check tts-debug.log.",
                    }

                if await self.is_api_running():
                    _log(f"TTS API server ready on port {self.api_port} ({i+1}s)")
                    return {"success": True, "message": f"API server started on port {self.api_port}"}

                if (i + 1) % 5 == 0:
                    _log(f"Still waiting for TTS API server... ({i+1}s/{API_STARTUP_TIMEOUT}s)")

            _log(f"TTS API server did not start within {API_STARTUP_TIMEOUT}s", "error")
            return {"success": False, "error": f"API server did not start within {API_STARTUP_TIMEOUT}s"}

        except Exception as e:
            _log(f"Failed to start TTS API server: {type(e).__name__}: {e}", "error")
            return {"success": False, "error": str(e)}

    async def stop_api_server(self) -> Dict[str, Any]:
        """Stop the TTS API gateway server."""
        _log("stop_api_server() called")

        if not self.api_process and not await self.is_api_running():
            _log("API server not running, nothing to stop")
            return {"success": True, "message": "API server not running"}

        # Kill the API server process
        if self.api_process:
            pid = self.api_process.pid
            _log(f"Killing TTS API server process PID={pid}")
            try:
                if os.name == 'nt':
                    result = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    _log(f"taskkill result: rc={result.returncode}, stdout={result.stdout.decode().strip()}")
                    # Fallback to wmic if taskkill fails
                    if result.returncode != 0:
                        try:
                            subprocess.run(
                                ["wmic", "process", "where", f"ProcessId={pid}", "call", "terminate"],
                                capture_output=True, timeout=10,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                            )
                        except Exception:
                            pass
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception as e:
                _log(f"Error killing TTS API server process: {e}", "warning")

            if self.process_registry:
                self.process_registry.unregister(self.api_port)

            self.api_process = None

        # Invalidate caches
        self._api_running_cache = None
        self._status_cache = None

        _log("TTS API server stopped")
        return {"success": True, "message": "API server stopped"}

    # ======================== Proxy ========================

    async def proxy(self, method: str, path: str, **kwargs) -> Any:
        """Forward a request to the TTS API gateway."""
        url = f"http://127.0.0.1:{self.api_port}{path}"
        _log_to_file(f"[PROXY] {method} {url}")

        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            if any(op in path for op in ["/spawn", "/load", "/scale", "/tts/"]):
                timeout = 300  # 5 min — model loading can take a while
            else:
                timeout = PROXY_TIMEOUT

        try:
            client = await self._get_client()
            if method == "GET":
                r = await client.get(url, params=kwargs.get("params"), timeout=timeout)
            elif method == "POST":
                r = await client.post(url, json=kwargs.get("json"), timeout=timeout)
            elif method == "PUT":
                r = await client.put(url, json=kwargs.get("json"), timeout=timeout)
            elif method == "DELETE":
                r = await client.delete(url, params=kwargs.get("params"), timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            _log_to_file(f"[PROXY] {method} {path} -> {r.status_code} ({len(r.text)} bytes)")

            if r.status_code >= 400:
                error_text = r.text[:500]
                _log(f"Proxy error: {method} {path} -> {r.status_code}: {error_text}", "error")
                raise Exception(f"API returned {r.status_code}: {error_text}")

            try:
                return r.json()
            except Exception:
                return {"raw": r.text}
        except httpx.ConnectError:
            _log_to_file(f"[PROXY] {method} {path} -> ConnectError")
            _log(f"Proxy ConnectError: {method} {path}", "error")
            raise Exception(f"Cannot connect to TTS API on port {self.api_port}. Is the API server running?")
        except httpx.TimeoutException:
            _log_to_file(f"[PROXY] {method} {path} -> Timeout after {timeout}s")
            _log(f"Proxy timeout: {method} {path} after {timeout}s", "error")
            raise Exception(f"Request to TTS API timed out after {timeout}s: {method} {path}")

    # ======================== Shutdown ========================

    async def shutdown(self):
        """Clean shutdown: stop API server, close client."""
        _log("TTS manager shutting down...")

        try:
            await self.stop_api_server()
        except Exception as e:
            _log(f"Error during TTS shutdown: {e}", "error")

        await self._close_client()
        _log("TTS manager shutdown complete")
