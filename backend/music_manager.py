"""
Music Module Manager

Manages the portable Music server lifecycle:
- Download/clone the portable Music server from GitHub
- Bootstrap (Python, Git, FFmpeg)
- Start/stop the Music API gateway server
- Proxy requests to the Music API
- Clean shutdown of all processes
"""

import os
import asyncio
import logging
import subprocess
import signal
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("MusicManager")

REPO_URL = "https://github.com/rookiemann/portable-music-server"
DEFAULT_API_PORT = 9150
API_STARTUP_TIMEOUT = 120
PROXY_TIMEOUT = 30

# Debug log file for Music subprocess output
_debug_log_path: Optional[Path] = None


def _get_debug_log() -> Optional[Path]:
    global _debug_log_path
    if _debug_log_path is None:
        base = Path(__file__).parent.parent
        instances_dir = base / ".n8n-instances"
        if instances_dir.is_dir():
            _debug_log_path = instances_dir / "music-debug.log"
        else:
            _debug_log_path = base / "music-debug.log"
    return _debug_log_path


def _log_to_file(message: str):
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
    getattr(logger, level, logger.info)(msg)
    prefix = {"error": "ERROR", "warning": "WARN", "debug": "DEBUG"}.get(level, "INFO")
    _log_to_file(f"[{prefix}] {msg}")


def _pipe_reader(pipe, label: str):
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


class MusicManager:
    """Manages the portable Music server lifecycle."""

    def __init__(self, modules_dir: Path, process_registry=None):
        self.modules_dir = modules_dir
        self.module_dir = modules_dir / "music"
        self.api_port = DEFAULT_API_PORT
        self.api_process: Optional[subprocess.Popen] = None
        self.process_registry = process_registry
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
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
                _log(f"Music debug log initialized: {log_path}")
        except Exception:
            pass

        _log(f"Music manager init: modules_dir={modules_dir}, module_dir={self.module_dir}")

    # ======================== Status Checks ========================

    def is_module_downloaded(self) -> bool:
        result = (self.module_dir / "music_api_server.py").is_file()
        _log(f"is_module_downloaded: {result}", "debug")
        return result

    def is_bootstrapped(self) -> bool:
        path = self.module_dir / "python_embedded" / "python.exe"
        result = path.is_file()
        _log(f"is_bootstrapped: {result}", "debug")
        return result

    def is_installed(self) -> bool:
        sp = self.module_dir / "python_embedded" / "Lib" / "site-packages"
        result = (sp / "fastapi").is_dir() and (sp / "uvicorn").is_dir()
        _log(f"is_installed: {result}", "debug")
        return result

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=PROXY_TIMEOUT,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    async def _close_client(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def is_api_running(self) -> bool:
        now = time.time()
        if self._api_running_cache is not None and (now - self._api_running_cache_time) < self._API_RUNNING_CACHE_TTL:
            return self._api_running_cache

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
        now = time.time()
        if self._status_cache and (now - self._status_cache_time) < self._STATUS_CACHE_TTL:
            return self._status_cache

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
        if self.is_module_downloaded():
            _log("Module already downloaded, skipping clone")
            return {"success": True, "message": "Module already downloaded"}

        self.modules_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Cloning Music server: {REPO_URL} -> {self.module_dir}")

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
                _log("Music module downloaded successfully")
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
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded yet"}

        _log("Updating Music module via git pull...")

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
        return f'''@echo off
rem ============================================
rem  Music Module - Headless Bootstrap
rem  Generated by AgentNate MusicManager
rem ============================================

set "SD={sd}"
set "PYTHON_DIR=%SD%python_embedded"
set "PYTHON_EXE=%PYTHON_DIR%\\python.exe"
set "GIT_DIR=%SD%git_portable"
set "GIT_EXE=%GIT_DIR%\\cmd\\git.exe"
set "FFMPEG_DIR=%SD%ffmpeg"

rem ---- Step 1: Download embedded Python ----
if exist "%PYTHON_EXE%" (
    echo [OK] Python already installed.
    goto :step_pip
)
echo [1/6] Downloading Python 3.10.11 embedded...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile '%SD%python_embedded.zip'"
if not exist "%SD%python_embedded.zip" echo ERROR: Python download failed. && exit /b 1

echo [1/6] Extracting Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%python_embedded.zip' -DestinationPath '%PYTHON_DIR%' -Force"
del "%SD%python_embedded.zip" 2>nul
if not exist "%PYTHON_EXE%" echo ERROR: Python extraction failed. && exit /b 1

echo [1/6] Configuring Python pth file...
if not exist "%PYTHON_DIR%\\Lib\\site-packages" mkdir "%PYTHON_DIR%\\Lib\\site-packages"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$pth = Get-ChildItem '%PYTHON_DIR%\\python*._pth' | Select-Object -First 1; if ($pth) {{ $zip = (Get-ChildItem '%PYTHON_DIR%\\python*.zip' | Select-Object -First 1).Name; if (-not $zip) {{ $zip = 'python310.zip' }}; @($zip, '.', 'Lib', 'Lib\\site-packages', '', 'import site') | Set-Content -Path $pth.FullName -Encoding ASCII; Write-Host 'Configured:' $pth.Name }}"

rem ---- Step 2: Install pip ----
:step_pip
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] pip already available.
    goto :step_git
)
echo [2/6] Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\\get-pip.py'"
if not exist "%PYTHON_DIR%\\get-pip.py" echo ERROR: get-pip download failed. && exit /b 1
echo [2/6] Installing pip...
"%PYTHON_EXE%" "%PYTHON_DIR%\\get-pip.py"
del "%PYTHON_DIR%\\get-pip.py" 2>nul
"%PYTHON_EXE%" -m pip install --upgrade pip 2>nul

rem ---- Step 3: Download portable Git ----
:step_git
if exist "%GIT_EXE%" (
    echo [OK] Portable Git already installed.
    goto :step_ffmpeg
)
echo [3/6] Downloading portable Git 2.47.1...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip' -OutFile '%SD%git_portable.zip'"
if not exist "%SD%git_portable.zip" (
    echo WARNING: Git download failed.
    goto :step_ffmpeg
)
echo [3/6] Extracting Git...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%git_portable.zip' -DestinationPath '%GIT_DIR%' -Force"
del "%SD%git_portable.zip" 2>nul

rem ---- Step 4: Download portable FFmpeg ----
:step_ffmpeg
if exist "%FFMPEG_DIR%\\bin\\ffmpeg.exe" (
    echo [OK] FFmpeg already installed.
    goto :step_path
)
echo [4/6] Downloading portable FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%SD%ffmpeg_portable.zip'"
if not exist "%SD%ffmpeg_portable.zip" (
    echo WARNING: FFmpeg download failed.
    goto :step_path
)
echo [4/6] Extracting FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%SD%ffmpeg_portable.zip' -DestinationPath '%FFMPEG_DIR%' -Force"
del "%SD%ffmpeg_portable.zip" 2>nul

rem ---- Step 5: Set up PATH and install requirements ----
:step_path
if exist "%GIT_EXE%" set "PATH=%GIT_DIR%\\cmd;%PATH%"
if exist "%FFMPEG_DIR%\\bin\\ffmpeg.exe" set "PATH=%FFMPEG_DIR%\\bin;%PATH%"
for /d %%D in ("%FFMPEG_DIR%\\ffmpeg-*") do if exist "%%D\\bin\\ffmpeg.exe" set "PATH=%%D\\bin;%PATH%"

echo [5/6] Installing requirements...
"%PYTHON_EXE%" -m pip install -r "%SD%requirements.txt" --quiet 2>nul
if errorlevel 1 "%PYTHON_EXE%" -m pip install -r "%SD%requirements.txt"

echo [6/6] Done.
echo.
echo [DONE] Bootstrap complete.
'''

    async def bootstrap(self) -> Dict[str, Any]:
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded. Download it first."}

        if self.is_installed():
            _log("Already fully installed, skipping bootstrap")
            return {"success": True, "message": "Already installed"}

        _log("Running Music bootstrap (headless)...")

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
                    timeout=1800,
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
                return {"success": False, "error": f"Bootstrap exited with code {process.returncode}. Check music-debug.log for details."}
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

    # ======================== API Server Lifecycle ========================

    async def start_api_server(self) -> Dict[str, Any]:
        _log("start_api_server() called")

        if await self.is_api_running():
            _log("API server already running, skipping")
            return {"success": True, "message": "API server already running"}

        if not self.is_bootstrapped():
            _log("Not bootstrapped, cannot start API server", "error")
            return {"success": False, "error": "Module not bootstrapped. Run bootstrap first."}

        python_exe = self.module_dir / "python_embedded" / "python.exe"
        server_script = self.module_dir / "music_api_server.py"

        if not python_exe.is_file():
            _log(f"Python not found at {python_exe}", "error")
            return {"success": False, "error": f"Python not found at {python_exe}"}

        if not server_script.is_file():
            _log(f"music_api_server.py not found at {server_script}", "error")
            return {"success": False, "error": f"music_api_server.py not found at {server_script}"}

        cmd = [str(python_exe), str(server_script), "--port", str(self.api_port)]
        _log(f"Starting Music API server: {' '.join(cmd)}")

        try:
            self.api_process = subprocess.Popen(
                cmd,
                cwd=str(self.module_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            _log(f"Music API server process started, PID={self.api_process.pid}")

            self._stdout_thread = threading.Thread(
                target=_pipe_reader,
                args=(self.api_process.stdout, "MUSIC-STDOUT"),
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=_pipe_reader,
                args=(self.api_process.stderr, "MUSIC-STDERR"),
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            if self.process_registry:
                self.process_registry.register(self.api_port, self.api_process.pid, "music_api")

            for i in range(API_STARTUP_TIMEOUT):
                await asyncio.sleep(1)

                if self.api_process.poll() is not None:
                    _log(f"Music API server process DIED after {i+1}s with code {self.api_process.returncode}", "error")
                    return {
                        "success": False,
                        "error": f"Process exited with code {self.api_process.returncode} after {i+1}s. Check music-debug.log.",
                    }

                if await self.is_api_running():
                    _log(f"Music API server ready on port {self.api_port} ({i+1}s)")
                    return {"success": True, "message": f"API server started on port {self.api_port}"}

                if (i + 1) % 5 == 0:
                    _log(f"Still waiting for Music API server... ({i+1}s/{API_STARTUP_TIMEOUT}s)")

            _log(f"Music API server did not start within {API_STARTUP_TIMEOUT}s", "error")
            return {"success": False, "error": f"API server did not start within {API_STARTUP_TIMEOUT}s"}

        except Exception as e:
            _log(f"Failed to start Music API server: {type(e).__name__}: {e}", "error")
            return {"success": False, "error": str(e)}

    async def stop_api_server(self) -> Dict[str, Any]:
        _log("stop_api_server() called")

        if not self.api_process and not await self.is_api_running():
            _log("API server not running, nothing to stop")
            return {"success": True, "message": "API server not running"}

        if self.api_process:
            pid = self.api_process.pid
            _log(f"Killing Music API server process PID={pid}")
            try:
                if os.name == 'nt':
                    result = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    _log(f"taskkill result: rc={result.returncode}")
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
                _log(f"Error killing Music API server process: {e}", "warning")

            if self.process_registry:
                self.process_registry.unregister(self.api_port)

            self.api_process = None

        self._api_running_cache = None
        self._status_cache = None

        _log("Music API server stopped")
        return {"success": True, "message": "API server stopped"}

    # ======================== Proxy ========================

    async def proxy(self, method: str, path: str, **kwargs) -> Any:
        url = f"http://127.0.0.1:{self.api_port}{path}"
        _log_to_file(f"[PROXY] {method} {url}")

        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            if any(op in path for op in ["/spawn", "/load", "/scale", "/music/", "/install/"]):
                timeout = 600  # 10 min — model loading + generation can be slow
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
                r = await client.delete(url, json=kwargs.get("json"), params=kwargs.get("params"), timeout=timeout)
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
            _log(f"Proxy ConnectError: {method} {path}", "error")
            raise Exception(f"Cannot connect to Music API on port {self.api_port}. Is the API server running?")
        except httpx.TimeoutException:
            _log(f"Proxy timeout: {method} {path} after {timeout}s", "error")
            raise Exception(f"Request to Music API timed out after {timeout}s: {method} {path}")

    # ======================== Shutdown ========================

    async def shutdown(self):
        _log("Music manager shutting down...")
        try:
            await self.stop_api_server()
        except Exception as e:
            _log(f"Error during Music shutdown: {e}", "error")
        await self._close_client()
        _log("Music manager shutdown complete")
