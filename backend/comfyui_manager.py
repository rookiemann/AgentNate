"""
ComfyUI Module Manager

Manages the portable ComfyUI installation lifecycle:
- Download/clone the portable installer from GitHub
- Bootstrap (Python, Git, FFmpeg)
- Start/stop the management API server
- Proxy requests to the management API
- Clean shutdown of all ComfyUI processes
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

logger = logging.getLogger("ComfyUIManager")

REPO_URL = "https://github.com/rookiemann/comfyui-portable-installer"
DEFAULT_API_PORT = 5000
API_STARTUP_TIMEOUT = 60
PROXY_TIMEOUT = 30

# Debug log file for ComfyUI subprocess output
_debug_log_path: Optional[Path] = None


def _get_debug_log() -> Optional[Path]:
    """Get the debug log path (set once from server.py's BASE_DIR)."""
    global _debug_log_path
    if _debug_log_path is None:
        # Try to find .n8n-instances directory relative to this file
        base = Path(__file__).parent.parent
        instances_dir = base / ".n8n-instances"
        if instances_dir.is_dir():
            _debug_log_path = instances_dir / "comfyui-debug.log"
        else:
            _debug_log_path = base / "comfyui-debug.log"
    return _debug_log_path


def _log_to_file(message: str):
    """Append a timestamped message to the ComfyUI debug log."""
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


class ComfyUIManager:
    """Manages the portable ComfyUI module lifecycle."""

    def __init__(self, modules_dir: Path, process_registry=None):
        self.modules_dir = modules_dir
        self.module_dir = modules_dir / "comfyui"
        self.api_port = DEFAULT_API_PORT
        self.api_process: Optional[subprocess.Popen] = None
        self.process_registry = process_registry
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # Shared HTTP client for proxy calls (connection pooling)
        self._http_client: Optional[httpx.AsyncClient] = None
        # TTL cache for get_status() with stampede lock
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_time: float = 0.0
        self._STATUS_CACHE_TTL: float = 5.0  # seconds
        self._status_lock: Optional[asyncio.Lock] = None
        # TTL cache for is_api_running (avoids 3s timeout spam)
        self._api_running_cache: Optional[bool] = None
        self._api_running_cache_time: float = 0.0
        self._API_RUNNING_CACHE_TTL: float = 3.0  # seconds

        # Truncate the debug log on init
        try:
            log_path = _get_debug_log()
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("")
                _log(f"ComfyUI debug log initialized: {log_path}")
        except Exception:
            pass

        _log(f"ComfyUI manager init: modules_dir={modules_dir}, module_dir={self.module_dir}")

    # ======================== Status Checks ========================

    def is_module_downloaded(self) -> bool:
        """Check if the portable installer repo has been cloned."""
        result = (self.module_dir / "installer_app.py").is_file()
        _log(f"is_module_downloaded: {result} (checking {self.module_dir / 'installer_app.py'})", "debug")
        return result

    def is_bootstrapped(self) -> bool:
        """Check if Python embedded + Git + FFmpeg have been downloaded."""
        path = self.module_dir / "python_embedded" / "python.exe"
        result = path.is_file()
        _log(f"is_bootstrapped: {result} (checking {path})", "debug")
        return result

    def is_comfyui_installed(self) -> bool:
        """Check if ComfyUI itself has been installed."""
        path = self.module_dir / "comfyui" / "main.py"
        result = path.is_file()
        _log(f"is_comfyui_installed: {result} (checking {path})", "debug")
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
        """Health check on the management API (cached for 3s)."""
        now = time.time()
        if self._api_running_cache is not None and (now - self._api_running_cache_time) < self._API_RUNNING_CACHE_TTL:
            return self._api_running_cache

        # Fast path: if we know the process isn't running, skip HTTP
        if self.api_process is not None and self.api_process.poll() is not None:
            self._api_running_cache = False
            self._api_running_cache_time = time.time()
            return False

        url = f"http://127.0.0.1:{self.api_port}/api/status"
        try:
            client = await self._get_client()
            r = await client.get(url, timeout=0.5)
            ok = r.status_code == 200
            self._api_running_cache = ok
            self._api_running_cache_time = time.time()
            return ok
        except Exception:
            self._api_running_cache = False
            self._api_running_cache_time = time.time()
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get combined status of the ComfyUI module (cached with 5s TTL, stampede-safe)."""
        now = time.time()
        if self._status_cache and (now - self._status_cache_time) < self._STATUS_CACHE_TTL:
            return self._status_cache

        # Lazy-init lock (must be in running event loop)
        if self._status_lock is None:
            self._status_lock = asyncio.Lock()

        async with self._status_lock:
            # Double-check after acquiring lock (another request may have refreshed)
            now = time.time()
            if self._status_cache and (now - self._status_cache_time) < self._STATUS_CACHE_TTL:
                return self._status_cache

            status = {
                "module_downloaded": self.is_module_downloaded(),
                "bootstrapped": self.is_bootstrapped(),
                "comfyui_installed": self.is_comfyui_installed(),
                "api_running": False,
                "api_port": self.api_port,
                "instances": [],
                "gpus": [],
            }

            if await self.is_api_running():
                status["api_running"] = True
                try:
                    data = await self.proxy("GET", "/api/status")
                    status["instances"] = data.get("instances", [])
                    status["gpus"] = data.get("gpus", [])
                except Exception as e:
                    _log(f"Failed to fetch API status details: {e}", "warning")

                try:
                    instances_data = await self.proxy("GET", "/api/instances")
                    if isinstance(instances_data, list):
                        status["instances"] = instances_data
                    elif isinstance(instances_data, dict) and "instances" in instances_data:
                        status["instances"] = instances_data["instances"]
                except Exception:
                    pass

            self._status_cache = status
            self._status_cache_time = time.time()
            return status

    # ======================== Module Lifecycle ========================

    async def download_module(self) -> Dict[str, Any]:
        """Clone the portable installer from GitHub."""
        if self.is_module_downloaded():
            _log("Module already downloaded, skipping clone")
            return {"success": True, "message": "Module already downloaded"}

        self.modules_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Cloning ComfyUI portable installer: {REPO_URL} -> {self.module_dir}")

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
                _log("ComfyUI module downloaded successfully")
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
        """Update the portable installer via git pull."""
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded yet"}

        _log("Updating ComfyUI module via git pull...")

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

    async def bootstrap(self) -> Dict[str, Any]:
        """
        Run the bootstrap process: download Python embedded, Git portable, FFmpeg.
        Creates a headless version of install.bat that skips the GUI launch at the end.
        """
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded. Download it first."}

        if self.is_bootstrapped():
            _log("Already bootstrapped, skipping")
            return {"success": True, "message": "Already bootstrapped"}

        install_bat = self.module_dir / "install.bat"
        if not install_bat.is_file():
            _log(f"install.bat not found at {install_bat}", "error")
            return {"success": False, "error": "install.bat not found in module directory"}

        _log("Running ComfyUI bootstrap (headless)...")

        # Create a modified script that does everything EXCEPT launching the GUI
        bootstrap_script = self.module_dir / "_bootstrap_headless.bat"
        try:
            with open(install_bat, "r", encoding="utf-8") as f:
                content = f.read()

            _log(f"install.bat is {len(content)} chars")

            # Find the GUI launch section and truncate before it
            for marker in [":: Step 7: Launch the installer", "[8/8] Launching", "installer_app.py"]:
                idx = content.find(marker)
                if idx > 0:
                    sep_idx = content.rfind(":: ====", 0, idx)
                    cut_point = sep_idx if sep_idx > 0 else idx
                    content = content[:cut_point]
                    _log(f"Truncated install.bat at marker '{marker}' (pos {idx}, cut at {cut_point})")
                    break
            else:
                _log("WARNING: No GUI launch marker found in install.bat, using full script", "warning")

            # Remove 'pause' commands that would block on piped stdin
            content = re.sub(r'^\s*pause\s*$', '    rem pause removed', content, flags=re.MULTILINE)
            content += "\necho [DONE] Bootstrap complete.\nendlocal\n"

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

            # Read output in real-time
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
                    timeout=600,
                )
            except asyncio.TimeoutError:
                _log("Bootstrap timed out after 10 minutes", "error")
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    bootstrap_script.unlink(missing_ok=True)
                except Exception:
                    pass
                return {"success": False, "error": "Bootstrap timed out after 10 minutes"}

            # Clean up temp script
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass

            _log(f"Bootstrap process exited with code {process.returncode}")

            if process.returncode == 0 or self.is_bootstrapped():
                _log("Bootstrap completed successfully")
                return {"success": True, "message": "Bootstrap completed"}
            else:
                _log(f"Bootstrap failed (rc={process.returncode})", "error")
                return {"success": False, "error": f"Bootstrap exited with code {process.returncode}. Check comfyui-debug.log for details."}
        except asyncio.TimeoutError:
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": "Bootstrap timed out after 10 minutes"}
        except Exception as e:
            _log(f"Bootstrap exception: {type(e).__name__}: {e}", "error")
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": str(e)}

    # ======================== API Server Lifecycle ========================

    async def start_api_server(self) -> Dict[str, Any]:
        """Start the ComfyUI management API server."""
        _log("start_api_server() called")

        if await self.is_api_running():
            _log("API server already running, skipping")
            return {"success": True, "message": "API server already running"}

        if not self.is_bootstrapped():
            _log("Not bootstrapped, cannot start API server", "error")
            return {"success": False, "error": "Module not bootstrapped. Run bootstrap first."}

        python_exe = self.module_dir / "python_embedded" / "python.exe"
        installer_app = self.module_dir / "installer_app.py"

        if not python_exe.is_file():
            _log(f"Python not found at {python_exe}", "error")
            return {"success": False, "error": f"Python not found at {python_exe}"}

        if not installer_app.is_file():
            _log(f"installer_app.py not found at {installer_app}", "error")
            return {"success": False, "error": f"installer_app.py not found at {installer_app}"}

        # Ensure aiohttp is installed
        _log("Ensuring aiohttp dependency is installed...")
        try:
            dep_proc = await asyncio.create_subprocess_exec(
                str(python_exe), "-m", "pip", "install", "aiohttp", "--quiet",
                cwd=str(self.module_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            dep_stdout, dep_stderr = await asyncio.wait_for(dep_proc.communicate(), timeout=120)
            dep_out = dep_stdout.decode().strip()
            dep_err = dep_stderr.decode().strip()
            if dep_out:
                _log(f"[PIP STDOUT] {dep_out}")
            if dep_err:
                _log(f"[PIP STDERR] {dep_err}")
            _log(f"pip install aiohttp exited with code {dep_proc.returncode}")
        except Exception as e:
            _log(f"Failed to install aiohttp: {type(e).__name__}: {e}", "warning")

        cmd = [str(python_exe), str(installer_app), "--api", "--api-port", str(self.api_port)]
        _log(f"Starting API server: {' '.join(cmd)}")
        _log(f"  CWD: {self.module_dir}")

        # Pre-spawn port availability check to catch bind conflicts early
        import socket as _socket
        _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            _sock.bind(('127.0.0.1', self.api_port))
            _sock.close()
        except OSError as e:
            _sock.close()
            _log(f"Port {self.api_port} already in use: {e}", "error")
            return {"success": False, "error": f"Port {self.api_port} already in use ({e}). Stop the conflicting process or use a different port."}

        try:
            self.api_process = subprocess.Popen(
                cmd,
                cwd=str(self.module_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            _log(f"API server process started, PID={self.api_process.pid}")

            # Start background threads to read subprocess output
            self._stdout_thread = threading.Thread(
                target=_pipe_reader,
                args=(self.api_process.stdout, "API-STDOUT"),
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=_pipe_reader,
                args=(self.api_process.stderr, "API-STDERR"),
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            if self.process_registry:
                self.process_registry.register(self.api_port, self.api_process.pid, "comfyui_api")

            # Invalidate cached readiness so we don't get stale True from a previous instance
            self._api_running_cache = None
            self._api_running_cache_time = 0

            # Wait for API to become ready
            for i in range(API_STARTUP_TIMEOUT):
                await asyncio.sleep(1)

                # Check if process died
                if self.api_process.poll() is not None:
                    _log(f"API server process DIED after {i+1}s with code {self.api_process.returncode}", "error")
                    _log("Check comfyui-debug.log for API-STDOUT/API-STDERR output", "error")
                    return {
                        "success": False,
                        "error": f"Process exited with code {self.api_process.returncode} after {i+1}s. Check comfyui-debug.log.",
                    }

                if await self.is_api_running():
                    _log(f"ComfyUI API server ready on port {self.api_port} ({i+1}s)")
                    return {"success": True, "message": f"API server started on port {self.api_port}"}

                if (i + 1) % 5 == 0:
                    _log(f"Still waiting for API server... ({i+1}s/{API_STARTUP_TIMEOUT}s)")

            _log(f"API server did not start within {API_STARTUP_TIMEOUT}s", "error")
            return {"success": False, "error": f"API server did not start within {API_STARTUP_TIMEOUT}s"}

        except Exception as e:
            _log(f"Failed to start API server: {type(e).__name__}: {e}", "error")
            return {"success": False, "error": str(e)}

    async def stop_api_server(self) -> Dict[str, Any]:
        """Stop the ComfyUI management API server."""
        _log("stop_api_server() called")

        if not self.api_process and not await self.is_api_running():
            _log("API server not running, nothing to stop")
            return {"success": True, "message": "API server not running"}

        # First try to stop all ComfyUI instances via API
        if await self.is_api_running():
            try:
                _log("Sending stop-all to instances...")
                await self.proxy("POST", "/api/instances/stop-all")
                await asyncio.sleep(1)
            except Exception as e:
                _log(f"stop-all failed (non-fatal): {e}", "warning")

        # Kill the API server process
        if self.api_process:
            pid = self.api_process.pid
            _log(f"Killing API server process PID={pid}")
            try:
                if os.name == 'nt':
                    result = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    _log(f"taskkill result: rc={result.returncode}, stdout={result.stdout.decode().strip()}")
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception as e:
                _log(f"Error killing API server process: {e}", "warning")

            if self.process_registry:
                self.process_registry.unregister(self.api_port)

            self.api_process = None

        _log("ComfyUI API server stopped")
        return {"success": True, "message": "API server stopped"}

    # ======================== Proxy ========================

    async def proxy(self, method: str, path: str, **kwargs) -> Any:
        """Forward a request to the management API."""
        url = f"http://127.0.0.1:{self.api_port}{path}"
        _log_to_file(f"[PROXY] {method} {url}")

        # Use longer timeout for install/download/update operations
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            if any(op in path for op in ["/install", "/download", "/update", "/start"]):
                timeout = 120
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
            _log_to_file(f"[PROXY] {method} {path} -> ConnectError (API not running?)")
            _log(f"Proxy ConnectError: {method} {path} — API server on port {self.api_port} is not reachable", "error")
            raise Exception(f"Cannot connect to ComfyUI API on port {self.api_port}. Is the API server running?")
        except httpx.TimeoutException:
            _log_to_file(f"[PROXY] {method} {path} -> Timeout after {timeout}s")
            _log(f"Proxy timeout: {method} {path} after {timeout}s", "error")
            raise Exception(f"Request to ComfyUI API timed out after {timeout}s: {method} {path}")

    # ======================== Shutdown ========================

    async def shutdown(self):
        """Clean shutdown: stop all instances, stop API server."""
        _log("ComfyUI manager shutting down...")

        try:
            await self.stop_api_server()
        except Exception as e:
            _log(f"Error during ComfyUI shutdown: {e}", "error")

        await self._close_client()
        _log("ComfyUI manager shutdown complete")
