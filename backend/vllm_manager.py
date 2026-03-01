"""
vLLM Module Manager

Manages the portable vLLM installation lifecycle:
- Single-step install: download release → extract → bootstrap (Python 3.10 + PyTorch + vLLM)
- Real-time progress tracking across all phases
- Report module status

Does NOT manage vLLM server processes — that stays in providers/vllm_provider.py.
"""

import os
import re
import asyncio
import logging
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("VLLMManager")

GITHUB_REPO = "rookiemann/vllm-windows-build"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# Debug log file for vLLM manager output
_debug_log_path: Optional[Path] = None


def _get_debug_log() -> Optional[Path]:
    global _debug_log_path
    if _debug_log_path is None:
        base = Path(__file__).parent.parent
        instances_dir = base / ".n8n-instances"
        if instances_dir.is_dir():
            _debug_log_path = instances_dir / "vllm-debug.log"
        else:
            _debug_log_path = base / "vllm-debug.log"
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


class VLLMManager:
    """Manages the portable vLLM module lifecycle (download + install only)."""

    def __init__(self, modules_dir: Path):
        self.modules_dir = modules_dir
        self.module_dir = modules_dir / "vllm"

        # Unified install progress tracking
        self._progress: Dict[str, Any] = {
            "active": False,
            "phase": "",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "detail": "",
            "error": "",
        }

        # Truncate the debug log on init
        try:
            log_path = _get_debug_log()
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("")
                _log(f"vLLM debug log initialized: {log_path}")
        except Exception:
            pass

        _log(f"vLLM manager init: modules_dir={modules_dir}, module_dir={self.module_dir}")

    # ======================== Status Checks ========================

    def is_module_downloaded(self) -> bool:
        """Check if the vLLM release has been downloaded and extracted."""
        result = (self.module_dir / "install.bat").is_file()
        _log(f"is_module_downloaded: {result} (checking {self.module_dir / 'install.bat'})", "debug")
        return result

    def is_installed(self) -> bool:
        """Check if vLLM has been fully installed (Python + dependencies)."""
        result = (self.module_dir / "python" / "python.exe").is_file()
        _log(f"is_installed: {result} (checking {self.module_dir / 'python' / 'python.exe'})", "debug")
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        installed = self.is_installed()

        if self._progress["active"]:
            status = "installing"
        elif installed:
            status = "installed"
        else:
            status = "not_installed"

        return {
            "status": status,
            "installed": installed,
            "module_dir": str(self.module_dir),
        }

    def get_install_progress(self) -> Dict[str, Any]:
        """Get current install progress."""
        return dict(self._progress)

    # ======================== Unified Install ========================

    async def install(self) -> Dict[str, Any]:
        """Single-step install: download → extract → bootstrap → missing deps."""
        if self.is_installed():
            _log("Already installed, skipping")
            return {"success": True, "message": "Already installed"}

        if self._progress["active"]:
            return {"success": False, "error": "Installation already in progress"}

        self._progress = {
            "active": True,
            "phase": "fetching_release",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "detail": "Checking latest release...",
            "error": "",
        }

        try:
            # Phase 1-3: Download and extract (skip if already downloaded)
            if not self.is_module_downloaded():
                result = await self._download_module()
                if not result["success"]:
                    return result
            else:
                _log("Module already downloaded, skipping to bootstrap")

            # Phase 4-7: Bootstrap (install Python, PyTorch, vLLM)
            if not self.is_installed():
                result = await self._bootstrap()
                if not result["success"]:
                    return result

            # Done
            self._progress["phase"] = "done"
            self._progress["detail"] = "Installation complete!"
            self._progress["active"] = False
            _log("vLLM installation completed successfully")
            return {"success": True, "message": "vLLM installed successfully"}

        except asyncio.CancelledError:
            self._progress["phase"] = "error"
            self._progress["error"] = "Installation cancelled"
            self._progress["detail"] = "Installation cancelled"
            self._progress["active"] = False
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            _log(f"Install exception: {error}", "error")
            self._progress["phase"] = "error"
            self._progress["error"] = error
            self._progress["detail"] = f"Error: {error}"
            self._progress["active"] = False
            return {"success": False, "error": error}

    # ======================== Internal: Download ========================

    async def _download_module(self) -> Dict[str, Any]:
        """Download the latest vLLM release from GitHub and extract to modules/vllm/."""
        self.modules_dir.mkdir(parents=True, exist_ok=True)

        self._progress["phase"] = "fetching_release"
        self._progress["detail"] = "Checking latest release..."

        try:
            # Step 1: Get latest release info from GitHub API
            _log(f"Fetching latest release from {GITHUB_API_URL}")
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(GITHUB_API_URL)
                if resp.status_code != 200:
                    error = f"GitHub API returned {resp.status_code}: {resp.text[:200]}"
                    _log(error, "error")
                    self._progress["phase"] = "error"
                    self._progress["error"] = error
                    self._progress["detail"] = f"Error: {error}"
                    self._progress["active"] = False
                    return {"success": False, "error": error}

                release = resp.json()

            # Find the .zip asset
            zip_asset = None
            for asset in release.get("assets", []):
                if asset["name"].endswith(".zip"):
                    zip_asset = asset
                    break

            if not zip_asset:
                error = "No .zip asset found in latest release"
                _log(error, "error")
                self._progress["phase"] = "error"
                self._progress["error"] = error
                self._progress["detail"] = f"Error: {error}"
                self._progress["active"] = False
                return {"success": False, "error": error}

            download_url = zip_asset["browser_download_url"]
            total_bytes = zip_asset.get("size", 0)
            filename = zip_asset["name"]

            _log(f"Downloading {filename} ({total_bytes / 1024 / 1024:.1f} MB) from {download_url}")

            self._progress.update({
                "phase": "downloading",
                "total_bytes": total_bytes,
                "downloaded_bytes": 0,
                "detail": "Starting download...",
            })

            # Step 2: Download the zip file with progress tracking
            zip_path = self.modules_dir / filename
            async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
                async with client.stream("GET", download_url) as response:
                    if response.status_code != 200:
                        error = f"Download failed with status {response.status_code}"
                        _log(error, "error")
                        self._progress["phase"] = "error"
                        self._progress["error"] = error
                        self._progress["detail"] = f"Error: {error}"
                        self._progress["active"] = False
                        return {"success": False, "error": error}

                    with open(zip_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            self._progress["downloaded_bytes"] += len(chunk)

            _log(f"Download complete: {zip_path} ({self._progress['downloaded_bytes']} bytes)")

            # Step 3: Extract the zip file
            self._progress["phase"] = "extracting"
            self._progress["detail"] = "Extracting files..."
            _log(f"Extracting {filename} to {self.module_dir}")

            await asyncio.to_thread(self._extract_zip, zip_path, self.module_dir)

            # Clean up the zip file
            try:
                zip_path.unlink()
                _log("Cleaned up zip file")
            except Exception as e:
                _log(f"Failed to clean up zip: {e}", "warning")

            if self.is_module_downloaded():
                _log("vLLM module downloaded and extracted successfully")
                return {"success": True, "message": "Module downloaded successfully"}
            else:
                error = "Extraction completed but install.bat not found — zip may have unexpected structure"
                _log(error, "error")
                self._progress["phase"] = "error"
                self._progress["error"] = error
                self._progress["detail"] = f"Error: {error}"
                self._progress["active"] = False
                return {"success": False, "error": error}

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            _log(f"Download exception: {error}", "error")
            self._progress["phase"] = "error"
            self._progress["error"] = error
            self._progress["detail"] = f"Error: {error}"
            self._progress["active"] = False
            return {"success": False, "error": error}

    def _extract_zip(self, zip_path: Path, target_dir: Path):
        """Extract zip file, handling nested directory structure."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check if zip has a single top-level directory
            top_level = set()
            for name in zf.namelist():
                parts = name.split("/")
                if parts[0]:
                    top_level.add(parts[0])

            if len(top_level) == 1:
                # Single top-level dir — extract contents directly to target_dir
                prefix = top_level.pop() + "/"
                target_dir.mkdir(parents=True, exist_ok=True)
                for member in zf.infolist():
                    if not member.filename.startswith(prefix):
                        continue
                    relative = member.filename[len(prefix):]
                    if not relative:
                        continue
                    dest = target_dir / relative
                    if member.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
                _log(f"Extracted {len(zf.namelist())} entries (stripped prefix '{prefix[:-1]}')")
            else:
                # Multiple top-level entries — extract as-is into target_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(target_dir)
                _log(f"Extracted {len(zf.namelist())} entries to {target_dir}")

    # ======================== Internal: Bootstrap ========================

    async def _bootstrap(self) -> Dict[str, Any]:
        """
        Run install.bat to install Python 3.10, PyTorch, and vLLM wheel.
        Parses stdout for [N/5] phase markers to update progress.
        """
        install_bat = self.module_dir / "install.bat"
        if not install_bat.is_file():
            error = "install.bat not found in module directory"
            _log(f"install.bat not found at {install_bat}", "error")
            self._progress["phase"] = "error"
            self._progress["error"] = error
            self._progress["detail"] = f"Error: {error}"
            self._progress["active"] = False
            return {"success": False, "error": error}

        _log("Running vLLM bootstrap (headless)...")
        self._progress["phase"] = "installing_python"
        self._progress["detail"] = "Installing Python 3.10..."

        # Create a modified script that removes pause commands
        bootstrap_script = self.module_dir / "_bootstrap_headless.bat"
        try:
            with open(install_bat, "r", encoding="utf-8") as f:
                content = f.read()

            _log(f"install.bat is {len(content)} chars")

            # Remove 'pause' commands that would block on piped stdin
            content = re.sub(r'^\s*pause\s*$', '    rem pause removed', content, flags=re.MULTILINE)
            content += "\necho [DONE] Bootstrap complete.\n"

            with open(bootstrap_script, "w", encoding="utf-8") as f:
                f.write(content)
            _log(f"Created bootstrap script: {bootstrap_script} ({len(content)} chars)")
        except Exception as e:
            error = f"Failed to create bootstrap script: {e}"
            _log(error, "error")
            self._progress["phase"] = "error"
            self._progress["error"] = error
            self._progress["detail"] = f"Error: {error}"
            self._progress["active"] = False
            return {"success": False, "error": error}

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

            async def _read_stdout(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        _log_to_file(f"[BOOTSTRAP OUT] {text}")
                        self._parse_bootstrap_phase(text)

            async def _read_stderr(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        _log_to_file(f"[BOOTSTRAP ERR] {text}")

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        _read_stdout(process.stdout),
                        _read_stderr(process.stderr),
                        process.wait(),
                    ),
                    timeout=1800,  # 30 minutes — vLLM + PyTorch downloads are large
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
                self._progress["phase"] = "error"
                self._progress["error"] = "Bootstrap timed out after 30 minutes"
                self._progress["detail"] = "Error: timed out after 30 minutes"
                self._progress["active"] = False
                return {"success": False, "error": "Bootstrap timed out after 30 minutes"}

            # Clean up temp script
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass

            _log(f"Bootstrap process exited with code {process.returncode}")

            if process.returncode == 0 or self.is_installed():
                _log("Bootstrap completed successfully")
                # Install missing deps
                self._progress["phase"] = "installing_deps"
                self._progress["detail"] = "Installing additional dependencies..."
                await self._install_missing_deps()
                return {"success": True, "message": "Bootstrap completed — vLLM is installed"}
            else:
                error = f"Bootstrap exited with code {process.returncode}. Check vllm-debug.log for details."
                _log(f"Bootstrap failed (rc={process.returncode})", "error")
                self._progress["phase"] = "error"
                self._progress["error"] = error
                self._progress["detail"] = f"Error: {error}"
                self._progress["active"] = False
                return {"success": False, "error": error}

        except asyncio.TimeoutError:
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            self._progress["phase"] = "error"
            self._progress["error"] = "Bootstrap timed out after 30 minutes"
            self._progress["detail"] = "Error: timed out after 30 minutes"
            self._progress["active"] = False
            return {"success": False, "error": "Bootstrap timed out after 30 minutes"}
        except Exception as e:
            _log(f"Bootstrap exception: {type(e).__name__}: {e}", "error")
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            error = str(e)
            self._progress["phase"] = "error"
            self._progress["error"] = error
            self._progress["detail"] = f"Error: {error}"
            self._progress["active"] = False
            return {"success": False, "error": error}

    def _parse_bootstrap_phase(self, line: str):
        """Parse install.bat stdout for [N/5] stage markers to update progress phase."""
        if "[1/5]" in line:
            self._progress["phase"] = "installing_python"
            self._progress["detail"] = "Installing Python 3.10..."
        elif "[2/5]" in line:
            self._progress["phase"] = "installing_python"
            self._progress["detail"] = "Setting up pip..."
        elif "[3/5]" in line:
            self._progress["phase"] = "installing_pytorch"
            self._progress["detail"] = "Installing PyTorch (~2.5 GB)... this takes several minutes"
        elif "[4/5]" in line:
            self._progress["phase"] = "installing_vllm"
            self._progress["detail"] = "Installing vLLM wheel..."
        elif "[5/5]" in line:
            self._progress["phase"] = "verifying"
            self._progress["detail"] = "Verifying installation..."
        elif "[DONE]" in line:
            self._progress["detail"] = "Bootstrap complete, finishing up..."

    async def _install_missing_deps(self):
        """Install dependencies missing from the vLLM wheel metadata.

        The upstream vllm-windows-build wheel doesn't declare these in its
        Requires-Dist, so pip install doesn't pull them in automatically.
        """
        python_exe = str(self.module_dir / "python" / "python.exe")
        if not os.path.exists(python_exe):
            _log("Cannot install missing deps — python.exe not found", "warning")
            return

        missing_deps = ["cbor2", "openai-harmony", "llguidance", "xgrammar"]
        _log(f"Installing missing vLLM deps: {', '.join(missing_deps)}")

        try:
            process = await asyncio.create_subprocess_exec(
                python_exe, "-m", "pip", "install",
                *missing_deps,
                "--no-warn-script-location",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=300
            )

            if process.returncode == 0:
                _log("Missing deps installed successfully")
            else:
                out = stdout.decode("utf-8", errors="replace")
                err = stderr.decode("utf-8", errors="replace")
                _log(f"Missing deps install failed (rc={process.returncode}): {err[:500]}", "warning")
                _log_to_file(f"[DEPS STDOUT] {out}")
                _log_to_file(f"[DEPS STDERR] {err}")
        except asyncio.TimeoutError:
            _log("Missing deps install timed out after 5 minutes", "warning")
        except Exception as e:
            _log(f"Missing deps install error: {e}", "warning")
