"""
vLLM Module Manager

Manages the portable vLLM installation lifecycle:
- Download pre-built release from GitHub (rookiemann/vllm-windows-build)
- Bootstrap (run install.bat which downloads Python 3.10, PyTorch, installs vllm wheel)
- Track download progress for UI progress bar
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

        # Download progress tracking
        self._download_progress: Dict[str, Any] = {
            "active": False,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "filename": "",
            "phase": "",  # "downloading", "extracting", "done", "error"
            "error": "",
        }

        # Bootstrap state
        self._bootstrapping = False

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
        downloaded = self.is_module_downloaded()
        installed = self.is_installed()

        if self._bootstrapping:
            status = "bootstrapping"
        elif installed:
            status = "installed"
        elif downloaded:
            status = "downloaded"
        elif self._download_progress["active"]:
            status = "downloading"
        else:
            status = "not_downloaded"

        return {
            "status": status,
            "module_downloaded": downloaded,
            "installed": installed,
            "bootstrapping": self._bootstrapping,
            "module_dir": str(self.module_dir),
        }

    def get_download_progress(self) -> Dict[str, Any]:
        """Get current download progress."""
        return dict(self._download_progress)

    # ======================== Module Lifecycle ========================

    async def download_module(self) -> Dict[str, Any]:
        """Download the latest vLLM release from GitHub and extract to modules/vllm/."""
        if self.is_module_downloaded():
            _log("Module already downloaded, skipping")
            return {"success": True, "message": "Module already downloaded"}

        self.modules_dir.mkdir(parents=True, exist_ok=True)

        self._download_progress = {
            "active": True,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "filename": "",
            "phase": "fetching_release",
            "error": "",
        }

        try:
            # Step 1: Get latest release info from GitHub API
            _log(f"Fetching latest release from {GITHUB_API_URL}")
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(GITHUB_API_URL)
                if resp.status_code != 200:
                    error = f"GitHub API returned {resp.status_code}: {resp.text[:200]}"
                    _log(error, "error")
                    self._download_progress["phase"] = "error"
                    self._download_progress["error"] = error
                    self._download_progress["active"] = False
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
                self._download_progress["phase"] = "error"
                self._download_progress["error"] = error
                self._download_progress["active"] = False
                return {"success": False, "error": error}

            download_url = zip_asset["browser_download_url"]
            total_bytes = zip_asset.get("size", 0)
            filename = zip_asset["name"]

            _log(f"Downloading {filename} ({total_bytes / 1024 / 1024:.1f} MB) from {download_url}")

            self._download_progress.update({
                "phase": "downloading",
                "filename": filename,
                "total_bytes": total_bytes,
                "downloaded_bytes": 0,
            })

            # Step 2: Download the zip file with progress tracking
            zip_path = self.modules_dir / filename
            async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
                async with client.stream("GET", download_url) as response:
                    if response.status_code != 200:
                        error = f"Download failed with status {response.status_code}"
                        _log(error, "error")
                        self._download_progress["phase"] = "error"
                        self._download_progress["error"] = error
                        self._download_progress["active"] = False
                        return {"success": False, "error": error}

                    with open(zip_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            self._download_progress["downloaded_bytes"] += len(chunk)

            _log(f"Download complete: {zip_path} ({self._download_progress['downloaded_bytes']} bytes)")

            # Step 3: Extract the zip file
            self._download_progress["phase"] = "extracting"
            _log(f"Extracting {filename} to {self.module_dir}")

            # Run extraction in a thread to avoid blocking the event loop
            await asyncio.to_thread(self._extract_zip, zip_path, self.module_dir)

            # Clean up the zip file
            try:
                zip_path.unlink()
                _log("Cleaned up zip file")
            except Exception as e:
                _log(f"Failed to clean up zip: {e}", "warning")

            self._download_progress["phase"] = "done"
            self._download_progress["active"] = False

            if self.is_module_downloaded():
                _log("vLLM module downloaded and extracted successfully")
                return {"success": True, "message": "Module downloaded successfully"}
            else:
                error = "Extraction completed but install.bat not found — zip may have unexpected structure"
                _log(error, "error")
                self._download_progress["phase"] = "error"
                self._download_progress["error"] = error
                return {"success": False, "error": error}

        except asyncio.CancelledError:
            self._download_progress["phase"] = "error"
            self._download_progress["error"] = "Download cancelled"
            self._download_progress["active"] = False
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            _log(f"Download exception: {error}", "error")
            self._download_progress["phase"] = "error"
            self._download_progress["error"] = error
            self._download_progress["active"] = False
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

    async def bootstrap(self) -> Dict[str, Any]:
        """
        Run install.bat to install Python 3.10, PyTorch, and vLLM wheel.
        Creates a headless version that skips any interactive prompts.
        """
        if not self.is_module_downloaded():
            return {"success": False, "error": "Module not downloaded. Download it first."}

        if self.is_installed():
            _log("Already installed, skipping bootstrap")
            return {"success": True, "message": "Already installed"}

        install_bat = self.module_dir / "install.bat"
        if not install_bat.is_file():
            _log(f"install.bat not found at {install_bat}", "error")
            return {"success": False, "error": "install.bat not found in module directory"}

        _log("Running vLLM bootstrap (headless)...")
        self._bootstrapping = True

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
            self._bootstrapping = False
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
                self._bootstrapping = False
                return {"success": False, "error": "Bootstrap timed out after 30 minutes"}

            # Clean up temp script
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass

            _log(f"Bootstrap process exited with code {process.returncode}")
            self._bootstrapping = False

            if process.returncode == 0 or self.is_installed():
                _log("Bootstrap completed successfully")
                return {"success": True, "message": "Bootstrap completed — vLLM is installed"}
            else:
                _log(f"Bootstrap failed (rc={process.returncode})", "error")
                return {"success": False, "error": f"Bootstrap exited with code {process.returncode}. Check vllm-debug.log for details."}

        except asyncio.TimeoutError:
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            self._bootstrapping = False
            return {"success": False, "error": "Bootstrap timed out after 30 minutes"}
        except Exception as e:
            _log(f"Bootstrap exception: {type(e).__name__}: {e}", "error")
            try:
                bootstrap_script.unlink(missing_ok=True)
            except Exception:
                pass
            self._bootstrapping = False
            return {"success": False, "error": str(e)}
