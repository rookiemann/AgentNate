"""
llama.cpp Module Manager

Manages llama-cpp-python installation into AgentNate's own python/ environment:
- Check installation status via marker file
- Download pre-built CUDA wheel from GitHub
- pip install into python/
- Track install progress for UI

Much simpler than vLLM/ComfyUI — no separate Python, no bootstrap, just a single wheel install.
"""

import os
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

import httpx

logger = logging.getLogger("LlamaCppManager")

WHEEL_URL = (
    "https://github.com/rookiemann/llama-cpp-python-py314-cuda131-wheel"
    "/releases/download/v0.3.16-cuda13.1-py3.14"
    "/llama_cpp_python-0.3.16-cp314-cp314-win_amd64.whl"
)
WHEEL_FILENAME = "llama_cpp_python-0.3.16-cp314-cp314-win_amd64.whl"


class LlamaCppManager:
    """Manages llama-cpp-python wheel installation into AgentNate's python/ env."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.python_dir = base_dir / "python"
        self.python_exe = self.python_dir / "python.exe"
        self.marker_file = self.python_dir / ".llama-cuda-installed"

        # Install progress tracking
        self._progress: Dict[str, Any] = {
            "active": False,
            "phase": "",       # "downloading", "installing", "done", "error"
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "error": "",
        }

    def is_installed(self) -> bool:
        """Check if llama-cpp-python is installed (marker file exists)."""
        return self.marker_file.is_file()

    def get_status(self) -> Dict[str, Any]:
        """Get current installation status."""
        installed = self.is_installed()

        if self._progress["active"]:
            status = self._progress["phase"]
        elif installed:
            status = "installed"
        else:
            status = "not_installed"

        return {
            "status": status,
            "installed": installed,
        }

    def get_install_progress(self) -> Dict[str, Any]:
        """Get current install progress."""
        return dict(self._progress)

    async def install_wheel(self) -> Dict[str, Any]:
        """Download the CUDA wheel from GitHub and pip install it."""
        if self.is_installed():
            return {"success": True, "message": "Already installed"}

        if not self.python_exe.is_file():
            return {"success": False, "error": "python/python.exe not found. Run install.bat first."}

        self._progress = {
            "active": True,
            "phase": "downloading",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "error": "",
        }

        wheel_path = self.python_dir / WHEEL_FILENAME

        try:
            # Step 1: Download wheel
            logger.info(f"Downloading llama-cpp-python wheel from {WHEEL_URL}")

            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                async with client.stream("GET", WHEEL_URL) as response:
                    if response.status_code != 200:
                        error = f"Download failed with status {response.status_code}"
                        logger.error(error)
                        self._progress.update(phase="error", error=error, active=False)
                        return {"success": False, "error": error}

                    total = int(response.headers.get("content-length", 0))
                    self._progress["total_bytes"] = total

                    with open(wheel_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            self._progress["downloaded_bytes"] += len(chunk)

            logger.info(f"Download complete: {wheel_path} ({self._progress['downloaded_bytes']} bytes)")

            # Step 2: pip install
            self._progress["phase"] = "installing"
            logger.info("Installing wheel with pip...")

            process = await asyncio.create_subprocess_exec(
                str(self.python_exe), "-m", "pip", "install",
                str(wheel_path), "--no-deps", "--no-warn-script-location",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_dir),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

            # Clean up wheel file
            try:
                wheel_path.unlink()
            except Exception:
                pass

            if process.returncode != 0:
                error_text = stderr.decode("utf-8", errors="replace").strip()
                error = f"pip install failed (rc={process.returncode}): {error_text[:500]}"
                logger.error(error)
                self._progress.update(phase="error", error=error, active=False)
                return {"success": False, "error": error}

            # Step 3: Create marker file
            import datetime
            self.marker_file.write_text(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                encoding="utf-8",
            )

            self._progress.update(phase="done", active=False)
            logger.info("llama-cpp-python installed successfully")
            return {"success": True, "message": "llama-cpp-python installed successfully"}

        except asyncio.TimeoutError:
            try:
                wheel_path.unlink(missing_ok=True)
            except Exception:
                pass
            error = "Installation timed out"
            logger.error(error)
            self._progress.update(phase="error", error=error, active=False)
            return {"success": False, "error": error}

        except asyncio.CancelledError:
            self._progress.update(phase="error", error="Cancelled", active=False)
            raise

        except Exception as e:
            try:
                wheel_path.unlink(missing_ok=True)
            except Exception:
                pass
            error = f"{type(e).__name__}: {e}"
            logger.error(f"Install exception: {error}")
            self._progress.update(phase="error", error=error, active=False)
            return {"success": False, "error": error}
