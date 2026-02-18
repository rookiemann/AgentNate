"""
GGUF Model Search & Download Manager

Searches HuggingFace for GGUF models and downloads them directly
via HTTP into the configured models directory. No huggingface_hub dependency.
"""

import os
import uuid
import time
import shutil
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

import httpx

logger = logging.getLogger("GGUFDownloader")

HF_API_BASE = "https://huggingface.co/api/models"
HF_DOWNLOAD_BASE = "https://huggingface.co"

# Popular GGUF quantizers
KNOWN_QUANTIZERS = ["bartowski", "unsloth", "QuantFactory", "TheBloke", "mradermacher"]

# Quant quality ordering (best first)
QUANT_RANKING = [
    "Q8_0", "Q6_K_L", "Q6_K", "Q5_K_L", "Q5_K_M", "Q5_K_S",
    "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_0",
    "IQ4_XS", "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_M", "IQ3_S",
    "Q2_K_L", "Q2_K", "Q2_K_S", "IQ2_M", "IQ2_S",
]

# Default portable path for new users
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gguf")

MAX_CONCURRENT_DOWNLOADS = 2
CHUNK_SIZE = 1024 * 1024  # 1 MB
SPEED_WINDOW = 10  # seconds for rolling average


@dataclass
class DownloadJob:
    """Tracks a single GGUF file download."""
    id: str
    repo_id: str
    filename: str
    url: str
    destination: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "pending"  # pending, downloading, completed, failed, cancelled
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        pct = 0.0
        if self.total_bytes > 0:
            pct = round(self.downloaded_bytes / self.total_bytes * 100, 1)
        speed_mb = round(self.speed_bps / (1024 * 1024), 2)
        eta_str = ""
        if self.eta_seconds > 0:
            mins, secs = divmod(int(self.eta_seconds), 60)
            eta_str = f"{mins}:{secs:02d}"
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "destination": self.destination,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "progress_pct": pct,
            "speed_mb_s": speed_mb,
            "eta": eta_str,
            "status": self.status,
            "error": self.error,
        }


def _extract_quant(filename: str) -> str:
    """Extract quantization level from a GGUF filename."""
    name = filename.upper().replace(".GGUF", "")
    for q in QUANT_RANKING:
        if q.upper() in name:
            return q
    return ""


def _quant_sort_key(filename: str) -> int:
    """Sort key for quant quality (lower = better)."""
    q = _extract_quant(filename)
    try:
        return QUANT_RANKING.index(q)
    except ValueError:
        return 999


class GGUFDownloader:
    """Manages GGUF model search and download from HuggingFace."""

    def __init__(self, settings_manager):
        self.settings = settings_manager
        self._downloads: Dict[str, DownloadJob] = {}
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
                follow_redirects=True,
                headers={"User-Agent": "AgentNate/1.0"},
            )
        return self._client

    def _get_models_directory(self) -> Optional[str]:
        """Get the configured models directory."""
        d = self.settings.get("providers.llama_cpp.models_directory")
        if d and os.path.isdir(d):
            return d
        return None

    # ---- Search ----

    async def search(
        self,
        query: str,
        author: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search HuggingFace for GGUF model repos."""
        client = self._get_client()

        params = {
            "search": query,
            "library": "gguf",
            "sort": sort,
            "direction": "-1",
            "limit": min(limit, 50),
        }
        if author:
            params["author"] = author

        try:
            resp = await client.get(HF_API_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"success": False, "error": f"HuggingFace API error: {e}"}

        results = []
        for model in data:
            results.append({
                "repo_id": model.get("id", ""),
                "author": model.get("author", ""),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "last_modified": model.get("lastModified", ""),
                "tags": model.get("tags", [])[:10],
                "pipeline_tag": model.get("pipeline_tag", ""),
            })

        return {"success": True, "results": results, "count": len(results)}

    async def get_model_files(self, repo_id: str) -> Dict[str, Any]:
        """List GGUF files in a HuggingFace repo with sizes."""
        client = self._get_client()

        # Use the tree API which works without auth for public repos
        try:
            resp = await client.get(
                f"https://huggingface.co/api/models/{repo_id}/tree/main",
            )
            resp.raise_for_status()
            tree = resp.json()
        except Exception as e:
            return {"success": False, "error": f"HuggingFace API error: {e}"}

        files = []
        for f in tree:
            name = f.get("path", "")
            if not name.lower().endswith(".gguf"):
                continue
            if "mmproj" in name.lower():
                continue
            size = f.get("size", 0)
            quant = _extract_quant(name)
            size_gb = round(size / (1024 ** 3), 2) if size else 0
            files.append({
                "filename": name,
                "size_bytes": size,
                "size_gb": size_gb,
                "quant": quant,
            })

        files.sort(key=lambda x: _quant_sort_key(x["filename"]))

        return {
            "success": True,
            "repo_id": repo_id,
            "files": files,
            "count": len(files),
        }

    # ---- Download ----

    async def start_download(
        self,
        repo_id: str,
        filename: str,
    ) -> Dict[str, Any]:
        """Start downloading a GGUF file."""
        models_dir = self._get_models_directory()
        if not models_dir:
            return {
                "success": False,
                "error": "Models directory not configured or does not exist. "
                         "Use ensure_models_directory() first or set providers.llama_cpp.models_directory in settings.",
            }

        # Check concurrent limit
        active = sum(1 for j in self._downloads.values() if j.status == "downloading")
        if active >= MAX_CONCURRENT_DOWNLOADS:
            return {
                "success": False,
                "error": f"Maximum {MAX_CONCURRENT_DOWNLOADS} concurrent downloads. Wait for one to finish or cancel it.",
            }

        url = f"{HF_DOWNLOAD_BASE}/{repo_id}/resolve/main/{filename}"
        destination = os.path.join(models_dir, filename)

        # Check if file already exists and is complete
        if os.path.isfile(destination):
            existing_size = os.path.getsize(destination)
            if existing_size > 0:
                return {
                    "success": True,
                    "status": "already_exists",
                    "destination": destination,
                    "size_bytes": existing_size,
                    "message": f"File already exists at {destination} ({round(existing_size / 1024**3, 2)} GB)",
                }

        # Check disk space by doing a HEAD request first
        try:
            client = self._get_client()
            head_resp = await client.head(url, follow_redirects=True)
            head_resp.raise_for_status()
            total_bytes = int(head_resp.headers.get("content-length", 0))
        except Exception as e:
            return {"success": False, "error": f"Cannot reach download URL: {e}"}

        if total_bytes > 0:
            try:
                free_space = shutil.disk_usage(models_dir).free
                if free_space < total_bytes * 1.1:  # 10% margin
                    needed_gb = round(total_bytes / 1024**3, 2)
                    free_gb = round(free_space / 1024**3, 2)
                    return {
                        "success": False,
                        "error": f"Insufficient disk space. Need {needed_gb} GB but only {free_gb} GB free.",
                    }
            except Exception:
                pass  # disk_usage may fail on some paths

        job_id = str(uuid.uuid4())[:8]
        job = DownloadJob(
            id=job_id,
            repo_id=repo_id,
            filename=filename,
            url=url,
            destination=destination,
            total_bytes=total_bytes,
        )
        self._downloads[job_id] = job

        job._task = asyncio.create_task(self._download_file(job))
        logger.info(f"Started download {job_id}: {repo_id}/{filename} -> {destination} ({round(total_bytes/1024**3, 2)} GB)")

        return {
            "success": True,
            "download_id": job_id,
            "filename": filename,
            "destination": destination,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1024**3, 2),
        }

    async def _download_file(self, job: DownloadJob):
        """Async streaming download with resume support."""
        job.status = "downloading"
        job.started_at = time.time()

        part_path = job.destination + ".part"
        existing_bytes = 0
        headers = {}

        # Resume if partial file exists
        if os.path.isfile(part_path):
            existing_bytes = os.path.getsize(part_path)
            if existing_bytes > 0 and job.total_bytes > 0 and existing_bytes < job.total_bytes:
                headers["Range"] = f"bytes={existing_bytes}-"
                job.downloaded_bytes = existing_bytes
                logger.info(f"Resuming download {job.id} from {existing_bytes} bytes")

        # Create a separate client for downloads with long timeouts
        download_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0),
            follow_redirects=True,
            headers={"User-Agent": "AgentNate/1.0"},
        )

        speed_samples: List[tuple] = []  # (timestamp, bytes_at_that_time)

        try:
            async with download_client.stream("GET", job.url, headers=headers) as response:
                if response.status_code == 416:
                    # Range not satisfiable — file already complete
                    if os.path.isfile(part_path):
                        os.rename(part_path, job.destination)
                    job.status = "completed"
                    job.completed_at = time.time()
                    return

                response.raise_for_status()

                # Update total from content-range if resuming
                if "content-range" in response.headers:
                    # content-range: bytes 12345-67890/67891
                    cr = response.headers["content-range"]
                    if "/" in cr:
                        total_str = cr.split("/")[-1]
                        if total_str.isdigit():
                            job.total_bytes = int(total_str)
                elif not job.total_bytes:
                    cl = response.headers.get("content-length", "0")
                    job.total_bytes = existing_bytes + int(cl)

                mode = "ab" if existing_bytes > 0 else "wb"
                with open(part_path, mode) as f:
                    async for chunk in response.aiter_bytes(CHUNK_SIZE):
                        f.write(chunk)
                        job.downloaded_bytes += len(chunk)

                        # Update speed
                        now = time.time()
                        speed_samples.append((now, job.downloaded_bytes))
                        # Keep only samples within window
                        cutoff = now - SPEED_WINDOW
                        speed_samples = [(t, b) for t, b in speed_samples if t >= cutoff]

                        if len(speed_samples) >= 2:
                            dt = speed_samples[-1][0] - speed_samples[0][0]
                            db = speed_samples[-1][1] - speed_samples[0][1]
                            if dt > 0:
                                job.speed_bps = db / dt
                                remaining = job.total_bytes - job.downloaded_bytes
                                if job.speed_bps > 0:
                                    job.eta_seconds = remaining / job.speed_bps

            # Download complete — rename .part to final
            if os.path.isfile(part_path):
                # Remove existing file if any
                if os.path.isfile(job.destination):
                    os.remove(job.destination)
                os.rename(part_path, job.destination)

            job.status = "completed"
            job.completed_at = time.time()
            elapsed = job.completed_at - job.started_at
            logger.info(f"Download {job.id} completed: {job.filename} ({round(job.total_bytes/1024**3, 2)} GB in {round(elapsed)}s)")

        except asyncio.CancelledError:
            job.status = "cancelled"
            logger.info(f"Download {job.id} cancelled")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Download {job.id} failed: {e}")
        finally:
            await download_client.aclose()

    # ---- Status ----

    def get_download_status(self, download_id: str) -> Dict[str, Any]:
        """Get status of a download."""
        job = self._downloads.get(download_id)
        if not job:
            return {"success": False, "error": f"Download not found: {download_id}"}
        return {"success": True, **job.to_dict()}

    def list_downloads(self) -> Dict[str, Any]:
        """List all active and recent downloads."""
        # Active first, then recent completed (last 10)
        active = [j.to_dict() for j in self._downloads.values() if j.status in ("pending", "downloading")]
        completed = [j.to_dict() for j in self._downloads.values() if j.status in ("completed", "failed", "cancelled")]
        completed.sort(key=lambda x: x.get("id", ""), reverse=True)
        completed = completed[:10]
        return {"success": True, "active": active, "completed": completed}

    async def cancel_download(self, download_id: str) -> Dict[str, Any]:
        """Cancel an active download."""
        job = self._downloads.get(download_id)
        if not job:
            return {"success": False, "error": f"Download not found: {download_id}"}
        if job.status != "downloading":
            return {"success": False, "error": f"Download is not active (status: {job.status})"}
        if job._task:
            job._task.cancel()
        job.status = "cancelled"
        return {"success": True, "message": f"Download {download_id} cancelled"}

    # ---- Directory Management ----

    async def ensure_models_directory(self) -> Dict[str, Any]:
        """Ensure a models directory exists, creating the default if needed."""
        current = self.settings.get("providers.llama_cpp.models_directory")

        # If configured and exists, nothing to do
        if current and os.path.isdir(current):
            return {"success": True, "directory": current, "created": False}

        # Create default portable directory
        target = DEFAULT_MODELS_DIR
        try:
            os.makedirs(target, exist_ok=True)
            self.settings.set("providers.llama_cpp.models_directory", target)
            logger.info(f"Created default models directory: {target}")
            return {"success": True, "directory": target, "created": True}
        except Exception as e:
            return {"success": False, "error": f"Failed to create directory: {e}"}

    def get_directory_info(self) -> Dict[str, Any]:
        """Get current models directory info."""
        current = self.settings.get("providers.llama_cpp.models_directory")
        exists = bool(current and os.path.isdir(current))
        model_count = 0
        if exists:
            try:
                for root, _, files in os.walk(current):
                    for f in files:
                        if f.lower().endswith(".gguf"):
                            model_count += 1
            except Exception:
                pass
        return {
            "directory": current or "",
            "exists": exists,
            "model_count": model_count,
            "default_directory": DEFAULT_MODELS_DIR,
        }

    # ---- Lifecycle ----

    async def shutdown(self):
        """Cancel all downloads and close client."""
        for job in self._downloads.values():
            if job._task and not job._task.done():
                job._task.cancel()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        logger.info("GGUF downloader shut down")
