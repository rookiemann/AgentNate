"""
GGUF Tools - Search and download GGUF models from HuggingFace.

Provides tools for the Meta Agent to discover, browse, and download
GGUF language models into the local models directory for llama.cpp loading.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("tools.gguf")


TOOL_DEFINITIONS = [
    {
        "name": "gguf_search",
        "description": "Search HuggingFace for downloadable GGUF language models. Returns repos with download counts and authors. Popular quantizers: bartowski, unsloth, QuantFactory. Search for the base model name (e.g. 'qwen 2.5 7b', 'llama 3.1 8b', 'phi-4') to find quant options.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. 'qwen 2.5 7b instruct', 'phi-4-mini', 'mistral 7b')"
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author/quantizer (e.g. 'bartowski', 'unsloth'). Optional."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10, max: 30)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "gguf_list_files",
        "description": "List all GGUF files in a HuggingFace repo. Shows filename, size in GB, and quant level (Q8_0=best quality, Q4_K_M=good balance, Q3_K_S=smallest). Use after gguf_search to see download options.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_id": {
                    "type": "string",
                    "description": "HuggingFace repo ID from gguf_search results (e.g. 'bartowski/Llama-3.1-8B-Instruct-GGUF')"
                }
            },
            "required": ["repo_id"]
        }
    },
    {
        "name": "gguf_download",
        "description": "Download a GGUF model file from HuggingFace into the local models directory. Returns a download_id for tracking. Typical sizes: 4-8 GB for Q4_K_M of a 7-8B model. Once complete, the model appears in list_available_models(provider='llama_cpp').",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_id": {
                    "type": "string",
                    "description": "HuggingFace repo ID (e.g. 'bartowski/Llama-3.1-8B-Instruct-GGUF')"
                },
                "filename": {
                    "type": "string",
                    "description": "GGUF filename from gguf_list_files (e.g. 'Llama-3.1-8B-Instruct-Q4_K_M.gguf')"
                }
            },
            "required": ["repo_id", "filename"]
        }
    },
    {
        "name": "gguf_download_status",
        "description": "Check progress of a GGUF download. Returns percentage, speed (MB/s), ETA, status. Poll every 15-30 seconds. Status: pending, downloading, completed, failed, cancelled.",
        "parameters": {
            "type": "object",
            "properties": {
                "download_id": {
                    "type": "string",
                    "description": "Download ID from gguf_download result"
                }
            },
            "required": ["download_id"]
        }
    },
    {
        "name": "gguf_cancel_download",
        "description": "Cancel an active GGUF model download.",
        "parameters": {
            "type": "object",
            "properties": {
                "download_id": {
                    "type": "string",
                    "description": "Download ID to cancel"
                }
            },
            "required": ["download_id"]
        }
    },
]


class GGUFTools:
    """Handler class for GGUF search and download tools."""

    def __init__(self, gguf_downloader):
        self.downloader = gguf_downloader

    async def _check(self) -> Dict[str, Any]:
        if not self.downloader:
            return {"success": False, "error": "GGUF downloader not initialized."}
        return None

    async def gguf_search(self, **kwargs) -> Dict[str, Any]:
        """Search HuggingFace for GGUF models."""
        err = await self._check()
        if err:
            return err
        query = kwargs.get("query")
        if not query:
            return {"success": False, "error": "Missing required parameter: query"}
        try:
            result = await self.downloader.search(
                query,
                author=kwargs.get("author"),
                limit=min(kwargs.get("limit", 10), 30),
            )
            if result.get("success") and result.get("results"):
                top = result["results"][0]
                result["hint"] = (
                    f"Found {result['count']} repos. "
                    f"Use gguf_list_files(repo_id='{top['repo_id']}') to see available quant files, "
                    f"then gguf_download to fetch one."
                )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def gguf_list_files(self, **kwargs) -> Dict[str, Any]:
        """List GGUF files in a HuggingFace repo."""
        err = await self._check()
        if err:
            return err
        repo_id = kwargs.get("repo_id")
        if not repo_id:
            return {"success": False, "error": "Missing required parameter: repo_id"}
        try:
            result = await self.downloader.get_model_files(repo_id)
            if result.get("success") and result.get("files"):
                # Find a recommended file (Q4_K_M if available, otherwise first)
                recommended = None
                for f in result["files"]:
                    if "Q4_K_M" in f["filename"].upper():
                        recommended = f["filename"]
                        break
                if not recommended:
                    recommended = result["files"][0]["filename"]
                result["hint"] = (
                    f"Use gguf_download(repo_id='{repo_id}', filename='{recommended}') to download. "
                    f"Q4_K_M = best quality/size balance. Q8_0 = highest quality but larger. "
                    f"Q3_K_S = smallest file size."
                )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def gguf_download(self, **kwargs) -> Dict[str, Any]:
        """Start downloading a GGUF file."""
        err = await self._check()
        if err:
            return err
        repo_id = kwargs.get("repo_id")
        filename = kwargs.get("filename")
        if not repo_id or not filename:
            return {"success": False, "error": "Missing required parameters: repo_id, filename"}

        # Auto-ensure directory exists
        dir_info = self.downloader.get_directory_info()
        if not dir_info["exists"]:
            ensure = await self.downloader.ensure_models_directory()
            if not ensure.get("success"):
                return ensure

        try:
            result = await self.downloader.start_download(repo_id, filename)
            if result.get("success"):
                if result.get("status") == "already_exists":
                    result["hint"] = (
                        f"File already exists. Use load_model(model_name='{filename.replace('.gguf', '')}', "
                        f"provider='llama_cpp', gpu_index=1) to load it."
                    )
                elif result.get("download_id"):
                    result["hint"] = (
                        f"Download started (ID: {result['download_id']}). "
                        f"Use gguf_download_status(download_id='{result['download_id']}') to track progress. "
                        f"Once complete, use load_model(model_name='{filename.replace('.gguf', '')}', "
                        f"provider='llama_cpp', gpu_index=1) to load it."
                    )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def gguf_download_status(self, **kwargs) -> Dict[str, Any]:
        """Check download progress."""
        err = await self._check()
        if err:
            return err
        download_id = kwargs.get("download_id")
        if not download_id:
            return {"success": False, "error": "Missing required parameter: download_id"}
        try:
            result = self.downloader.get_download_status(download_id)
            if result.get("success") and result.get("status") == "completed":
                fn = result.get("filename", "")
                result["hint"] = (
                    f"Download complete! Use load_model(model_name='{fn.replace('.gguf', '')}', "
                    f"provider='llama_cpp', gpu_index=1) to load the model."
                )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def gguf_cancel_download(self, **kwargs) -> Dict[str, Any]:
        """Cancel an active download."""
        err = await self._check()
        if err:
            return err
        download_id = kwargs.get("download_id")
        if not download_id:
            return {"success": False, "error": "Missing required parameter: download_id"}
        try:
            return await self.downloader.cancel_download(download_id)
        except Exception as e:
            return {"success": False, "error": str(e)}
