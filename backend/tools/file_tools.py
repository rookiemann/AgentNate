"""
File Tools - File system operations.
"""

from typing import Dict, Any, List, Optional
import logging
import os
import re
import shutil
import base64
import mimetypes
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("tools.file")


TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Supports text files, JSON, CSV, and images (returned as base64).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to workspace or absolute)"
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding (default 'utf-8')",
                    "default": "utf-8"
                },
                "max_size_mb": {
                    "type": "number",
                    "description": "Maximum file size to read in MB (default 10)",
                    "default": 10
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding (default 'utf-8')",
                    "default": "utf-8"
                },
                "backup": {
                    "type": "boolean",
                    "description": "Create backup if file exists (default false)",
                    "default": False
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to file instead of overwrite (default false)",
                    "default": False
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default current directory)",
                    "default": "."
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter (e.g., '*.py', '*.txt')",
                    "default": "*"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively (default false)",
                    "default": False
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (default false)",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "search_files",
        "description": "Search for files by name pattern recursively.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.py', '**/test_*.py')"
                },
                "path": {
                    "type": "string",
                    "description": "Starting directory (default current)",
                    "default": "."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 100)",
                    "default": 100
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "search_content",
        "description": "Search for text pattern inside files (like grep).",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in",
                    "default": "."
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g., '*.py')",
                    "default": "*"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat pattern as regex (default false)",
                    "default": False
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search (default true)",
                    "default": True
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return (default 50)",
                    "default": 50
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "file_info",
        "description": "Get detailed information about a file or directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "delete_file",
        "description": "Delete a file or empty directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to delete"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Delete directories recursively (default false for safety)",
                    "default": False
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "move_file",
        "description": "Move or rename a file or directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source path"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite if destination exists (default false)",
                    "default": False
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "copy_file",
        "description": "Copy a file or directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source path"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite if destination exists (default false)",
                    "default": False
                }
            },
            "required": ["source", "destination"]
        }
    },
]


class FileTools:
    """Tools for file system operations."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize file tools.

        Args:
            config: Optional configuration dict with keys:
                - base_path: str - Base directory for relative paths
                - allowed_extensions: list - Allowed file extensions (None = all)
                - max_file_size_mb: int - Maximum file size
        """
        self.config = config or {}
        base = self.config.get("base_path")
        self.base_path = Path(base) if base else Path.cwd()
        self.allowed_extensions = self.config.get("allowed_extensions")
        self.max_file_size_mb = self.config.get("max_file_size_mb", 10)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    def _check_extension(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        if self.allowed_extensions is None:
            return True
        return path.suffix.lower() in self.allowed_extensions

    async def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        max_size_mb: float = 10
    ) -> Dict[str, Any]:
        """
        Read a file's contents.

        Args:
            path: File path
            encoding: Text encoding
            max_size_mb: Maximum file size to read

        Returns:
            Dict with file content
        """
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            if not full_path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}

            # Check size
            size_mb = full_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                return {
                    "success": False,
                    "error": f"File too large ({size_mb:.2f} MB > {max_size_mb} MB)"
                }

            # Check if it's a binary/image file
            mime_type, _ = mimetypes.guess_type(str(full_path))
            if mime_type and mime_type.startswith("image/"):
                content = base64.b64encode(full_path.read_bytes()).decode("ascii")
                return {
                    "success": True,
                    "path": str(full_path),
                    "type": "image",
                    "mime_type": mime_type,
                    "size_bytes": full_path.stat().st_size,
                    "content_base64": content
                }

            # Read as text
            try:
                content = full_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                # Try binary
                content = base64.b64encode(full_path.read_bytes()).decode("ascii")
                return {
                    "success": True,
                    "path": str(full_path),
                    "type": "binary",
                    "size_bytes": full_path.stat().st_size,
                    "content_base64": content
                }

            return {
                "success": True,
                "path": str(full_path),
                "type": "text",
                "size_bytes": len(content.encode(encoding)),
                "lines": content.count("\n") + 1,
                "content": content
            }

        except Exception as e:
            logger.error(f"read_file error: {e}")
            return {"success": False, "error": str(e)}

    async def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        backup: bool = False,
        append: bool = False
    ) -> Dict[str, Any]:
        """
        Write content to a file.

        Args:
            path: File path
            content: Content to write
            encoding: Text encoding
            backup: Create backup if file exists
            append: Append instead of overwrite

        Returns:
            Dict with write result
        """
        try:
            full_path = self._resolve_path(path)

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if requested
            if backup and full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + ".bak")
                shutil.copy2(full_path, backup_path)

            # Write content
            mode = "a" if append else "w"
            with open(full_path, mode, encoding=encoding) as f:
                f.write(content)

            return {
                "success": True,
                "path": str(full_path),
                "size_bytes": len(content.encode(encoding)),
                "mode": "appended" if append else "written"
            }

        except Exception as e:
            logger.error(f"write_file error: {e}")
            return {"success": False, "error": str(e)}

    async def list_directory(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Glob pattern
            recursive: List recursively
            include_hidden: Include hidden files

        Returns:
            Dict with directory listing
        """
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            if not full_path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            if recursive:
                matches = list(full_path.rglob(pattern))
            else:
                matches = list(full_path.glob(pattern))

            files = []
            for p in matches:
                if not include_hidden and p.name.startswith("."):
                    continue

                try:
                    stat = p.stat()
                    files.append({
                        "name": p.name,
                        "path": str(p.relative_to(full_path)) if p != full_path else ".",
                        "is_dir": p.is_dir(),
                        "size": stat.st_size if p.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except (OSError, ValueError):
                    continue

            # Sort: directories first, then by name
            files.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

            return {
                "success": True,
                "path": str(full_path),
                "count": len(files),
                "files": files
            }

        except Exception as e:
            logger.error(f"list_directory error: {e}")
            return {"success": False, "error": str(e)}

    async def search_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 100
    ) -> Dict[str, Any]:
        """
        Search for files by name pattern.

        Args:
            pattern: Glob pattern
            path: Starting directory
            max_results: Maximum results

        Returns:
            Dict with matching files
        """
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            matches = []
            for p in full_path.rglob(pattern):
                if len(matches) >= max_results:
                    break
                try:
                    matches.append({
                        "path": str(p.relative_to(full_path)),
                        "absolute_path": str(p),
                        "is_dir": p.is_dir(),
                        "size": p.stat().st_size if p.is_file() else 0
                    })
                except (OSError, ValueError):
                    continue

            return {
                "success": True,
                "pattern": pattern,
                "search_path": str(full_path),
                "count": len(matches),
                "truncated": len(matches) >= max_results,
                "matches": matches
            }

        except Exception as e:
            logger.error(f"search_files error: {e}")
            return {"success": False, "error": str(e)}

    async def search_content(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        regex: bool = False,
        case_sensitive: bool = True,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Search for text pattern inside files.

        Args:
            pattern: Text or regex pattern
            path: Directory or file to search
            file_pattern: Glob pattern for files
            regex: Treat pattern as regex
            case_sensitive: Case sensitive search
            max_results: Maximum matches

        Returns:
            Dict with search results
        """
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            # Build regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            if regex:
                search_re = re.compile(pattern, flags)
            else:
                search_re = re.compile(re.escape(pattern), flags)

            # Get files to search
            if full_path.is_file():
                files = [full_path]
            else:
                files = list(full_path.rglob(file_pattern))

            results = []
            for file_path in files:
                if len(results) >= max_results:
                    break

                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if len(results) >= max_results:
                            break

                        if search_re.search(line):
                            results.append({
                                "file": str(file_path.relative_to(full_path)) if full_path.is_dir() else file_path.name,
                                "line_number": i,
                                "line": line.strip()[:200]  # Truncate long lines
                            })

                except (OSError, UnicodeDecodeError):
                    continue

            return {
                "success": True,
                "pattern": pattern,
                "regex": regex,
                "count": len(results),
                "truncated": len(results) >= max_results,
                "results": results
            }

        except re.error as e:
            return {"success": False, "error": f"Invalid regex pattern: {e}"}
        except Exception as e:
            logger.error(f"search_content error: {e}")
            return {"success": False, "error": str(e)}

    async def file_info(self, path: str) -> Dict[str, Any]:
        """
        Get detailed file information.

        Args:
            path: File or directory path

        Returns:
            Dict with file info
        """
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            stat = full_path.stat()
            mime_type, _ = mimetypes.guess_type(str(full_path))

            info = {
                "success": True,
                "path": str(full_path),
                "name": full_path.name,
                "is_file": full_path.is_file(),
                "is_dir": full_path.is_dir(),
                "size_bytes": stat.st_size,
                "size_human": self._human_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": full_path.suffix,
                "mime_type": mime_type
            }

            # Add line count for text files
            if full_path.is_file() and mime_type and mime_type.startswith("text/"):
                try:
                    content = full_path.read_text()
                    info["line_count"] = content.count("\n") + 1
                except Exception:
                    pass

            return info

        except Exception as e:
            logger.error(f"file_info error: {e}")
            return {"success": False, "error": str(e)}

    async def delete_file(
        self,
        path: str,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a file or directory.

        Args:
            path: Path to delete
            recursive: Delete directories recursively

        Returns:
            Dict with delete result
        """
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            if full_path.is_file():
                full_path.unlink()
                return {
                    "success": True,
                    "deleted": str(full_path),
                    "type": "file"
                }

            if full_path.is_dir():
                if recursive:
                    shutil.rmtree(full_path)
                    return {
                        "success": True,
                        "deleted": str(full_path),
                        "type": "directory (recursive)"
                    }
                else:
                    # Try to remove empty directory
                    try:
                        full_path.rmdir()
                        return {
                            "success": True,
                            "deleted": str(full_path),
                            "type": "directory"
                        }
                    except OSError:
                        return {
                            "success": False,
                            "error": "Directory not empty. Use recursive=true to delete."
                        }

        except Exception as e:
            logger.error(f"delete_file error: {e}")
            return {"success": False, "error": str(e)}

    async def move_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Move or rename a file/directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite if destination exists

        Returns:
            Dict with move result
        """
        try:
            src_path = self._resolve_path(source)
            dst_path = self._resolve_path(destination)

            if not src_path.exists():
                return {"success": False, "error": f"Source not found: {source}"}

            if dst_path.exists() and not overwrite:
                return {"success": False, "error": f"Destination exists: {destination}"}

            # Create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(src_path), str(dst_path))

            return {
                "success": True,
                "source": str(src_path),
                "destination": str(dst_path)
            }

        except Exception as e:
            logger.error(f"move_file error: {e}")
            return {"success": False, "error": str(e)}

    async def copy_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Copy a file or directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite if destination exists

        Returns:
            Dict with copy result
        """
        try:
            src_path = self._resolve_path(source)
            dst_path = self._resolve_path(destination)

            if not src_path.exists():
                return {"success": False, "error": f"Source not found: {source}"}

            if dst_path.exists() and not overwrite:
                return {"success": False, "error": f"Destination exists: {destination}"}

            # Create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            else:
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)

            return {
                "success": True,
                "source": str(src_path),
                "destination": str(dst_path),
                "type": "file" if src_path.is_file() else "directory"
            }

        except Exception as e:
            logger.error(f"copy_file error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
