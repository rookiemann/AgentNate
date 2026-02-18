"""
Debug Logger Middleware for AgentNate

Intercepts all n8n and workflow API requests, logging request/response details
to .n8n-instances/debug.log for post-session analysis.
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Paths we want to capture (prefix match)
_LOGGED_PREFIXES = ("/api/n8n/", "/api/workflows/", "/api/debug/", "/api/comfyui/", "/api/routing/")

# Body size limits to prevent huge logs
_MAX_REQUEST_BODY = 1000
_MAX_RESPONSE_BODY = 2000

# Module-level logger — configured by init_debug_logger()
debug_logger: logging.Logger = logging.getLogger("debug_file")
debug_logger.propagate = False  # Don't spam the console

_log_file_path: str = ""


def init_debug_logger(log_dir: str):
    """
    Set up the file-based debug logger. Call once at startup.
    Truncates the log file for a fresh session.
    """
    global _log_file_path
    os.makedirs(log_dir, exist_ok=True)
    _log_file_path = os.path.join(log_dir, "debug.log")

    # Truncate for fresh session
    with open(_log_file_path, "w") as f:
        f.write("")

    # File handler — 5MB max, 1 backup
    handler = RotatingFileHandler(
        _log_file_path, maxBytes=5 * 1024 * 1024, backupCount=1, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))
    handler.setLevel(logging.DEBUG)

    debug_logger.handlers.clear()
    debug_logger.addHandler(handler)
    debug_logger.setLevel(logging.DEBUG)

    debug_logger.info("[STARTUP] Debug logger initialized")


def get_log_file_path() -> str:
    return _log_file_path


def clear_log():
    """Truncate the debug log file."""
    if _log_file_path and os.path.exists(_log_file_path):
        # Close and re-open handlers to avoid file lock issues on Windows
        for h in debug_logger.handlers:
            h.close()
        with open(_log_file_path, "w") as f:
            f.write("")
        # Re-init handler
        handler = RotatingFileHandler(
            _log_file_path, maxBytes=5 * 1024 * 1024, backupCount=1, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))
        handler.setLevel(logging.DEBUG)
        debug_logger.handlers.clear()
        debug_logger.addHandler(handler)
        debug_logger.info("[STARTUP] Debug log cleared")


class DebugLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request/response for n8n and workflow routes.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only log specific route prefixes
        if not any(path.startswith(p) for p in _LOGGED_PREFIXES):
            return await call_next(request)

        # Skip the debug/log endpoint itself to avoid recursion
        if path == "/api/debug/log":
            return await call_next(request)

        method = request.method
        query = str(request.url.query) if request.url.query else ""
        query_str = f"?{query}" if query else ""

        # Read request body for mutating methods
        req_body = ""
        if method in ("POST", "PUT", "PATCH", "DELETE"):
            try:
                body_bytes = await request.body()
                req_body = body_bytes.decode("utf-8", errors="replace")
                if len(req_body) > _MAX_REQUEST_BODY:
                    req_body = req_body[:_MAX_REQUEST_BODY] + "...(truncated)"
            except Exception:
                req_body = "<unreadable>"

        body_str = f"  body={req_body}" if req_body else ""
        debug_logger.info(f"[REQ] >>> {method} {path}{query_str}{body_str}")

        # Call the actual endpoint
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            debug_logger.error(f"[ERROR] {method} {path}  EXCEPTION  {elapsed:.0f}ms  {type(exc).__name__}: {exc}")
            raise

        elapsed = (time.monotonic() - start) * 1000
        status = response.status_code

        # Read response body (skip binary media to avoid log flooding)
        resp_body = ""
        content_type = (response.headers.get("content-type") or response.media_type or "").lower()
        _is_binary = any(t in content_type for t in ("image/", "video/", "audio/", "octet-stream"))
        try:
            body_parts = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    body_parts.append(chunk)
                else:
                    body_parts.append(chunk.encode("utf-8"))
            raw = b"".join(body_parts)
            if _is_binary:
                resp_body = f"<binary {content_type} {len(raw)} bytes>"
            else:
                resp_body = raw.decode("utf-8", errors="replace")
                if len(resp_body) > _MAX_RESPONSE_BODY:
                    resp_body = resp_body[:_MAX_RESPONSE_BODY] + "...(truncated)"

            # Reconstruct the response since we consumed the body iterator
            response = Response(
                content=raw,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        except Exception:
            resp_body = "<unreadable>"

        level = "RES" if status < 400 else "ERROR"
        debug_logger.info(f"[{level}] <<< {method} {path}  {status}  {elapsed:.0f}ms  body={resp_body}")

        return response
