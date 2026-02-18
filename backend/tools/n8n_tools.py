"""
n8n Tools - Manage n8n workflow automation instances.
"""

from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger("tools.n8n")
AGENTNATE_BASE = os.getenv("AGENTNATE_BASE_URL", "http://127.0.0.1:8000")


TOOL_DEFINITIONS = [
    {
        "name": "spawn_n8n",
        "description": "Start a new n8n workflow automation instance",
        "parameters": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "Port to run n8n on (default: auto-assign starting from 5678)"
                }
            },
            "required": []
        }
    },
    {
        "name": "stop_n8n",
        "description": "Stop an n8n instance",
        "parameters": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "Port of the n8n instance to stop"
                }
            },
            "required": ["port"]
        }
    },
    {
        "name": "list_n8n_instances",
        "description": "List all running n8n instances",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_n8n_status",
        "description": "Get status of an n8n instance",
        "parameters": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "Port of the n8n instance"
                }
            },
            "required": []
        }
    }
]


class N8nTools:
    """Tools for n8n management."""

    def __init__(self, n8n_manager):
        self.n8n_manager = n8n_manager

    async def spawn_n8n(self, port: Optional[int] = None) -> Dict[str, Any]:
        """Spawn a new n8n instance."""
        import os

        # Pre-flight checks
        checks = {
            "node_path": self.n8n_manager.node_path,
            "node_exists": os.path.exists(self.n8n_manager.node_path) if self.n8n_manager.node_path else False,
            "n8n_path": self.n8n_manager.n8n_path,
            "n8n_exists": os.path.exists(self.n8n_manager.n8n_path) if self.n8n_manager.n8n_path else False,
        }

        if not checks["node_exists"]:
            return {
                "success": False,
                "error": f"Node.js not found at: {checks['node_path']}",
                "checks": checks
            }

        if not checks["n8n_exists"]:
            return {
                "success": False,
                "error": f"n8n not found at: {checks['n8n_path']}",
                "checks": checks
            }

        try:
            instance = await self.n8n_manager.spawn(port)

            return {
                "success": True,
                "port": instance.port,
                "url": f"{AGENTNATE_BASE}/api/n8n/{instance.port}/proxy/",
                "message": f"n8n started on port {instance.port}",
                "pid": instance.pid
            }

        except Exception as e:
            logger.error(f"spawn_n8n error: {e}")
            return {
                "success": False,
                "error": str(e),
                "checks": checks
            }

    async def stop_n8n(self, port: int) -> Dict[str, Any]:
        """Stop an n8n instance."""
        try:
            success = await self.n8n_manager.stop(port)

            if success:
                return {
                    "success": True,
                    "message": f"n8n on port {port} stopped"
                }
            else:
                return {
                    "success": False,
                    "error": f"n8n on port {port} not found or already stopped"
                }

        except Exception as e:
            logger.error(f"stop_n8n error: {e}")
            return {"success": False, "error": str(e)}

    async def list_n8n_instances(self) -> Dict[str, Any]:
        """List running n8n instances."""
        try:
            instances = []
            for port, inst in self.n8n_manager.instances.items():
                is_running = inst.is_running if inst else False
                instances.append({
                    "port": port,
                    "running": is_running,
                    "url": f"{AGENTNATE_BASE}/api/n8n/{port}/proxy/"
                })

            return {
                "success": True,
                "count": len(instances),
                "instances": instances
            }

        except Exception as e:
            logger.error(f"list_n8n_instances error: {e}")
            return {"success": False, "error": str(e)}

    async def get_n8n_status(self, port: Optional[int] = None) -> Dict[str, Any]:
        """Get n8n instance status."""
        try:
            # Auto-resolve if port omitted: prefer running main, then any running instance,
            # then first known instance.
            if port is None:
                if hasattr(self.n8n_manager, "main") and self.n8n_manager.main:
                    main_running = getattr(self.n8n_manager.main, "is_running", False)
                    if callable(main_running):
                        main_running = main_running()
                    if main_running and getattr(self.n8n_manager.main, "port", None):
                        port = self.n8n_manager.main.port
                if port is None and getattr(self.n8n_manager, "instances", None):
                    running_ports = []
                    for p, inst in self.n8n_manager.instances.items():
                        if not inst:
                            continue
                        running = getattr(inst, "is_running", False)
                        if callable(running):
                            running = running()
                        if running:
                            running_ports.append(p)
                    if running_ports:
                        port = running_ports[0]
                    elif self.n8n_manager.instances:
                        port = next(iter(self.n8n_manager.instances))

            if port is None:
                return {"success": False, "error": "No n8n instances found."}

            if port not in self.n8n_manager.instances:
                return {
                    "success": False,
                    "error": f"n8n on port {port} not found"
                }

            inst = self.n8n_manager.instances[port]
            is_running = inst.is_running if inst else False

            # Try to get workflow count
            workflow_count = None
            if is_running:
                try:
                    import aiohttp
                    from backend.routes.n8n import _get_or_create_auth

                    auth_cookie = await _get_or_create_auth(port)
                    headers = {}
                    if auth_cookie:
                        headers["Cookie"] = f"n8n-auth={auth_cookie}"

                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://127.0.0.1:{port}/rest/workflows",
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                workflow_count = len(data.get("data", []))
                except Exception as e:
                    logger.debug(f"Failed to fetch workflow count for port {port}: {e}")

            return {
                "success": True,
                "port": port,
                "running": is_running,
                "workflow_count": workflow_count,
                "url": f"{AGENTNATE_BASE}/api/n8n/{port}/proxy/"
            }

        except Exception as e:
            logger.error(f"get_n8n_status error: {e}")
            return {"success": False, "error": str(e)}
