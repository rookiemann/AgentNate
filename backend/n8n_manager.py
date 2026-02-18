"""
n8n Process Manager

Manages multiple n8n instances on different ports for parallel workflow execution.

Includes:
- N8nManager: Original manager with shared database (for backwards compatibility)
- N8nQueueManager: New queue-based manager with isolated worker databases
"""

import os
import json
import subprocess
import asyncio
import logging
import httpx
import socket
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("n8n_manager")

# Debug file logger (initialized by server.py → init_debug_logger)
from backend.middleware.debug_logger import debug_logger


# =============================================================================
# Process Registry - Persistent PID tracking for orphan cleanup
# =============================================================================

def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if os.name == 'nt':
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _kill_pid(pid: int) -> bool:
    """Kill a process by PID. On Windows uses /T to kill process tree."""
    try:
        if os.name == 'nt':
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return result.returncode == 0
        else:
            import signal as sig
            os.kill(pid, sig.SIGTERM)
            # Give it a moment, then force kill
            import time
            time.sleep(1)
            try:
                os.kill(pid, sig.SIGKILL)
            except (OSError, ProcessLookupError):
                pass  # Already dead
            return True
    except Exception as e:
        logger.debug(f"Failed to kill PID {pid}: {e}")
        return False


def _kill_n8n_port_orphans(port_range: tuple = (5678, 5778)) -> int:
    """
    Kill orphaned node.exe processes listening on n8n ports.

    This catches orphans that predate the ProcessRegistry or weren't tracked.
    Safe because these ports are exclusively used by AgentNate's n8n instances,
    and this runs BEFORE any new instances are spawned.
    """
    killed = 0

    if os.name == 'nt':
        try:
            # Get all listening TCP connections with PIDs
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                return 0

            pids_on_ports = set()
            for line in result.stdout.split('\n'):
                line = line.strip()
                if 'LISTENING' not in line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                # Parse local address (e.g., "127.0.0.1:5679" or "0.0.0.0:5678")
                addr = parts[1]
                if ':' in addr:
                    try:
                        port = int(addr.rsplit(':', 1)[1])
                        if port_range[0] <= port <= port_range[1]:
                            pid = int(parts[-1])
                            if pid > 0:
                                pids_on_ports.add((pid, port))
                    except (ValueError, IndexError):
                        pass

            # Only kill node.exe processes
            for pid, port in pids_on_ports:
                try:
                    check = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                        capture_output=True, text=True, timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if 'node.exe' in check.stdout.lower():
                        logger.info(f"Killing orphaned node.exe PID {pid} on port {port}")
                        if _kill_pid(pid):
                            killed += 1
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Port orphan scan error: {e}")

    else:
        # Unix: single lsof call for entire port range
        try:
            result = subprocess.run(
                ["lsof", f"-iTCP:{port_range[0]}-{port_range[1]}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                for pid_str in result.stdout.strip().split('\n'):
                    try:
                        pid = int(pid_str.strip())
                        # Verify it's a node process
                        ps = subprocess.run(
                            ["ps", "-p", str(pid), "-o", "comm="],
                            capture_output=True, text=True, timeout=5,
                        )
                        if 'node' in ps.stdout.lower():
                            logger.info(f"Killing orphaned node PID {pid}")
                            if _kill_pid(pid):
                                killed += 1
                    except (ValueError, Exception):
                        pass
        except FileNotFoundError:
            pass  # lsof not installed
        except Exception as e:
            logger.debug(f"Port orphan scan error: {e}")

    if killed > 0:
        logger.info(f"Killed {killed} orphaned node process(es) on n8n ports")

    return killed


class ProcessRegistry:
    """
    Persistent PID registry for tracking spawned n8n processes.

    Writes to {data_base_dir}/process_registry.json so that on crash/restart
    we can find and kill orphaned processes from the previous run.
    """

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._data = {"server_pid": None, "processes": {}}
        self._load()

    def _load(self):
        """Load registry from disk."""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    self._data = json.load(f)
                    # Ensure structure
                    if "processes" not in self._data:
                        self._data["processes"] = {}
                    if "server_pid" not in self._data:
                        self._data["server_pid"] = None
        except Exception as e:
            logger.warning(f"Failed to load process registry: {e}")
            self._data = {"server_pid": None, "processes": {}}

    def _save(self):
        """Save registry to disk."""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save process registry: {e}")

    def set_server_pid(self, pid: Optional[int] = None):
        """Record the current server's PID."""
        self._data["server_pid"] = pid or os.getpid()
        self._save()

    def register(self, port: int, pid: int, proc_type: str, **extra):
        """Register a spawned process."""
        key = str(port)
        existing = self._data["processes"].get(key)
        if existing and existing.get("type") == "main" and proc_type == "legacy":
            # Don't let a legacy registration overwrite a main admin registration
            existing_pid = existing.get("pid")
            if existing_pid and _is_pid_alive(existing_pid):
                logger.warning(
                    f"Refusing to overwrite main PID {existing_pid} on port {port} "
                    f"with legacy PID {pid}"
                )
                debug_logger.info(
                    f"[REGISTRY] REFUSED legacy PID={pid} port={port} "
                    f"(main PID={existing_pid} still alive)"
                )
                return

        self._data["processes"][key] = {
            "pid": pid,
            "type": proc_type,
            "port": port,
            "started": datetime.now().isoformat(),
            **extra,
        }
        self._save()
        logger.debug(f"Registered {proc_type} process PID {pid} on port {port}")
        debug_logger.info(f"[REGISTRY] Registered {proc_type} PID={pid} port={port} {extra}")

    def unregister(self, port: int):
        """Remove a stopped process from the registry."""
        key = str(port)
        if key in self._data["processes"]:
            del self._data["processes"][key]
            self._save()
            logger.debug(f"Unregistered process on port {port}")
            debug_logger.info(f"[REGISTRY] Unregistered port={port}")

    def kill_orphans(self) -> int:
        """
        Kill processes from a previous server run.

        Only kills if the old server PID is dead (meaning it crashed or was killed).
        Returns the number of processes killed.
        """
        old_server_pid = self._data.get("server_pid")
        processes = self._data.get("processes", {})

        if not processes:
            return 0

        # If old server is still alive, don't touch its processes
        if old_server_pid and _is_pid_alive(old_server_pid):
            logger.info(f"Previous server (PID {old_server_pid}) still alive, skipping orphan kill")
            return 0

        killed = 0
        for port_str, info in list(processes.items()):
            pid = info.get("pid")
            proc_type = info.get("type", "unknown")
            if pid and _is_pid_alive(pid):
                logger.info(f"Killing orphaned {proc_type} process PID {pid} (port {info.get('port', '?')})")
                if _kill_pid(pid):
                    killed += 1
                else:
                    logger.warning(f"Failed to kill orphaned PID {pid}")

        # Clear registry after cleanup
        self._data["processes"] = {}
        self._data["server_pid"] = None
        self._save()

        if killed > 0:
            logger.info(f"Killed {killed} orphaned process(es) from previous run")
        return killed

    def kill_all_registered(self) -> int:
        """
        Kill all registered processes. Used during intentional shutdown.

        Returns the number of processes killed.
        """
        killed = 0
        for port_str, info in list(self._data.get("processes", {}).items()):
            pid = info.get("pid")
            proc_type = info.get("type", "unknown")
            if pid and _is_pid_alive(pid):
                logger.info(f"Shutdown: killing {proc_type} PID {pid} (port {info.get('port', '?')})")
                if _kill_pid(pid):
                    killed += 1

        # Clear registry
        self._data["processes"] = {}
        self._save()
        return killed

    def clear(self):
        """Clear the entire registry."""
        self._data = {"server_pid": None, "processes": {}}
        self._save()


@dataclass
class N8nInstance:
    """Represents a running n8n instance."""
    port: int
    process: Optional[subprocess.Popen]
    data_folder: str
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def is_running(self) -> bool:
        if self.process is None:
            # Adopted instance — check if port is still responding
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect(('127.0.0.1', self.port))
                    return True
            except (OSError, ConnectionRefusedError):
                return False
        return self.process.poll() is None

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "url": self.url,
            "pid": self.pid,
            "data_folder": self.data_folder,
            "started_at": self.started_at.isoformat(),
            "is_running": self.is_running,
        }


class N8nManager:
    """
    Manages multiple n8n instances on different ports.

    Each instance gets its own data folder for isolation.
    """

    def __init__(
        self,
        base_port: int = 5678,
        n8n_path: str = "n8n",
        data_base_dir: Optional[str] = None,
        node_path: Optional[str] = None,
        registry: Optional['ProcessRegistry'] = None,
    ):
        self.base_port = base_port
        self.n8n_path = n8n_path
        self.node_path = node_path
        self.registry = registry
        self.data_base_dir = data_base_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".n8n-instances"
        )
        self.instances: Dict[int, N8nInstance] = {}
        self._lock = asyncio.Lock()

        # Ensure data directory exists
        os.makedirs(self.data_base_dir, exist_ok=True)

    async def spawn(self, port: Optional[int] = None) -> N8nInstance:
        """
        Spawn a new n8n instance.

        Args:
            port: Specific port to use, or None for auto-assign

        Returns:
            N8nInstance with details about the spawned instance
        """
        async with self._lock:
            if port is None:
                port = self._next_available_port()

            if port in self.instances:
                existing = self.instances[port]
                if existing.is_running:
                    logger.warning(f"n8n already running on port {port}")
                    return existing
                else:
                    # Clean up dead instance
                    del self.instances[port]

            # Use main data folder (same as QueueManager main admin)
            # so workflows deployed by agent tools appear in the deployed tab
            data_folder = os.path.join(self.data_base_dir, "main")
            os.makedirs(data_folder, exist_ok=True)

            # Check if port is already in use (e.g. by QueueManager's main admin)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect(('127.0.0.1', port))
                    # Port is occupied — check if it's n8n and adopt it
                    logger.info(f"Port {port} already in use, checking if it's n8n...")
                    try:
                        async with httpx.AsyncClient(timeout=5) as client:
                            resp = await client.get(f"http://127.0.0.1:{port}/")
                            if resp.status_code in (200, 301, 302):
                                logger.info(f"Adopting existing n8n on port {port}")
                                instance = N8nInstance(
                                    port=port,
                                    process=None,
                                    data_folder=data_folder,
                                )
                                self.instances[port] = instance
                                return instance
                    except (httpx.ConnectError, httpx.ReadTimeout):
                        pass
                    raise RuntimeError(f"Port {port} is already in use by another process")
            except (OSError, ConnectionRefusedError):
                pass  # Port is free, proceed with spawn

            # Environment for this instance
            env = os.environ.copy()
            env["N8N_PORT"] = str(port)
            env["N8N_USER_FOLDER"] = data_folder

            # CRITICAL: Disable all auth and setup screens
            # For n8n v1.x+, these are the key settings
            env["N8N_AUTH_EXCLUDE_ENDPOINTS"] = "*"  # Exclude all from auth
            env["N8N_SECURITY_AUDIT_DAYS_ABANDONED_WORKFLOW"] = "0"

            # Disable user management completely
            env["N8N_USER_MANAGEMENT_DISABLED"] = "true"
            env["N8N_BASIC_AUTH_ACTIVE"] = "false"
            env["N8N_SKIP_OWNER_SETUP"] = "true"

            # Skip all onboarding/setup flows
            env["N8N_PERSONALIZATION_ENABLED"] = "false"
            env["N8N_ONBOARDING_FLOW_DISABLED"] = "true"
            env["N8N_TEMPLATES_ENABLED"] = "false"

            # NOTE: E2E_TESTS and N8N_PREVIEW_MODE were removed
            # They caused the UI to disable workflow creation

            # Enterprise features off
            env["N8N_LICENSE_ACTIVATION_KEY"] = ""
            env["N8N_HIDE_USAGE_PAGE"] = "true"

            env["N8N_EDITOR_BASE_URL"] = f"http://localhost:{port}"

            # Disable telemetry and banners
            env["N8N_DIAGNOSTICS_ENABLED"] = "false"
            env["N8N_HIRING_BANNER_ENABLED"] = "false"
            env["N8N_VERSION_NOTIFICATIONS_ENABLED"] = "false"
            env["N8N_PUBLIC_API_DISABLED"] = "false"
            env["N8N_RUNNERS_DISABLED"] = "true"

            # Build command
            if self.node_path and os.path.exists(self.node_path):
                cmd = [self.node_path, self.n8n_path, "start"]
            elif os.path.exists(self.n8n_path):
                cmd = ["node", self.n8n_path, "start"]
            else:
                cmd = ["npx", "n8n", "start"]

            logger.info(f"Spawning n8n on port {port}: {' '.join(cmd)}")

            try:
                # Don't capture output - let it flow to avoid pipe buffer blocking
                # On Windows, use CREATE_NO_WINDOW to hide console
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )

                instance = N8nInstance(
                    port=port,
                    process=proc,
                    data_folder=data_folder,
                )
                self.instances[port] = instance

                # Wait a moment to check if it started
                await asyncio.sleep(1)
                if not instance.is_running:
                    logger.error(f"n8n failed to start on port {port}")
                    del self.instances[port]
                    raise RuntimeError(f"n8n process exited immediately")

                logger.info(f"n8n started on port {port} (PID: {proc.pid})")

                # Register PID for orphan tracking
                if self.registry:
                    self.registry.register(port, proc.pid, "legacy")

                # Wait for n8n to be ready to serve HTTP
                ready = await self._wait_for_ready(port, timeout=60)
                if not ready:
                    logger.warning(f"n8n on port {port} started but not responding yet")
                else:
                    logger.info(f"n8n on port {port} is ready")
                    # Configure default user for auto-login
                    await self._configure_n8n_user(port, data_folder)

                return instance

            except Exception as e:
                logger.error(f"Failed to spawn n8n on port {port}: {e}")
                raise

    async def _wait_for_ready(self, port: int, timeout: int = 60) -> bool:
        """Wait for n8n to be ready to serve HTTP requests."""
        url = f"http://127.0.0.1:{port}/"
        start = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=5) as client:
            while (asyncio.get_event_loop().time() - start) < timeout:
                try:
                    resp = await client.get(url)
                    if resp.status_code in (200, 301, 302):
                        return True
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass

                # Check if process died
                if port in self.instances and not self.instances[port].is_running:
                    return False

                await asyncio.sleep(2)

        return False

    async def is_ready(self, port: int) -> bool:
        """Check if an n8n instance is ready to serve requests."""
        if port not in self.instances:
            return False
        if not self.instances[port].is_running:
            return False

        url = f"http://127.0.0.1:{port}/"
        async with httpx.AsyncClient(timeout=3) as client:
            try:
                resp = await client.get(url)
                return resp.status_code in (200, 301, 302)
            except:
                return False

    async def _configure_n8n_user(self, port: int, data_folder: str):
        """Configure default user in n8n database for auto-login."""
        import sqlite3
        try:
            import bcrypt
        except ImportError:
            logger.warning("bcrypt not available, skipping n8n user configuration")
            return

        db_path = os.path.join(data_folder, ".n8n", "database.sqlite")

        # Retry a few times - n8n may still be creating the schema
        max_retries = 5
        for attempt in range(max_retries):
            if not os.path.exists(db_path):
                logger.info(f"Waiting for n8n database (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(2)
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check if user table exists and has the roleSlug column
                cursor.execute("PRAGMA table_info(user)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'roleSlug' not in columns:
                    conn.close()
                    logger.info(f"n8n schema not ready yet (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(2)
                    continue

                # Check if user needs to be configured
                cursor.execute("SELECT email FROM user WHERE roleSlug = 'global:owner'")
                row = cursor.fetchone()

                if row is None:
                    # No owner user yet, wait for n8n to create it
                    conn.close()
                    logger.info(f"No owner user yet (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(2)
                    continue

                if row[0] == "admin@example.com":
                    # Already configured
                    conn.close()
                    return

                # Generate password hash
                password = 'password123'.encode('utf-8')
                hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=10)).decode('utf-8')

                # Update user
                cursor.execute("""
                    UPDATE user SET
                        email = 'admin@example.com',
                        firstName = 'Admin',
                        lastName = 'User',
                        password = ?,
                        settings = '{"userActivated":true,"isOnboarded":true}'
                    WHERE roleSlug = 'global:owner'
                """, (hashed,))

                # Mark owner as set up
                cursor.execute('''
                    INSERT OR REPLACE INTO settings (key, value, loadOnStartup)
                    VALUES ('userManagement.isInstanceOwnerSetUp', 'true', 1)
                ''')

                conn.commit()
                conn.close()
                logger.info(f"Configured n8n user for port {port}")
                return

            except Exception as e:
                logger.warning(f"Error configuring n8n user (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(2)

        logger.error(f"Failed to configure n8n user after {max_retries} attempts")

    async def stop(self, port: int) -> bool:
        """Stop an n8n instance by port."""
        async with self._lock:
            if port not in self.instances:
                logger.warning(f"No n8n instance on port {port}")
                return False

            instance = self.instances[port]

            if instance.process:
                try:
                    instance.process.terminate()

                    # Wait for graceful shutdown
                    try:
                        instance.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        instance.process.kill()
                        instance.process.wait()

                    logger.info(f"Stopped n8n on port {port}")

                except Exception as e:
                    logger.error(f"Error stopping n8n on port {port}: {e}")
            else:
                logger.info(f"Detaching from adopted n8n on port {port}")

            # Unregister from PID registry
            if self.registry:
                self.registry.unregister(port)

            del self.instances[port]
            return True

    async def stop_all(self):
        """Stop all running n8n instances."""
        ports = list(self.instances.keys())
        for port in ports:
            await self.stop(port)
        logger.info("All n8n instances stopped")

    def list(self) -> List[dict]:
        """List all instances."""
        # Clean up dead instances
        dead = [port for port, inst in self.instances.items() if not inst.is_running]
        for port in dead:
            del self.instances[port]

        return [inst.to_dict() for inst in self.instances.values()]

    def get(self, port: int) -> Optional[N8nInstance]:
        """Get instance by port."""
        return self.instances.get(port)

    def _next_available_port(self) -> int:
        """Find next available port starting from base_port."""
        used = set(self.instances.keys())
        port = self.base_port
        while port in used:
            port += 1
        return port

    @property
    def count(self) -> int:
        """Number of running instances."""
        return len([i for i in self.instances.values() if i.is_running])


# =============================================================================
# N8nQueueManager - New queue-based manager with isolated worker databases
# =============================================================================

@dataclass
class MainInstance:
    """Represents the main admin n8n instance."""
    port: int
    process: Optional[subprocess.Popen]
    data_folder: str
    started_at: Optional[datetime] = None
    adopted: bool = False  # True if we adopted an existing process we don't own

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def is_running(self) -> bool:
        if self.adopted:
            # For adopted instances, check if the port is still responding
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect(('127.0.0.1', self.port))
                    return True
            except (OSError, ConnectionRefusedError):
                return False
        return self.process is not None and self.process.poll() is None

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "url": self.url,
            "pid": self.pid,
            "data_folder": self.data_folder,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "is_running": self.is_running,
            "adopted": self.adopted,
        }


@dataclass
class WorkerInstance:
    """Represents a worker n8n instance for a single workflow."""
    port: int
    process: subprocess.Popen
    workflow_id: str
    workflow_name: str
    data_folder: str
    mode: str  # 'once' | 'loop' | 'standby'
    started_at: datetime = field(default_factory=datetime.now)
    last_execution: Optional[datetime] = None
    last_status: Optional[str] = None  # 'success' | 'error' | 'running' | None
    loop_count: int = 0
    loop_target: Optional[int] = None  # None = infinite
    execution_count: int = 0
    active: bool = False  # Whether workflow is activated (triggers enabled)
    trigger_count: int = 0  # Number of trigger nodes in the workflow
    # Queue-based execution fields
    _queued_total: Optional[int] = 0  # Total runs requested (None = infinite)
    _queued_parallel: bool = False  # True = fire all at once, False = sequential
    _paused: bool = False  # Queue processing paused
    _processing: bool = False  # Background queue processor is active

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "url": self.url,
            "pid": self.pid,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "data_folder": self.data_folder,
            "mode": self.mode,
            "started_at": self.started_at.isoformat(),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_status": self.last_status,
            "is_running": self.is_running,
            "loop_count": self.loop_count,
            "loop_target": self.loop_target,
            "execution_count": self.execution_count,
            "active": self.active,
            "trigger_count": self.trigger_count,
            "queued_total": self._queued_total,
            "queued_parallel": self._queued_parallel,
            "paused": self._paused,
            "processing": self._processing,
        }


class N8nQueueManager:
    """
    Queue-based n8n manager with isolated worker databases.

    Architecture:
    - Main Admin (port 5678): For editing workflows and credentials
    - Workers (ports 5679+): Each gets isolated database, runs one workflow

    This solves SQLite lock contention when running 100+ workflows concurrently.
    """

    def __init__(
        self,
        n8n_path: str = "n8n",
        node_path: Optional[str] = None,
        data_base_dir: Optional[str] = None,
        main_port: int = 5678,
        worker_port_range: tuple = (5679, 5778),
        registry: Optional['ProcessRegistry'] = None,
    ):
        self.n8n_path = n8n_path
        self.node_path = node_path
        self.main_port = main_port
        self.worker_port_range = worker_port_range
        self.registry = registry

        self.data_base_dir = data_base_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".n8n-instances"
        )

        # Main admin instance
        self.main: Optional[MainInstance] = None

        # Worker instances: port -> WorkerInstance
        self.workers: Dict[int, WorkerInstance] = {}

        # Locks
        self._lock = asyncio.Lock()
        self._credential_sync_task: Optional[asyncio.Task] = None
        self._worker_monitor_task: Optional[asyncio.Task] = None

        # Database utilities (lazy init)
        self._db_utils = None

        # Cached live execution data from worker monitor polls
        self._cached_executions: Dict[int, List[dict]] = {}  # port -> list of exec dicts

        # Ensure directories exist
        os.makedirs(os.path.join(self.data_base_dir, "main"), exist_ok=True)
        os.makedirs(os.path.join(self.data_base_dir, "workers"), exist_ok=True)
        os.makedirs(os.path.join(self.data_base_dir, "history"), exist_ok=True)

        # Clean up stale workers from previous runs
        self._cleanup_stale_workers()

    def _cleanup_stale_workers(self):
        """
        Clean up orphaned worker folders from previous runs.

        Retries with backoff since orphaned processes may have just been killed
        and Windows file locks take time to release.
        """
        import shutil
        import time

        workers_dir = os.path.join(self.data_base_dir, "workers")
        if not os.path.exists(workers_dir):
            return

        cleaned = 0
        skipped = 0
        try:
            for folder in os.listdir(workers_dir):
                folder_path = os.path.join(workers_dir, folder)
                if os.path.isdir(folder_path) and folder.startswith('wf-'):
                    removed = False
                    for attempt in range(3):
                        try:
                            shutil.rmtree(folder_path)
                            cleaned += 1
                            removed = True
                            break
                        except (PermissionError, OSError):
                            if attempt < 2:
                                time.sleep(1)  # Wait for file lock release
                        except Exception as e:
                            logger.warning(f"Could not remove {folder}: {e}")
                            break
                    if not removed:
                        logger.info(f"Skipped worker folder {folder} (still locked by another process)")
                        skipped += 1
        except Exception as e:
            logger.warning(f"Worker cleanup error: {e}")

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale worker folder(s)")
        if skipped > 0:
            logger.warning(f"Could not clean {skipped} worker folder(s) - processes may still be running")

    @property
    def db_utils(self):
        """Lazy-load database utilities."""
        if self._db_utils is None:
            from backend.n8n_db_utils import N8nDatabaseUtils
            main_db = os.path.join(self.data_base_dir, "main", ".n8n", "database.sqlite")
            history_db = os.path.join(self.data_base_dir, "history", "history.sqlite")
            self._db_utils = N8nDatabaseUtils(main_db, history_db)
        return self._db_utils

    def _get_n8n_env(self, port: int, data_folder: str) -> dict:
        """Get environment variables for n8n instance."""
        env = os.environ.copy()
        env["N8N_PORT"] = str(port)
        env["N8N_USER_FOLDER"] = data_folder

        # Disable auth and setup
        env["N8N_AUTH_EXCLUDE_ENDPOINTS"] = "*"
        env["N8N_SECURITY_AUDIT_DAYS_ABANDONED_WORKFLOW"] = "0"
        env["N8N_USER_MANAGEMENT_DISABLED"] = "true"
        env["N8N_BASIC_AUTH_ACTIVE"] = "false"
        env["N8N_SKIP_OWNER_SETUP"] = "true"
        env["N8N_PERSONALIZATION_ENABLED"] = "false"
        env["N8N_ONBOARDING_FLOW_DISABLED"] = "true"
        env["N8N_TEMPLATES_ENABLED"] = "false"
        env["N8N_LICENSE_ACTIVATION_KEY"] = ""
        env["N8N_HIDE_USAGE_PAGE"] = "true"
        env["N8N_EDITOR_BASE_URL"] = f"http://localhost:{port}"
        env["N8N_DIAGNOSTICS_ENABLED"] = "false"
        env["N8N_HIRING_BANNER_ENABLED"] = "false"
        env["N8N_VERSION_NOTIFICATIONS_ENABLED"] = "false"
        env["N8N_PUBLIC_API_DISABLED"] = "false"
        env["N8N_RUNNERS_DISABLED"] = "true"

        # Enable native Python in Code nodes
        env["N8N_PYTHON_ENABLED"] = "true"

        return env

    def _get_n8n_cmd(self) -> List[str]:
        """Get command to start n8n."""
        if self.node_path and os.path.exists(self.node_path):
            return [self.node_path, self.n8n_path, "start"]
        elif os.path.exists(self.n8n_path):
            return ["node", self.n8n_path, "start"]
        else:
            return ["npx", "n8n", "start"]

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False

    def _next_available_port(self) -> Optional[int]:
        """Find next available port in worker range."""
        used = set(self.workers.keys())
        for port in range(self.worker_port_range[0], self.worker_port_range[1] + 1):
            if port not in used and self._is_port_available(port):
                return port
        return None

    async def _wait_for_ready(self, port: int, timeout: int = 60) -> bool:
        """Wait for n8n to be ready to serve HTTP requests."""
        url = f"http://127.0.0.1:{port}/"
        start = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=5) as client:
            while (asyncio.get_event_loop().time() - start) < timeout:
                try:
                    resp = await client.get(url)
                    if resp.status_code in (200, 301, 302):
                        return True
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                await asyncio.sleep(2)

        return False

    # =========================================================================
    # Main Admin Instance
    # =========================================================================

    async def start_main(self) -> MainInstance:
        """
        Start the main admin n8n instance.

        If an n8n is already running on the main port (e.g. started by the
        legacy N8nManager via meta-agent tools), adopt it instead of spawning
        a new one.

        Returns:
            MainInstance with details about the admin instance
        """
        async with self._lock:
            # Check if already running (tracked by us)
            if self.main and self.main.is_running:
                logger.info(f"Main admin already running on port {self.main_port}")
                return self.main

            data_folder = os.path.join(self.data_base_dir, "main")
            os.makedirs(data_folder, exist_ok=True)

            # Check if something is already running on the main port
            # (e.g. legacy N8nManager spawned by meta-agent)
            if not self._is_port_available(self.main_port):
                logger.info(f"Port {self.main_port} already in use, checking if it's n8n...")
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"http://127.0.0.1:{self.main_port}/")
                        if resp.status_code in (200, 301, 302):
                            logger.info(f"Adopting existing n8n on port {self.main_port}")
                            self.main = MainInstance(
                                port=self.main_port,
                                process=None,  # We don't own the process
                                data_folder=data_folder,
                                started_at=datetime.now(),
                                adopted=True,
                            )
                            return self.main
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                # Port busy but not responding as n8n - try to proceed anyway
                logger.warning(f"Port {self.main_port} in use but not responding, will attempt to start")

            env = self._get_n8n_env(self.main_port, data_folder)
            cmd = self._get_n8n_cmd()

            logger.info(f"Starting main admin on port {self.main_port}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )

                self.main = MainInstance(
                    port=self.main_port,
                    process=proc,
                    data_folder=data_folder,
                    started_at=datetime.now(),
                )

                # Register PID for orphan tracking
                if self.registry:
                    self.registry.register(self.main_port, proc.pid, "main")

                # Wait for startup
                await asyncio.sleep(1)
                if not self.main.is_running:
                    raise RuntimeError("Main admin process exited immediately")

                # Wait for HTTP ready
                ready = await self._wait_for_ready(self.main_port, timeout=60)
                if ready:
                    logger.info(f"Main admin ready on port {self.main_port}")
                else:
                    logger.warning(f"Main admin started but not responding yet")

                return self.main

            except Exception as e:
                logger.error(f"Failed to start main admin: {e}")
                self.main = None
                raise

    async def stop_main(self) -> bool:
        """Stop the main admin n8n instance."""
        async with self._lock:
            if not self.main:
                logger.warning("Main admin not running")
                return False

            if self.main.adopted:
                # Adopted instance - we don't own the process, just detach
                logger.info("Detaching from adopted main admin (not killing)")
                self.main = None
                return True

            if not self.main.process:
                logger.warning("Main admin has no process")
                self.main = None
                return False

            try:
                self.main.process.terminate()
                try:
                    self.main.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.main.process.kill()
                    self.main.process.wait()

                logger.info("Main admin stopped")

                # Unregister from PID registry
                if self.registry:
                    self.registry.unregister(self.main_port)

                self.main = None
                return True

            except Exception as e:
                logger.error(f"Error stopping main admin: {e}")
                return False

    def get_main_status(self) -> dict:
        """Get main admin status."""
        if self.main:
            return {
                "running": self.main.is_running,
                **self.main.to_dict()
            }
        return {"running": False, "port": self.main_port}

    async def get_workflows(self) -> List[dict]:
        """
        Get list of workflows.

        Tries the database first; if empty and main admin is running,
        falls back to querying the n8n REST API (handles cases where
        workflows were deployed via API but n8n hasn't flushed to disk yet).
        """
        workflows = self.db_utils.list_workflows()

        # If DB returned nothing but main admin is running, try the live API
        if not workflows and self.main and self.main.is_running:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    # Login to get auth cookie
                    login_resp = await client.post(
                        f"http://127.0.0.1:{self.main_port}/rest/login",
                        json={"emailOrLdapLoginId": "admin@example.com", "password": "password123"}
                    )
                    headers = {}
                    if login_resp.status_code == 200:
                        cookie = login_resp.cookies.get("n8n-auth")
                        if cookie:
                            headers["cookie"] = f"n8n-auth={cookie}"

                    resp = await client.get(
                        f"http://127.0.0.1:{self.main_port}/rest/workflows",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        api_workflows = data.get("data", data.get("workflows", []))
                        if isinstance(api_workflows, list):
                            workflows = [
                                {
                                    "id": w.get("id"),
                                    "name": w.get("name", "Unnamed"),
                                    "active": w.get("active", False),
                                    "node_count": len(w.get("nodes", [])),
                                    "trigger_count": w.get("triggerCount", 0) or 0,
                                    "created_at": w.get("createdAt"),
                                    "updated_at": w.get("updatedAt"),
                                }
                                for w in api_workflows
                            ]
                            logger.info(f"Got {len(workflows)} workflows from live n8n API (DB was empty)")
            except Exception as e:
                logger.warning(f"Failed to query live n8n API for workflows: {e}")

        return workflows

    # =========================================================================
    # Worker Instances
    # =========================================================================

    async def spawn_worker(
        self,
        workflow_id: str,
        mode: str = 'once',
        loop_target: Optional[int] = None
    ) -> WorkerInstance:
        """
        Spawn a worker instance for a specific workflow.

        Args:
            workflow_id: The workflow ID to run
            mode: Execution mode ('once', 'loop', 'standby')
            loop_target: For loop mode, number of iterations (None = infinite)

        Returns:
            WorkerInstance with details about the worker
        """
        async with self._lock:
            # Find available port
            port = self._next_available_port()
            if port is None:
                raise RuntimeError("No available ports for worker")

            # Get workflow info
            workflow = self.db_utils.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")

            workflow_name = workflow.get('name', 'Unknown')
            trigger_count = workflow.get('triggerCount', 0) or 0

            # Create worker data folder
            worker_id = f"wf-{workflow_id[:8]}-{port}"
            worker_folder = os.path.join(self.data_base_dir, "workers", worker_id)
            worker_db_path = os.path.join(worker_folder, ".n8n", "database.sqlite")

            os.makedirs(os.path.dirname(worker_db_path), exist_ok=True)

            # Copy workflow and credentials to worker database
            debug_logger.info(f"[MANAGER] Copying workflow '{workflow_name}' (id={workflow_id}) to worker db: {worker_db_path}")
            success = self.db_utils.copy_workflow_to_worker(
                workflow_id,
                worker_db_path,
                worker_name=worker_id
            )
            if not success:
                debug_logger.error(f"[MANAGER] Failed to copy workflow to worker db")
                raise RuntimeError(f"Failed to create worker database")
            debug_logger.info(f"[MANAGER] Workflow copied successfully")

            # Start n8n worker
            env = self._get_n8n_env(port, worker_folder)
            cmd = self._get_n8n_cmd()

            logger.info(f"Starting worker on port {port} for workflow '{workflow_name}'")
            debug_logger.info(f"[MANAGER] Spawning worker port={port} workflow='{workflow_name}' mode={mode} loop_target={loop_target}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )

                worker = WorkerInstance(
                    port=port,
                    process=proc,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    data_folder=worker_folder,
                    mode=mode,
                    loop_target=loop_target,
                    trigger_count=trigger_count,
                )

                self.workers[port] = worker

                # Register PID for orphan tracking
                if self.registry:
                    self.registry.register(port, proc.pid, "worker", workflow_id=workflow_id)

                # Wait for startup
                await asyncio.sleep(1)
                if not worker.is_running:
                    del self.workers[port]
                    if self.registry:
                        self.registry.unregister(port)
                    raise RuntimeError("Worker process exited immediately")

                # Wait for HTTP ready
                ready = await self._wait_for_ready(port, timeout=60)
                if ready:
                    logger.info(f"Worker ready on port {port}")
                    debug_logger.info(f"[MANAGER] Worker READY port={port} PID={proc.pid} workflow='{workflow_name}'")
                else:
                    logger.warning(f"Worker started but not responding yet")
                    debug_logger.warning(f"[MANAGER] Worker NOT READY port={port} PID={proc.pid} (started but not responding)")

                return worker

            except Exception as e:
                logger.error(f"Failed to start worker: {e}")
                if port in self.workers:
                    del self.workers[port]
                raise

    async def stop_worker(self, port: int, cleanup: bool = True) -> bool:
        """
        Stop a worker instance.

        Args:
            port: Worker port
            cleanup: Whether to delete worker database

        Returns:
            True if successful
        """
        async with self._lock:
            if port not in self.workers:
                logger.warning(f"No worker on port {port}")
                return False

            worker = self.workers[port]
            worker._processing = False  # Stop any background queue processor
            worker._paused = True

            # Aggregate execution history before stopping
            try:
                self.db_utils.aggregate_executions(
                    os.path.join(worker.data_folder, ".n8n", "database.sqlite"),
                    worker.workflow_id,
                    worker.workflow_name,
                    port,
                    worker.mode
                )
            except Exception as e:
                logger.warning(f"Failed to aggregate history: {e}")

            # Stop process
            debug_logger.info(f"[MANAGER] Stopping worker port={port} workflow='{worker.workflow_name}' mode={worker.mode} executions={worker.execution_count}")
            try:
                worker.process.terminate()
                try:
                    worker.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    worker.process.wait(timeout=5)

                logger.info(f"Worker on port {port} stopped")
                debug_logger.info(f"[MANAGER] Worker STOPPED port={port}")

            except Exception as e:
                logger.error(f"Error stopping worker on port {port}: {e}")

            # Cleanup database if requested (retry for Windows file lock release)
            if cleanup:
                import shutil
                for attempt in range(5):
                    try:
                        if os.path.exists(worker.data_folder):
                            shutil.rmtree(worker.data_folder)
                            logger.debug(f"Cleaned up worker folder: {worker.data_folder}")
                        break
                    except PermissionError:
                        if attempt < 4:
                            await asyncio.sleep(1)
                        else:
                            logger.warning(f"Failed to cleanup worker folder (locked): {worker.data_folder}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup worker folder: {e}")
                        break

            # Unregister from PID registry
            if self.registry:
                self.registry.unregister(port)

            # Clean up cached execution data
            self._cached_executions.pop(port, None)

            del self.workers[port]
            return True

    async def stop_all_workers(self):
        """Stop all worker instances."""
        ports = list(self.workers.keys())
        for port in ports:
            await self.stop_worker(port)
        logger.info("All workers stopped")

    def list_workers(self) -> List[dict]:
        """List all worker instances."""
        # Clean up dead workers
        dead = [port for port, w in self.workers.items() if not w.is_running]
        for port in dead:
            del self.workers[port]

        return [w.to_dict() for w in self.workers.values()]

    def get_worker(self, port: int) -> Optional[WorkerInstance]:
        """Get worker by port."""
        return self.workers.get(port)

    def get_worker_for_workflow(self, workflow_id: str) -> Optional[WorkerInstance]:
        """Get worker running a specific workflow."""
        for worker in self.workers.values():
            if worker.workflow_id == workflow_id and worker.is_running:
                return worker
        return None

    # =========================================================================
    # Credential Sync
    # =========================================================================

    async def start_credential_sync(self, interval: int = 30):
        """Start background credential sync task."""
        if self._credential_sync_task:
            return

        async def sync_loop():
            while True:
                try:
                    await self.sync_credentials_now()
                except Exception as e:
                    logger.error(f"Credential sync error: {e}")
                await asyncio.sleep(interval)

        self._credential_sync_task = asyncio.create_task(sync_loop())
        logger.info(f"Started credential sync (interval: {interval}s)")

    async def stop_credential_sync(self):
        """Stop credential sync task."""
        if self._credential_sync_task:
            self._credential_sync_task.cancel()
            try:
                await self._credential_sync_task
            except asyncio.CancelledError:
                pass
            self._credential_sync_task = None
            logger.info("Stopped credential sync")

    async def sync_credentials_now(self):
        """Sync credentials from main to all workers."""
        for port, worker in self.workers.items():
            if worker.is_running:
                worker_db = os.path.join(worker.data_folder, ".n8n", "database.sqlite")
                self.db_utils.sync_credentials_to_worker(worker_db)
        logger.debug(f"Synced credentials to {len(self.workers)} workers")

    # =========================================================================
    # Worker Monitoring
    # =========================================================================

    async def start_worker_monitor(self, interval: int = 10):
        """Start background worker execution monitor."""
        if self._worker_monitor_task:
            return

        async def monitor_loop():
            while True:
                try:
                    await self._poll_worker_executions()
                except Exception as e:
                    logger.error(f"Worker monitor error: {e}")
                await asyncio.sleep(interval)

        self._worker_monitor_task = asyncio.create_task(monitor_loop())
        logger.info(f"Started worker monitor (interval: {interval}s)")

    async def stop_worker_monitor(self):
        """Stop worker monitor task."""
        if self._worker_monitor_task:
            self._worker_monitor_task.cancel()
            try:
                await self._worker_monitor_task
            except asyncio.CancelledError:
                pass
            self._worker_monitor_task = None
            logger.info("Stopped worker monitor")

    async def _poll_worker_executions(self):
        """Poll each worker's n8n for execution status updates."""
        import httpx

        for port, worker in list(self.workers.items()):
            if not worker.is_running:
                continue

            try:
                # Get auth cookie for this worker
                from backend.routes.n8n import _get_or_create_auth
                auth = await _get_or_create_auth(port)
                headers = {"Cookie": f"n8n-auth={auth}"} if auth else {}

                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"http://127.0.0.1:{port}/rest/executions",
                        params={"workflowId": worker.workflow_id, "limit": 50},
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    executions = data.get("data", data.get("results", []))
                    # Handle nested dict: {"data": {"results": [...], "count": N}}
                    if isinstance(executions, dict):
                        executions = executions.get("results", executions.get("data", []))
                    if not isinstance(executions, list):
                        executions = []
                    if not executions:
                        continue

                    # Filter to executions since spawn (time-based, no baseline needed)
                    spawn_iso = worker.started_at.isoformat()
                    recent = [
                        e for e in executions
                        if (e.get("startedAt") or e.get("started") or "") > spawn_iso
                    ]

                    # Cache normalized execution data for the Executions tab
                    cached = []
                    for e in recent:
                        started = e.get("startedAt") or e.get("started") or ""
                        finished = e.get("stoppedAt") or e.get("finished")
                        e_status = e.get("status", "unknown")
                        exec_time_ms = None
                        if started and finished:
                            try:
                                s_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                                f_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
                                exec_time_ms = int((f_dt - s_dt).total_seconds() * 1000)
                            except Exception:
                                pass
                        cached.append({
                            "id": f"{port}-{e.get('id', '')}",
                            "workflow_id": worker.workflow_id,
                            "workflow_name": worker.workflow_name,
                            "worker_port": port,
                            "mode": worker.mode,
                            "status": e_status,
                            "started_at": started,
                            "finished_at": finished,
                            "execution_time_ms": exec_time_ms,
                        })
                    self._cached_executions[port] = cached

                    # Update execution_count (all runs since spawn)
                    old_count = worker.execution_count
                    worker.execution_count = len(recent)
                    if worker.execution_count != old_count:
                        debug_logger.info(f"[MONITOR] Worker :{port} execution_count {old_count} -> {worker.execution_count}")

                    # Update loop_count (successful runs since spawn)
                    completed = sum(
                        1 for e in recent
                        if e.get("status") in ("success", "finished")
                    )
                    old_loop = worker.loop_count
                    worker.loop_count = completed
                    if worker.loop_count != old_loop:
                        debug_logger.info(f"[MONITOR] Worker :{port} loop_count {old_loop} -> {worker.loop_count} (target={worker.loop_target})")

                    # Latest execution status
                    latest = recent[0] if recent else (executions[0] if executions else None)
                    if latest:
                        finished = latest.get("stoppedAt") or latest.get("finished")
                        started = latest.get("startedAt") or latest.get("started")
                        status = latest.get("status", "unknown")

                        if finished:
                            worker.last_execution = datetime.fromisoformat(
                                finished.replace("Z", "+00:00")
                            )
                        elif started:
                            worker.last_execution = datetime.fromisoformat(
                                started.replace("Z", "+00:00")
                            )

                        if status in ("success", "finished"):
                            worker.last_status = "success"
                        elif status in ("error", "failed", "crashed"):
                            worker.last_status = "error"
                        elif status in ("running", "waiting", "new"):
                            worker.last_status = "running"
                        else:
                            worker.last_status = status

                    # Auto-stop loop if target reached (trigger-based workflows)
                    if (worker.mode == "loop"
                            and worker.loop_target
                            and worker.loop_count >= worker.loop_target
                            and worker.active):
                        debug_logger.info(f"[MONITOR] Worker :{port} LOOP TARGET REACHED ({worker.loop_count}/{worker.loop_target}), deactivating")
                        logger.info(
                            f"Worker :{port} reached loop target "
                            f"({worker.loop_count}/{worker.loop_target}), deactivating"
                        )
                        await self.deactivate_worker(port)

            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout):
                debug_logger.info(f"[MONITOR] Worker :{port} unreachable")
            except Exception as e:
                debug_logger.info(f"[MONITOR] Poll failed for :{port}: {type(e).__name__}: {e}")

    async def activate_worker(self, port: int) -> dict:
        """
        Activate the workflow on a worker (enable triggers).
        Returns {"success": True} or {"success": False, "error": "reason"}.
        """
        worker = self.workers.get(port)
        if not worker or not worker.is_running:
            debug_logger.warning(f"[MANAGER] activate_worker :{port} - worker not found or not running")
            return {"success": False, "error": f"No running worker on port {port}"}

        debug_logger.info(f"[MANAGER] Activating worker :{port} workflow='{worker.workflow_name}'")
        try:
            import httpx
            from backend.routes.n8n import _get_or_create_auth
            auth = await _get_or_create_auth(port)
            headers = {"Cookie": f"n8n-auth={auth}"} if auth else {}

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.patch(
                    f"http://127.0.0.1:{port}/rest/workflows/{worker.workflow_id}",
                    json={"active": True},
                    headers=headers,
                )
                if resp.status_code == 200:
                    worker.active = True
                    logger.info(f"Activated workflow on worker :{port}")
                    debug_logger.info(f"[MANAGER] Worker :{port} ACTIVATED (HTTP {resp.status_code})")
                    return {"success": True}
                else:
                    # Parse n8n error message
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("message", resp.text[:200])
                    except Exception:
                        err_msg = resp.text[:200]
                    debug_logger.error(f"[MANAGER] activate_worker :{port} HTTP {resp.status_code}: {err_msg}")
                    return {"success": False, "error": err_msg}
        except Exception as e:
            logger.error(f"Failed to activate worker :{port}: {e}")
            debug_logger.error(f"[MANAGER] activate_worker :{port} EXCEPTION: {e}")
            return {"success": False, "error": str(e)}

    async def deactivate_worker(self, port: int) -> bool:
        """Deactivate the workflow on a worker (disable triggers, keep worker alive)."""
        worker = self.workers.get(port)
        if not worker or not worker.is_running:
            debug_logger.warning(f"[MANAGER] deactivate_worker :{port} - worker not found or not running")
            return False

        was_processing = worker._processing
        worker._processing = False  # Stop any background queue processor
        worker._paused = True
        debug_logger.info(f"[MANAGER] Deactivating worker :{port} workflow='{worker.workflow_name}' (was_processing={was_processing})")

        # For triggerless workflows, stopping the processor is all we need
        if not worker.trigger_count and not worker.active:
            worker.active = False
            logger.info(f"Stopped processing on worker :{port}")
            debug_logger.info(f"[MANAGER] Worker :{port} processing stopped (triggerless, was not activated)")
            return True

        try:
            import httpx
            from backend.routes.n8n import _get_or_create_auth
            auth = await _get_or_create_auth(port)
            headers = {"Cookie": f"n8n-auth={auth}"} if auth else {}

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.patch(
                    f"http://127.0.0.1:{port}/rest/workflows/{worker.workflow_id}",
                    json={"active": False},
                    headers=headers,
                )
                if resp.status_code == 200:
                    worker.active = False
                    logger.info(f"Deactivated workflow on worker :{port}")
                    debug_logger.info(f"[MANAGER] Worker :{port} DEACTIVATED (HTTP {resp.status_code})")
                    return True
                else:
                    debug_logger.error(f"[MANAGER] deactivate_worker :{port} HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Failed to deactivate worker :{port}: {e}")
            debug_logger.error(f"[MANAGER] deactivate_worker :{port} EXCEPTION: {e}")
        return False

    async def change_worker_mode(self, port: int, mode: str, loop_target: Optional[int] = None) -> bool:
        """Change a worker's execution mode."""
        worker = self.workers.get(port)
        if not worker or not worker.is_running:
            return False

        if mode not in ("once", "loop", "standby"):
            return False

        old_mode = worker.mode
        old_target = worker.loop_target
        worker.mode = mode
        worker.loop_target = loop_target

        # Stop background processor if leaving loop mode
        if mode != "loop":
            worker._processing = False
            worker._paused = True

        debug_logger.info(f"[MANAGER] Mode change :{port} {old_mode}(target={old_target}) -> {mode}(target={loop_target})")

        # Activate if switching to loop/standby (only for trigger-based workflows)
        if mode in ("loop", "standby") and not worker.active and worker.trigger_count > 0:
            await self.activate_worker(port)

        logger.info(f"Worker :{port} mode changed to {mode}" +
                     (f" (target: {loop_target})" if loop_target else ""))
        return True

    # =========================================================================
    # History
    # =========================================================================

    async def aggregate_all_history(self) -> int:
        """Aggregate execution history from all workers."""
        total = 0
        for port, worker in self.workers.items():
            worker_db = os.path.join(worker.data_folder, ".n8n", "database.sqlite")
            count = self.db_utils.aggregate_executions(
                worker_db,
                worker.workflow_id,
                worker.workflow_name,
                port,
                worker.mode
            )
            total += count
        return total

    def get_execution_history(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[dict]:
        """Get aggregated execution history."""
        return self.db_utils.get_execution_history(workflow_id, status, since, limit, offset)

    def get_history_stats(self, workflow_id: Optional[str] = None, since: Optional[str] = None) -> dict:
        """Get execution count stats grouped by status."""
        return self.db_utils.get_history_stats(workflow_id, since)

    def get_distinct_workflows(self) -> List[dict]:
        """Get distinct workflow IDs and names from history."""
        return self.db_utils.get_distinct_workflows()

    def clear_history(self) -> bool:
        """Clear all execution history."""
        return self.db_utils.clear_history()

    async def get_live_executions(self) -> dict:
        """Get live execution data from all running workers.

        Uses cached data from the worker monitor (updated every 10s) instead of
        making fresh HTTP calls. Returns dict with 'executions' and 'queued'.
        """
        executions = []
        queued = []

        for port, worker in list(self.workers.items()):
            # Queue state
            qt = worker._queued_total
            completed = worker.loop_count
            if qt is not None and qt > 0 or qt is None:
                remaining = None if qt is None else max(0, qt - completed)
                if remaining is None or remaining > 0 or worker._processing:
                    queued.append({
                        "worker_port": port,
                        "workflow_id": worker.workflow_id,
                        "workflow_name": worker.workflow_name,
                        "queued_total": qt,
                        "completed": completed,
                        "remaining": remaining,
                        "parallel": worker._queued_parallel,
                        "paused": worker._paused,
                        "processing": worker._processing,
                    })

            # Use cached execution data from monitor polls (no HTTP calls)
            cached = self._cached_executions.get(port, [])
            executions.extend(cached)

        return {"executions": executions, "queued": queued}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def shutdown(self):
        """Shutdown all instances and cleanup."""
        await self.stop_worker_monitor()
        await self.stop_credential_sync()
        await self.stop_all_workers()
        await self.stop_main()
        logger.info("N8nQueueManager shutdown complete")
