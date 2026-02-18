"""
Code Tools - Execute Python, JavaScript, and shell commands.
"""

from typing import Dict, Any, List, Optional
import logging
import subprocess
import tempfile
import os
import sys
import shutil
from pathlib import Path

logger = logging.getLogger("tools.code")


TOOL_DEFINITIONS = [
    {
        "name": "run_python",
        "description": "Execute Python code. Returns stdout, stderr, and return code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 300)",
                    "default": 30
                },
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Packages to install before running (pip install)"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "run_javascript",
        "description": "Execute JavaScript/Node.js code. Requires Node.js installed.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 300)",
                    "default": 30
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "run_shell",
        "description": "Execute a shell command (bash on Linux/Mac, cmd on Windows).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 300)",
                    "default": 30
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory (optional)"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "run_powershell",
        "description": "Execute PowerShell commands (Windows only).",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "PowerShell script/command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 300)",
                    "default": 30
                }
            },
            "required": ["script"]
        }
    },
]


class CodeTools:
    """Tools for code execution."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize code tools.

        Args:
            config: Optional configuration dict with keys:
                - timeout_seconds: int - Default timeout
                - max_output_chars: int - Max output length
                - allow_shell: bool - Allow shell commands
                - working_dir: str - Default working directory
                - blocked_commands: list - Commands to block
        """
        self.config = config or {}
        self.timeout = min(self.config.get("timeout_seconds", 30), 300)
        self.max_output = self.config.get("max_output_chars", 50000)
        self.allow_shell = self.config.get("allow_shell", True)
        self.working_dir = self.config.get("working_dir")
        self.blocked_commands = set(self.config.get("blocked_commands", [
            "rm -rf /", "format", "mkfs", "dd if=", ":(){:|:&};:",
            "shutdown", "reboot", "halt", "poweroff"
        ]))

    def _check_blocked(self, command: str) -> Optional[str]:
        """Check if command is blocked for safety."""
        cmd_lower = command.lower()
        for blocked in self.blocked_commands:
            if blocked in cmd_lower:
                return f"Command blocked for safety: contains '{blocked}'"
        return None

    def _truncate_output(self, output: str) -> str:
        """Truncate output if too long."""
        if len(output) > self.max_output:
            return output[:self.max_output] + f"\n\n[Output truncated at {self.max_output} chars...]"
        return output

    def _get_creationflags(self):
        """Get subprocess creation flags for Windows."""
        if sys.platform == "win32":
            return subprocess.CREATE_NO_WINDOW
        return 0

    async def run_python(
        self,
        code: str,
        timeout: Optional[int] = None,
        packages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            packages: Packages to pip install first

        Returns:
            Dict with execution results
        """
        timeout = min(timeout or self.timeout, 300)

        try:
            # Install packages if requested
            if packages:
                for pkg in packages:
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-q", pkg],
                            capture_output=True,
                            timeout=60,
                            creationflags=self._get_creationflags()
                        )
                    except Exception as e:
                        logger.warning(f"Failed to install {pkg}: {e}")

            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                script_path = f.name

            try:
                # Execute the script
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.working_dir or tempfile.gettempdir(),
                    creationflags=self._get_creationflags()
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": self._truncate_output(result.stdout),
                    "stderr": self._truncate_output(result.stderr),
                    "return_code": result.returncode
                }

            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout}s",
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            logger.error(f"run_python error: {e}")
            return {"success": False, "error": str(e)}

    async def run_javascript(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute JavaScript/Node.js code.

        Args:
            code: JavaScript code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with execution results
        """
        timeout = min(timeout or self.timeout, 300)

        # Check for Node.js
        node_cmd = shutil.which("node")
        if not node_cmd:
            return {
                "success": False,
                "error": "Node.js not found. Please install Node.js."
            }

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.js',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                script_path = f.name

            try:
                result = subprocess.run(
                    [node_cmd, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.working_dir or tempfile.gettempdir(),
                    creationflags=self._get_creationflags()
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": self._truncate_output(result.stdout),
                    "stderr": self._truncate_output(result.stderr),
                    "return_code": result.returncode
                }

            finally:
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout}s",
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            logger.error(f"run_javascript error: {e}")
            return {"success": False, "error": str(e)}

    async def run_shell(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds
            working_dir: Working directory

        Returns:
            Dict with execution results
        """
        if not self.allow_shell:
            return {
                "success": False,
                "error": "Shell commands are disabled in configuration"
            }

        # Safety check
        blocked = self._check_blocked(command)
        if blocked:
            return {"success": False, "error": blocked}

        timeout = min(timeout or self.timeout, 300)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir or self.working_dir,
                creationflags=self._get_creationflags()
            )

            return {
                "success": result.returncode == 0,
                "command": command,
                "stdout": self._truncate_output(result.stdout),
                "stderr": self._truncate_output(result.stderr),
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "command": command,
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            logger.error(f"run_shell error: {e}")
            return {"success": False, "error": str(e)}

    async def run_powershell(
        self,
        script: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute PowerShell commands (Windows only).

        Args:
            script: PowerShell script/commands
            timeout: Execution timeout in seconds

        Returns:
            Dict with execution results
        """
        if sys.platform != "win32":
            return {
                "success": False,
                "error": "PowerShell is only available on Windows"
            }

        if not self.allow_shell:
            return {
                "success": False,
                "error": "Shell commands are disabled in configuration"
            }

        # Safety check
        blocked = self._check_blocked(script)
        if blocked:
            return {"success": False, "error": blocked}

        timeout = min(timeout or self.timeout, 300)

        # Find PowerShell
        ps_cmd = shutil.which("powershell") or shutil.which("pwsh")
        if not ps_cmd:
            return {
                "success": False,
                "error": "PowerShell not found"
            }

        try:
            # Write script to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.ps1',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(script)
                script_path = f.name

            try:
                result = subprocess.run(
                    [ps_cmd, "-ExecutionPolicy", "Bypass", "-File", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.working_dir,
                    creationflags=self._get_creationflags()
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": self._truncate_output(result.stdout),
                    "stderr": self._truncate_output(result.stderr),
                    "return_code": result.returncode
                }

            finally:
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Script timed out after {timeout}s",
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            logger.error(f"run_powershell error: {e}")
            return {"success": False, "error": str(e)}
