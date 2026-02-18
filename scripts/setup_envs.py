#!/usr/bin/env python3
"""
Environment Setup Script for AgentNate Inference Backends

Creates isolated Python virtual environments for vLLM.
Each environment is completely separate to avoid dependency conflicts.

Usage:
    python setup_envs.py                    # Setup all environments
    python setup_envs.py --env vllm         # Setup only vLLM
    python setup_envs.py --list             # List environment status
    python setup_envs.py --clean            # Remove all environments
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Optional, List


# Default Python 3.10 path (required for vLLM compatibility)
DEFAULT_PYTHON_310 = r"C:\Users\chris\AppData\Local\Programs\Python\Python310\python.exe"

# Environment configurations
ENVIRONMENTS = {
    "vllm": {
        "requirements": "requirements/vllm.txt",
        "path": "envs/vllm",
        "description": "vLLM - High-throughput inference with PagedAttention",
        "torch_index": "https://download.pytorch.org/whl/cu121",  # CUDA 12.1
    },
}

# Global python executable (set by args)
PYTHON_EXE = None


def get_base_dir() -> Path:
    """Get the AgentNate base directory."""
    return Path(__file__).parent.parent


def get_python_executable(env_path: Path) -> Path:
    """Get the Python executable path for an environment."""
    if sys.platform == "win32":
        return env_path / "Scripts" / "python.exe"
    else:
        return env_path / "bin" / "python"


def get_pip_executable(env_path: Path) -> Path:
    """Get the pip executable path for an environment."""
    if sys.platform == "win32":
        return env_path / "Scripts" / "pip.exe"
    else:
        return env_path / "bin" / "pip"


def env_exists(env_path: Path) -> bool:
    """Check if an environment already exists."""
    python = get_python_executable(env_path)
    return python.exists()


def create_venv(env_path: Path) -> bool:
    """Create a new virtual environment."""
    print(f"Creating virtual environment at {env_path}...")
    print(f"  Using Python: {PYTHON_EXE}")

    try:
        # Create the venv
        subprocess.run(
            [PYTHON_EXE, "-m", "venv", str(env_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Upgrade pip (use python -m pip to avoid issues)
        python = get_python_executable(env_path)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
            text=True,
        )

        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to create environment: {e}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False


def install_torch(env_path: Path, index_url: str) -> bool:
    """Install PyTorch with CUDA support."""
    print(f"  Installing PyTorch with CUDA support...")

    python = get_python_executable(env_path)

    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "torch", "--index-url", index_url],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to install PyTorch: {e}")
        if e.stderr:
            print(f"  {e.stderr[:500]}")
        return False


def install_requirements(env_path: Path, requirements_file: Path) -> bool:
    """Install requirements from file."""
    print(f"  Installing requirements from {requirements_file.name}...")

    python = get_python_executable(env_path)

    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to install requirements: {e}")
        if e.stderr:
            print(f"  {e.stderr[:500]}")
        return False


def setup_environment(env_name: str, force: bool = False) -> bool:
    """Setup a single environment."""
    if env_name not in ENVIRONMENTS:
        print(f"Unknown environment: {env_name}")
        return False

    config = ENVIRONMENTS[env_name]
    base_dir = get_base_dir()
    env_path = base_dir / config["path"]
    requirements_file = base_dir / config["requirements"]

    print(f"\n{'=' * 60}")
    print(f"Setting up {env_name}: {config['description']}")
    print(f"{'=' * 60}")

    # Check if already exists
    if env_exists(env_path):
        if force:
            print(f"Removing existing environment...")
            shutil.rmtree(env_path)
        else:
            print(f"Environment already exists. Use --force to recreate.")
            return True

    # Create directory if needed
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Create venv
    if not create_venv(env_path):
        return False

    # Install PyTorch with CUDA
    if not install_torch(env_path, config["torch_index"]):
        return False

    # Install requirements (excluding torch since we installed it separately)
    if not install_requirements(env_path, requirements_file):
        return False

    print(f"\n  SUCCESS: {env_name} environment ready!")
    print(f"  Python: {get_python_executable(env_path)}")

    return True


def list_environments():
    """List status of all environments."""
    base_dir = get_base_dir()

    print("\nEnvironment Status:")
    print("-" * 60)

    for env_name, config in ENVIRONMENTS.items():
        env_path = base_dir / config["path"]
        python = get_python_executable(env_path)

        if python.exists():
            status = "READY"
            # Try to get Python version
            try:
                result = subprocess.run(
                    [str(python), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version = result.stdout.strip()
                status = f"READY ({version})"
            except:
                pass
        else:
            status = "NOT INSTALLED"

        print(f"  {env_name:12} {status:30} {config['description']}")

    print()


def clean_environments(env_names: Optional[List[str]] = None):
    """Remove environments."""
    base_dir = get_base_dir()

    if env_names is None:
        env_names = list(ENVIRONMENTS.keys())

    for env_name in env_names:
        if env_name not in ENVIRONMENTS:
            print(f"Unknown environment: {env_name}")
            continue

        config = ENVIRONMENTS[env_name]
        env_path = base_dir / config["path"]

        if env_path.exists():
            print(f"Removing {env_name} environment...")
            shutil.rmtree(env_path)
            # Recreate the directory (empty) to keep the structure
            env_path.mkdir(parents=True, exist_ok=True)
            print(f"  Removed: {env_path}")
        else:
            print(f"  {env_name}: Not installed")


def main():
    global PYTHON_EXE

    parser = argparse.ArgumentParser(
        description="Setup isolated Python environments for AgentNate inference backends"
    )
    parser.add_argument(
        "--env", "-e",
        choices=list(ENVIRONMENTS.keys()),
        help="Setup specific environment only",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reinstall even if environment exists",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_envs",
        help="List environment status",
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Remove environments",
    )
    parser.add_argument(
        "--cuda",
        choices=["11.8", "12.1", "12.4"],
        default="12.1",
        help="CUDA version for PyTorch (default: 12.1)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help=f"Python executable to use (default: {DEFAULT_PYTHON_310})",
    )

    args = parser.parse_args()

    # Determine Python executable
    if args.python:
        PYTHON_EXE = args.python
    elif os.path.exists(DEFAULT_PYTHON_310):
        PYTHON_EXE = DEFAULT_PYTHON_310
    else:
        print(f"WARNING: Python 3.10 not found at {DEFAULT_PYTHON_310}")
        print(f"Using system Python: {sys.executable}")
        print(f"This may cause compatibility issues with vLLM (need Python 3.9-3.11)")
        PYTHON_EXE = sys.executable

    # Verify Python exists
    if not os.path.exists(PYTHON_EXE):
        print(f"ERROR: Python executable not found: {PYTHON_EXE}")
        sys.exit(1)

    # Show Python version
    try:
        result = subprocess.run([PYTHON_EXE, "--version"], capture_output=True, text=True)
        print(f"Using {result.stdout.strip()}")
    except Exception as e:
        print(f"ERROR: Cannot run Python: {e}")
        sys.exit(1)

    # Update torch index URLs based on CUDA version
    cuda_map = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.4": "https://download.pytorch.org/whl/cu124",
    }
    for config in ENVIRONMENTS.values():
        config["torch_index"] = cuda_map[args.cuda]

    if args.list_envs:
        list_environments()
        return

    if args.clean:
        if args.env:
            clean_environments([args.env])
        else:
            response = input("Remove ALL environments? [y/N] ")
            if response.lower() == "y":
                clean_environments()
        return

    # Setup environments
    if args.env:
        success = setup_environment(args.env, force=args.force)
    else:
        print("Setting up all inference backend environments...")
        print(f"CUDA version: {args.cuda}")

        all_success = True
        for env_name in ENVIRONMENTS:
            if not setup_environment(env_name, force=args.force):
                all_success = False

        if all_success:
            print("\n" + "=" * 60)
            print("All environments setup successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Some environments failed to setup. Check errors above.")
            print("=" * 60)
            sys.exit(1)


if __name__ == "__main__":
    main()
