# gpu_utils.py
import subprocess
import logging

logger = logging.getLogger(__name__)


def get_available_gpus():
    """
    Uses nvidia-smi to detect GPUs.
    Returns list like: ['CPU', 'GPU 0: NVIDIA GeForce RTX 4090', 'GPU 1: NVIDIA GeForce RTX 3080']
    CPU is always first for convenience.
    """
    gpus = ["CPU"]  # Always offer CPU first

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10  # Don't hang if nvidia-smi is stuck
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(", ")
            if len(parts) >= 3:
                idx, name, memory = parts[0], parts[1], parts[2]
                gpus.append(f"GPU {idx}: {name.strip()} ({memory} MiB)")
            else:
                logger.warning(f"Unexpected nvidia-smi output: {line}")
    except FileNotFoundError:
        logger.info("nvidia-smi not found - no NVIDIA GPUs available")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out - GPU detection skipped")
    except subprocess.CalledProcessError as e:
        logger.warning(f"nvidia-smi failed with code {e.returncode}: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error detecting GPUs: {type(e).__name__}: {e}")

    logger.debug(f"Detected devices: {gpus}")
    return gpus

def get_default_selection():
    """Returns the default GPU selection (first GPU if available, else CPU)."""
    gpus = get_available_gpus()
    # Prefer first GPU if any, else CPU
    return gpus[1] if len(gpus) > 1 else "CPU"


def get_gpu_count():
    """Returns the number of GPUs detected (excludes CPU)."""
    gpus = get_available_gpus()
    return len([g for g in gpus if g.startswith("GPU")])