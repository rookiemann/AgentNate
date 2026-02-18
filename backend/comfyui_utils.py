"""
Shared ComfyUI utilities.

Extracted from comfyui_tools.py so both the agent tool and REST endpoint
can use the same workflow building and model detection logic.
"""

import random
from typing import Dict, Any


# Approximate VRAM usage during generation (MB)
MODEL_VRAM_ESTIMATES = {
    "sd15": 4000,
    "sdxl": 6000,
    "sd3": 12000,
    "flux": 12000,
    "flux_full": 24000,
    "ltx": 12000,
    "ltx_full": 20000,
    "wan": 14000,
    "wan_small": 8000,
    "qwen_image": 10000,
    "default": 6000,
}


def estimate_vram_mb(checkpoint: str) -> int:
    """Estimate VRAM requirement in MB from checkpoint filename."""
    if checkpoint is None:
        return MODEL_VRAM_ESTIMATES["default"]
    name = str(checkpoint).lower()
    if "wan" in name:
        if "1.3b" in name or "small" in name:
            return MODEL_VRAM_ESTIMATES["wan_small"]
        return MODEL_VRAM_ESTIMATES["wan"]
    if "qwen" in name and ("image" in name or "edit" in name):
        return MODEL_VRAM_ESTIMATES["qwen_image"]
    if "ltx" in name:
        if "fp32" in name or "full" in name:
            return MODEL_VRAM_ESTIMATES["ltx_full"]
        return MODEL_VRAM_ESTIMATES["ltx"]
    if "flux" in name:
        if "fp32" in name or "full" in name:
            return MODEL_VRAM_ESTIMATES["flux_full"]
        return MODEL_VRAM_ESTIMATES["flux"]
    if "sd3" in name or "sd_3" in name or "sd3.5" in name:
        return MODEL_VRAM_ESTIMATES["sd3"]
    if "xl" in name or "sdxl" in name:
        return MODEL_VRAM_ESTIMATES["sdxl"]
    if "1.5" in name or "v1-5" in name or "sd15" in name:
        return MODEL_VRAM_ESTIMATES["sd15"]
    return MODEL_VRAM_ESTIMATES["default"]


def detect_model_defaults(checkpoint: str) -> Dict[str, Any]:
    """Auto-detect generation defaults based on checkpoint filename."""
    name_lower = str(checkpoint or "").lower()

    if "ltx" in name_lower:
        return {
            "cfg": 3.0, "sampler_name": "euler", "scheduler": "normal",
            "steps": 30, "width": 768, "height": 512,
        }
    elif "wan" in name_lower:
        return {
            "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal",
            "steps": 30, "width": 1024, "height": 576,
        }
    elif "qwen" in name_lower and ("image" in name_lower or "edit" in name_lower):
        return {
            "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
            "steps": 25, "width": 640, "height": 640,
        }
    elif "flux" in name_lower:
        return {
            "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple",
            "steps": 25, "width": 1024, "height": 1024,
        }
    elif "xl" in name_lower or "sdxl" in name_lower:
        return {
            "cfg": 8.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "steps": 30, "width": 1024, "height": 1024,
        }
    else:
        return {
            "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "steps": 20, "width": 512, "height": 512,
        }


def ensure_checkpoint_extension(checkpoint: str) -> str:
    """
    Append .safetensors if the checkpoint name has no model file extension.
    ComfyUI requires exact filenames including extensions.
    """
    if not checkpoint:
        return checkpoint
    KNOWN_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")
    basename = checkpoint.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if any(basename.lower().endswith(ext) for ext in KNOWN_EXTENSIONS):
        return checkpoint
    return checkpoint + ".safetensors"


def build_txt2img_workflow(checkpoint: str, prompt: str, negative_prompt: str,
                           width: int, height: int, steps: int, cfg: float,
                           seed: int, sampler_name: str, scheduler: str) -> Dict:
    """Build a standard txt2img ComfyUI workflow (API format)."""
    if seed == -1:
        seed = random.randint(0, 2**63 - 1)

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0],
                "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler_name, "scheduler": scheduler,
                "denoise": 1.0,
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "AgentNate"}
        },
    }
