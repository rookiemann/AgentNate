"""
ComfyUI Tools - Manage ComfyUI image generation module.

Provides tools for the Meta Agent to control ComfyUI:
install, start/stop API, manage instances, list models.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import logging
import random
import time

import httpx

logger = logging.getLogger("tools.comfyui")

# --- Workflow analysis helpers (module-level for reuse) ---

_MODEL_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"})

_INPUT_FIELD_MAP = {
    "ckpt_name": "checkpoints",
    "vae_name": "vae",
    "clip_name": "text_encoders",
    "clip_name1": "clip",
    "clip_name2": "clip",
    "unet_name": "diffusion_models",
    "lora_name": "loras",
    "control_net_name": "controlnet",
    "model_name": "upscale_models",
    "style_model_name": "style_models",
}


def _is_model_filename(value: str) -> bool:
    """Check if a string looks like a model filename."""
    return any(value.lower().endswith(ext) for ext in _MODEL_EXTENSIONS)


def _infer_folder(class_type: str) -> str:
    """Infer model folder from a loader node's class_type."""
    ct = class_type.lower()
    if "checkpoint" in ct:
        return "checkpoints"
    if "lora" in ct:
        return "loras"
    if "vae" in ct:
        return "vae"
    if "clip" in ct or "textencoder" in ct or "text_encoder" in ct:
        return "text_encoders"
    if "upscale" in ct:
        return "upscale_models"
    if "controlnet" in ct or "control_net" in ct:
        return "controlnet"
    if "unet" in ct or "diffusion" in ct:
        return "diffusion_models"
    if "embedding" in ct:
        return "embeddings"
    return "unknown"


TOOL_DEFINITIONS = [
    {
        "name": "comfyui_status",
        "description": "Check ComfyUI module status (installed, API running, instances, GPUs)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_install",
        "description": "Install ComfyUI module (downloads portable installer, bootstraps Python/Git, installs ComfyUI). This is a multi-step process that may take several minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": "Specific step to run: 'download', 'bootstrap', 'start_api', 'install', or 'full' for all steps (default: full)"
                }
            },
            "required": []
        }
    },
    {
        "name": "comfyui_start_api",
        "description": "Start the ComfyUI management API server (required before managing instances)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_stop_api",
        "description": "Stop the ComfyUI management API server and all instances",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_list_instances",
        "description": "List all ComfyUI instances with their status, GPU, and port",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_add_instance",
        "description": "Add a new ComfyUI instance on a specific GPU and port",
        "parameters": {
            "type": "object",
            "properties": {
                "gpu_device": {
                    "type": "integer",
                    "description": "GPU device index (0, 1, etc.)"
                },
                "port": {
                    "type": "integer",
                    "description": "Port for the instance (default: auto-assign from 8188)"
                },
                "vram_mode": {
                    "type": "string",
                    "description": "VRAM mode: 'normal', 'low', 'none', or 'cpu' (default: normal)"
                }
            },
            "required": []
        }
    },
    {
        "name": "comfyui_start_instance",
        "description": "Start a ComfyUI instance by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "Instance ID to start"
                }
            },
            "required": ["instance_id"]
        }
    },
    {
        "name": "comfyui_stop_instance",
        "description": "Stop a running ComfyUI instance by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "Instance ID to stop"
                }
            },
            "required": ["instance_id"]
        }
    },
    {
        "name": "comfyui_list_models",
        "description": "List locally installed ComfyUI models (checkpoints, LoRAs, VAEs, etc.). To find what models a specific workflow NEEDS, use comfyui_list_templates + comfyui_analyze_workflow instead.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    # --- Generation pipeline tools ---
    {
        "name": "comfyui_search_models",
        "description": "Search for downloadable image generation models. PREFERRED: use category to browse the curated registry (has 80+ models with IDs ready for download). Use query only for models not in the registry. Categories: 'checkpoints', 'diffusion_models', 'vae', 'clip', 'text_encoders', 'loras', 'controlnet', 'gguf', 'unet', 'embeddings', 'upscale_models', 'clip_vision'. Omit both for all registry models.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for HuggingFace (e.g. 'flux', 'sdxl'). Only use if registry doesn't have what you need."
                },
                "category": {
                    "type": "string",
                    "description": "Model category to filter the curated registry (e.g. 'checkpoints', 'loras', 'vae')"
                }
            },
            "required": []
        }
    },
    {
        "name": "comfyui_download_model",
        "description": "Download models to the local ComfyUI installation. Returns a job_id for tracking progress with comfyui_job_status. Accepts registry IDs (from registry search) and/or HuggingFace model definitions (from HF search). WARNING: Large models (2-17 GB) take 5-20 minutes. Progress may stay at 0% during HuggingFace downloads — this is normal, check disk_progress in job status.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of registry model IDs to download (from comfyui_search_models registry results, e.g. ['sd15', 'flux_schnell'])"
                },
                "models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "filename": {"type": "string"},
                            "folder": {"type": "string"}
                        }
                    },
                    "description": "List of HuggingFace model defs to download (from HF search results, e.g. [{'repo': 'author/model', 'filename': 'model.safetensors', 'folder': 'checkpoints'}])"
                }
            },
            "required": []
        }
    },
    {
        "name": "comfyui_job_status",
        "description": "Check the status of an async ComfyUI job (model download, installation, node install). Returns status, progress, and any errors. NOTE: Model downloads may show 0% progress but still be active — check disk_progress field for real byte counts. Large models take 5-20 minutes. Poll every 30-60 seconds, not faster.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned by comfyui_download_model, comfyui_install, or comfyui_install_nodes"
                }
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "comfyui_await_job",
        "description": "Wait for a long-running ComfyUI job to complete (model download, node install, update). Blocks internally — polls every 15 seconds until the job finishes or fails. Use this instead of repeatedly calling comfyui_job_status. Returns final job status. Timeout: 30 minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned by comfyui_download_model, comfyui_install_nodes, etc."
                },
                "poll_interval": {
                    "type": "integer",
                    "description": "Seconds between status checks (default: 15, min: 5, max: 120)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait before giving up (default: 1800 = 30 min)"
                }
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "comfyui_generate_image",
        "description": "Generate an image using a ComfyUI instance. Builds a txt2img workflow and queues it. Returns prompt_id for tracking with comfyui_get_result. Auto-detects optimal settings from model type.",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "ComfyUI instance ID (from comfyui_list_instances)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Positive prompt describing the desired image"
                },
                "checkpoint": {
                    "type": "string",
                    "description": "Checkpoint filename with extension. Use comfyui_list_models to get exact filenames."
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Negative prompt — what to avoid (default: 'blurry, low quality, distorted')"
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels (default: auto based on model type)"
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels (default: auto based on model type)"
                },
                "steps": {
                    "type": "integer",
                    "description": "Sampling steps (default: auto based on model type)"
                },
                "cfg": {
                    "type": "number",
                    "description": "CFG scale / guidance (default: auto based on model type)"
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed, -1 for random (default: -1)"
                },
                "sampler_name": {
                    "type": "string",
                    "description": "Sampler (default: auto based on model type)"
                },
                "scheduler": {
                    "type": "string",
                    "description": "Scheduler (default: auto based on model type)"
                }
            },
            "required": ["instance_id", "prompt", "checkpoint"]
        }
    },
    {
        "name": "comfyui_get_result",
        "description": "Get the result of an image generation (non-blocking check). Returns status (running/completed/failed), output image filenames, view URLs, and disk paths. TIP: For long-running jobs (Flux ~5 min, SVD ~3 min), prefer comfyui_await_result which blocks until complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "ComfyUI instance ID"
                },
                "prompt_id": {
                    "type": "string",
                    "description": "Prompt ID returned by comfyui_generate_image"
                }
            },
            "required": ["instance_id", "prompt_id"]
        }
    },
    {
        "name": "comfyui_await_result",
        "description": "Wait for a ComfyUI generation to complete (blocking). Polls internally every 10 seconds until the job finishes, fails, or times out. RECOMMENDED for all generation jobs — avoids needing to poll comfyui_get_result manually. Returns the full result with filenames when done. To chain outputs to the next workflow step, pass any output filename to comfyui_prepare_input.",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "ComfyUI instance ID"
                },
                "prompt_id": {
                    "type": "string",
                    "description": "Prompt ID returned by comfyui_generate_image or comfyui_build_workflow"
                },
                "poll_interval": {
                    "type": "integer",
                    "description": "Seconds between status checks (default: 10, min: 5, max: 60)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (default: 600 = 10 min, max: 1800 = 30 min)"
                }
            },
            "required": ["instance_id", "prompt_id"]
        }
    },
    {
        "name": "comfyui_install_nodes",
        "description": "Install custom node packs for advanced ComfyUI workflows (ControlNet, IP-Adapter, etc.). Returns job_id for tracking with comfyui_job_status.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of custom node IDs to install (from nodes registry)"
                }
            },
            "required": ["node_ids"]
        }
    },
    # --- Workflow building tools ---
    {
        "name": "comfyui_describe_nodes",
        "description": "Discover available ComfyUI node types for building workflows. Returns simplified schemas with inputs, outputs, and descriptions. Use category/search to filter ~40 common nodes, or query a running instance for any of 4000+ node types. Use class_types=['all'] for full catalog, class_types=['templates'] for ready-made workflow templates.",
        "parameters": {
            "type": "object",
            "properties": {
                "class_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node class_type names to describe. Special: ['all'] for full catalog, ['templates'] for ready-made templates."
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category: loader, sampling, conditioning, latent, image, mask, utility, controlnet"
                },
                "search": {
                    "type": "string",
                    "description": "Search node names/descriptions. If no matches in static catalog, automatically queries the live instance (when instance_id provided) for custom nodes like LTX, ControlNet, etc."
                },
                "instance_id": {
                    "type": "string",
                    "description": "Query a running instance's /object_info for live node data (for custom nodes not in catalog)"
                }
            },
            "required": []
        }
    },
    {
        "name": "comfyui_build_workflow",
        "description": "Build a ComfyUI workflow from a ready-made template, native template, or custom node specs. When execute=true and instance_id is provided, also immediately executes the workflow (recommended — avoids needing to pass workflow JSON to comfyui_execute_workflow).",
        "parameters": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "Built-in template: 'txt2img', 'img2img', 'upscale', 'inpaint', 'txt2img_hires', 'controlnet_pose', 'svd_img2video', 'ltxv_img2video'. Use comfyui_describe_nodes(class_types=['templates']) to see all available templates with details."
                },
                "native_template": {
                    "type": "string",
                    "description": "Native ComfyUI template name from comfyui_list_templates (e.g. 'ltxv_text_to_video', 'video_ltx2_t2v'). Requires instance_id for conversion."
                },
                "instance_id": {
                    "type": "string",
                    "description": "Running instance ID. Required for native_template conversion and for execute=true."
                },
                "overrides": {
                    "type": "object",
                    "description": "Parameter overrides (e.g. {\"prompt\": \"a cat\", \"checkpoint\": \"model.safetensors\", \"width\": 1024}). For native templates, keys match node input names."
                },
                "execute": {
                    "type": "boolean",
                    "description": "If true (default), immediately execute the built workflow on instance_id. Returns prompt_id for tracking with comfyui_get_result."
                },
                "nodes": {
                    "type": "array",
                    "description": "Custom workflow: list of node specs. Each: {id, class_type, inputs}. Connections: {\"node_ref\": \"ref_name\", \"output_index\": N}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Reference name for connections"},
                            "class_type": {"type": "string", "description": "ComfyUI node class_type"},
                            "inputs": {"type": "object", "description": "Node input parameters and connections"}
                        },
                        "required": ["class_type"]
                    }
                }
            },
            "required": []
        }
    },
    {
        "name": "comfyui_execute_workflow",
        "description": "Execute any ComfyUI workflow on an instance. Takes workflow JSON from comfyui_build_workflow (or raw API format). Returns prompt_id for tracking with comfyui_get_result.",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "ComfyUI instance ID to run on"
                },
                "workflow": {
                    "type": "object",
                    "description": "Workflow in ComfyUI API format (dict of node_id -> {class_type, inputs})"
                },
                "workflow_json": {
                    "type": "object",
                    "description": "Alias for workflow. Accepts the same ComfyUI API format."
                }
            },
            "required": ["instance_id"]
        }
    },
    {
        "name": "comfyui_search_generations",
        "description": "Search the generation catalog for previously generated images. Filter by prompt text, checkpoint, tags, favorites, or rating. Returns a list of matching generations with metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search prompt text (optional)"
                },
                "checkpoint": {
                    "type": "string",
                    "description": "Filter by checkpoint/model name (optional)"
                },
                "tags": {
                    "type": "string",
                    "description": "Filter by comma-separated tags (optional)"
                },
                "favorite": {
                    "type": "boolean",
                    "description": "Only return favorites (optional)"
                },
                "min_rating": {
                    "type": "integer",
                    "description": "Minimum rating 1-5 (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)"
                },
            },
            "required": []
        }
    },
    {
        "name": "comfyui_prepare_input",
        "description": "Prepare an image as input for a ComfyUI workflow (e.g. img2img, upscale, SVD video, WAN video). Copies the image to the instance's input/ folder so LoadImage nodes can reference it. Source can be: an output filename from a previous generation (e.g. 'AgentNate_00008_.png'), a generation ID (UUID from the catalog), an absolute file path, or a URL. This is the KEY tool for chaining workflow steps — call it between generate and the next build_workflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Output filename (e.g. 'AgentNate_00008_.png'), generation ID (UUID), absolute file path, or URL of the source image"
                },
                "instance_id": {
                    "type": "string",
                    "description": "ComfyUI instance ID to prepare input for"
                },
                "file_index": {
                    "type": "integer",
                    "description": "If source is a generation ID with multiple files, which file to use (default: 0)"
                },
            },
            "required": ["source", "instance_id"]
        }
    },
    # --- Node management tools ---
    {
        "name": "comfyui_list_node_packs",
        "description": "Browse the curated registry of installable custom node packs for ComfyUI. Shows name, description, and install status. Use this to discover what's available before installing with comfyui_install_nodes.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_list_installed_nodes",
        "description": "List all currently installed custom node packs in ComfyUI with names, versions, and paths.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_update_nodes",
        "description": "Update all installed custom node packs to their latest versions. Returns a job_id for tracking with comfyui_job_status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_remove_node",
        "description": "Remove an installed custom node pack by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_name": {
                    "type": "string",
                    "description": "Name of the custom node pack to remove (from comfyui_list_installed_nodes)"
                }
            },
            "required": ["node_name"]
        }
    },
    # --- Instance management ---
    {
        "name": "comfyui_remove_instance",
        "description": "Remove a ComfyUI instance permanently. The instance must be stopped first.",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "Instance ID to remove"
                }
            },
            "required": ["instance_id"]
        }
    },
    {
        "name": "comfyui_start_all_instances",
        "description": "Start all ComfyUI instances at once.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_stop_all_instances",
        "description": "Stop all running ComfyUI instances at once.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    # --- Model categories ---
    {
        "name": "comfyui_model_categories",
        "description": "List all model folder categories in ComfyUI (e.g. checkpoints, loras, vae, controlnet). Shows where different model types are stored.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    # --- Settings ---
    {
        "name": "comfyui_get_settings",
        "description": "Get current ComfyUI module settings (extra model dirs, default VRAM mode, etc.).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_update_settings",
        "description": "Update ComfyUI module settings. Pass only the fields you want to change.",
        "parameters": {
            "type": "object",
            "properties": {
                "settings": {
                    "type": "object",
                    "description": "Settings to update (e.g. {\"default_vram_mode\": \"low\", \"extra_model_dirs\": [...]})"
                }
            },
            "required": ["settings"]
        }
    },
    # --- Update / Purge ---
    {
        "name": "comfyui_update_comfyui",
        "description": "Update ComfyUI to the latest version via git pull. Returns a job_id for tracking. Stop all instances first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_purge",
        "description": "Purge ComfyUI installation (removes ComfyUI but keeps models and Python). Use with caution — requires re-installation afterwards.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    # --- External ComfyUI ---
    {
        "name": "comfyui_manage_external",
        "description": "Manage external ComfyUI installations. Actions: 'list' saved directories, 'add' a new external dir, 'remove' a saved dir, 'switch' the active target directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'list', 'add', 'remove', or 'switch'"
                },
                "directory": {
                    "type": "string",
                    "description": "ComfyUI directory path (required for add, remove, switch)"
                },
                "name": {
                    "type": "string",
                    "description": "Display name for the directory (optional, for add)"
                }
            },
            "required": ["action"]
        }
    },
    # --- GPUs ---
    {
        "name": "comfyui_list_gpus",
        "description": "List available GPUs for ComfyUI with VRAM info (total, free, used). Useful for deciding which GPU to assign instances to.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_list_templates",
        "description": "List ComfyUI's native workflow templates (text-to-video, image-to-video, etc.). These are pre-built graph-format workflows shipped with ComfyUI. Use comfyui_describe_nodes to learn node inputs, then comfyui_build_workflow + comfyui_execute_workflow to run custom workflows. Templates prefixed 'api_' require ComfyUI cloud credits.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category: 'video', 'image', 'api', 'other', 'core', or 'all' (default)",
                    "default": "all"
                },
                "search": {
                    "type": "string",
                    "description": "Search filter for template names (e.g. 'ltx', 'wan', 'flux')"
                }
            },
            "required": []
        }
    },

    {
        "name": "comfyui_analyze_workflow",
        "description": "Analyze a ComfyUI workflow JSON to extract all required models and check which are installed locally. Works with both graph-format (from UI/custom nodes) and API-format workflows. Returns installed vs missing models with download URLs when available. Use this before running unfamiliar workflows to ensure all required models are present.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_path": {
                    "type": "string",
                    "description": "Path or template name. Accepts: full path, relative path (e.g. 'custom_nodes/ComfyUI-LTXVideo/example_workflows/LTX-2_T2V_Distilled_wLora.json'), or just a template name (e.g. 'ltxv_text_to_video', 'LTX-2_T2V_Distilled_wLora'). Template names are auto-resolved."
                },
                "workflow_json": {
                    "type": "object",
                    "description": "Workflow JSON object directly (API format or graph format)"
                }
            },
            "required": []
        }
    },

    # ---- Pool tools (multi-instance dispatch) ----
    {
        "name": "comfyui_pool_status",
        "description": "Get pool overview: per-instance queue depths, loaded checkpoints, GPU assignments, and active jobs. Shows which instances are idle vs busy and what models they have in VRAM.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "comfyui_pool_generate",
        "description": "Submit an image generation job to the pool. Auto-selects the best instance based on model affinity (prefers instance with checkpoint already loaded to avoid ~30s model swap) and queue depth. Auto-downloads the checkpoint if not present locally. Returns a pool job_id for tracking with comfyui_pool_results.",
        "parameters": {
            "type": "object",
            "properties": {
                "checkpoint": {
                    "type": "string",
                    "description": "Checkpoint filename with extension. Use comfyui_list_models to get exact filenames."
                },
                "prompt": {
                    "type": "string",
                    "description": "Positive prompt describing the image"
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Negative prompt",
                    "default": "blurry, low quality, distorted"
                },
                "width": {"type": "integer", "description": "Image width (auto-detected from model if omitted)"},
                "height": {"type": "integer", "description": "Image height (auto-detected from model if omitted)"},
                "steps": {"type": "integer", "description": "Sampling steps (auto-detected if omitted)"},
                "cfg": {"type": "number", "description": "CFG scale (auto-detected if omitted)"},
                "seed": {"type": "integer", "description": "Random seed (-1 for random)", "default": -1},
                "sampler_name": {"type": "string", "description": "Sampler name (auto-detected if omitted)"},
                "scheduler": {"type": "string", "description": "Scheduler (auto-detected if omitted)"}
            },
            "required": ["checkpoint", "prompt"]
        }
    },
    {
        "name": "comfyui_pool_batch",
        "description": "Submit N image generation jobs distributed across pool instances. Each job can have a different prompt/seed but shares the checkpoint and generation params. The pool distributes jobs to minimize total generation time via model affinity and load balancing. Returns a batch_id for tracking with comfyui_pool_results.",
        "parameters": {
            "type": "object",
            "properties": {
                "checkpoint": {
                    "type": "string",
                    "description": "Shared checkpoint for all jobs"
                },
                "jobs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Positive prompt"},
                            "negative_prompt": {"type": "string", "description": "Negative prompt"},
                            "seed": {"type": "integer", "description": "Seed (-1 for random)"}
                        },
                        "required": ["prompt"]
                    },
                    "description": "List of job specs (each needs at least a prompt)"
                },
                "steps": {"type": "integer", "description": "Sampling steps for all jobs"},
                "cfg": {"type": "number", "description": "CFG scale for all jobs"},
                "width": {"type": "integer", "description": "Image width for all jobs"},
                "height": {"type": "integer", "description": "Image height for all jobs"},
                "sampler_name": {"type": "string", "description": "Sampler for all jobs"},
                "scheduler": {"type": "string", "description": "Scheduler for all jobs"}
            },
            "required": ["checkpoint", "jobs"]
        }
    },
    {
        "name": "comfyui_pool_results",
        "description": "Get results for a pool job or batch. For single jobs: returns status + images. For batches: returns aggregated status + per-job results with image URLs. Poll this until status is 'completed'.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Pool job_id or batch_id to check"
                }
            },
            "required": ["job_id"]
        }
    },
]


class _ObjectInfoCache:
    """Cache /object_info responses per instance port, with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[tuple, tuple] = {}  # (port, key) -> (timestamp, data)
        self._ttl = ttl_seconds

    def get(self, port: int, key: str = "all") -> Optional[Any]:
        cache_key = (port, key)
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self._ttl:
                return data
        return None

    def put(self, port: int, key: str, data: Any):
        self._cache[(port, key)] = (time.time(), data)

    def invalidate(self, port: int = None):
        if port:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != port}
        else:
            self._cache.clear()


class ComfyUITools:
    """Tools for managing the ComfyUI image generation module."""

    _object_info_cache = _ObjectInfoCache()

    def __init__(self, comfyui_manager=None, media_catalog=None, pool=None):
        self.manager = comfyui_manager
        self.catalog = media_catalog
        self.pool = pool  # ComfyUIPool instance
        self._last_workflows: Dict[str, dict] = {}  # prompt_id -> workflow JSON
        self._pending_parent_ids: Dict[str, str] = {}  # prompt_id -> parent generation_id

    def _check_manager(self) -> Optional[Dict[str, Any]]:
        """Return error dict if manager is not available."""
        if not self.manager:
            return {
                "success": False,
                "error": "ComfyUI module not configured. The ComfyUI manager is not initialized."
            }
        return None

    async def _check_api(self) -> Optional[Dict[str, Any]]:
        """Return error dict if API is not running."""
        err = self._check_manager()
        if err:
            return err
        if not await self.manager.is_api_running():
            return {
                "success": False,
                "error": "ComfyUI management API is not running. Use comfyui_start_api first."
            }
        return None

    async def comfyui_status(self) -> Dict[str, Any]:
        """Get ComfyUI module status."""
        err = self._check_manager()
        if err:
            return err

        try:
            status = await self.manager.get_status()
            return {
                "success": True,
                "module_downloaded": status["module_downloaded"],
                "bootstrapped": status["bootstrapped"],
                "comfyui_installed": status["comfyui_installed"],
                "api_running": status["api_running"],
                "api_port": status["api_port"],
                "instances": status.get("instances", []),
                "gpus": status.get("gpus", []),
                "summary": self._build_status_summary(status),
            }
        except Exception as e:
            logger.error(f"comfyui_status failed: {e}")
            return {"success": False, "error": str(e)}

    def _build_status_summary(self, status: Dict) -> str:
        """Build a human-readable status summary."""
        parts = []
        if not status["module_downloaded"]:
            parts.append("Module not downloaded. Run comfyui_install to set up.")
            return " ".join(parts)
        if not status["bootstrapped"]:
            parts.append("Module downloaded but not bootstrapped. Run comfyui_install to continue setup.")
            return " ".join(parts)
        if not status["comfyui_installed"]:
            parts.append("Bootstrapped but ComfyUI not installed. Run comfyui_install to finish.")
            return " ".join(parts)

        parts.append("ComfyUI is installed.")

        if status["api_running"]:
            parts.append(f"API running on port {status['api_port']}.")
            instances = status.get("instances", [])
            running = [i for i in instances if i.get("status") == "running" or i.get("running")]
            parts.append(f"{len(running)}/{len(instances)} instances running.")
        else:
            parts.append("API not running. Use comfyui_start_api to start.")

        gpus = status.get("gpus", [])
        if gpus:
            gpu_strs = [f"GPU {g.get('id', '?')}: {g.get('name', 'unknown')} ({g.get('memory_free', '?')} free)" for g in gpus]
            parts.append("GPUs: " + ", ".join(gpu_strs))

        return " ".join(parts)

    async def comfyui_install(self, step: str = "full") -> Dict[str, Any]:
        """Install ComfyUI module. Steps: download, bootstrap, start_api, install, or full."""
        err = self._check_manager()
        if err:
            return err

        results = []

        try:
            if step in ("full", "download"):
                if not self.manager.is_module_downloaded():
                    logger.info("ComfyUI install: downloading module...")
                    result = await self.manager.download_module()
                    results.append({"step": "download", **result})
                    if not result.get("success"):
                        return {"success": False, "error": f"Download failed: {result.get('error')}", "steps_completed": results}
                else:
                    results.append({"step": "download", "success": True, "message": "Already downloaded"})

                if step == "download":
                    return {"success": True, "steps_completed": results}

            if step in ("full", "bootstrap"):
                if not self.manager.is_bootstrapped():
                    logger.info("ComfyUI install: bootstrapping (this may take a few minutes)...")
                    result = await self.manager.bootstrap()
                    results.append({"step": "bootstrap", **result})
                    if not result.get("success"):
                        return {"success": False, "error": f"Bootstrap failed: {result.get('error')}", "steps_completed": results}
                else:
                    results.append({"step": "bootstrap", "success": True, "message": "Already bootstrapped"})

                if step == "bootstrap":
                    return {"success": True, "steps_completed": results}

            if step in ("full", "start_api"):
                if not await self.manager.is_api_running():
                    logger.info("ComfyUI install: starting API server...")
                    result = await self.manager.start_api_server()
                    results.append({"step": "start_api", **result})
                    if not result.get("success"):
                        return {"success": False, "error": f"API start failed: {result.get('error')}", "steps_completed": results}
                else:
                    results.append({"step": "start_api", "success": True, "message": "API already running"})

                if step == "start_api":
                    return {"success": True, "steps_completed": results}

            if step in ("full", "install"):
                # Always verify API is running before install (even if we just started it)
                if not await self.manager.is_api_running():
                    # If called as step="install" individually, try to start API first
                    if step == "install" and self.manager.is_bootstrapped():
                        logger.info("ComfyUI install: API not running, starting it first...")
                        api_result = await self.manager.start_api_server()
                        if not api_result.get("success"):
                            return {"success": False, "error": f"Cannot install: API server failed to start: {api_result.get('error')}",
                                    "steps_completed": results}
                    else:
                        results.append({"step": "install", "success": False, "error": "API server not running"})
                        return {"success": False, "error": "Cannot install: ComfyUI API server is not running. Start it first.",
                                "steps_completed": results}

                if not self.manager.is_comfyui_installed():
                    logger.info("ComfyUI install: installing ComfyUI (this may take several minutes)...")
                    try:
                        data = await self.manager.proxy("POST", "/api/install")
                        job_id = data.get("job_id")
                        results.append({"step": "install", "success": True, "job_id": job_id,
                                        "message": f"ComfyUI installation started (job: {job_id}). "
                                                    f"Use comfyui_job_status(job_id='{job_id}') to track progress. "
                                                    f"This downloads PyTorch + ComfyUI and may take 5-15 minutes."})
                    except Exception as e:
                        error_msg = str(e)
                        if "ConnectError" in error_msg or "connection" in error_msg.lower():
                            error_msg = f"Cannot reach ComfyUI API server on port {self.manager.api_port}. It may have crashed. Try comfyui_start_api first."
                        results.append({"step": "install", "success": False, "error": error_msg})
                        return {"success": False, "error": f"Install failed: {error_msg}", "steps_completed": results}
                else:
                    results.append({"step": "install", "success": True, "message": "ComfyUI already installed"})

            return {
                "success": True,
                "steps_completed": results,
                "message": f"Completed {len(results)} step(s) successfully."
            }

        except Exception as e:
            logger.error(f"comfyui_install failed: {e}")
            return {"success": False, "error": str(e), "steps_completed": results}

    async def comfyui_start_api(self) -> Dict[str, Any]:
        """Start the ComfyUI management API server."""
        err = self._check_manager()
        if err:
            return err

        if not self.manager.is_bootstrapped():
            return {
                "success": False,
                "error": "ComfyUI not bootstrapped yet. Run comfyui_install first."
            }

        try:
            result = await self.manager.start_api_server()
            return result
        except Exception as e:
            logger.error(f"comfyui_start_api failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_stop_api(self) -> Dict[str, Any]:
        """Stop the ComfyUI management API server."""
        err = self._check_manager()
        if err:
            return err

        try:
            result = await self.manager.stop_api_server()
            return result
        except Exception as e:
            logger.error(f"comfyui_stop_api failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_list_instances(self) -> Dict[str, Any]:
        """List all ComfyUI instances."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/instances")
            instances = data if isinstance(data, list) else data.get("instances", [])
            return {
                "success": True,
                "instances": instances,
                "count": len(instances),
            }
        except Exception as e:
            logger.error(f"comfyui_list_instances failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_add_instance(self, gpu_device: int = 0, port: Optional[int] = None,
                                    vram_mode: str = "normal") -> Dict[str, Any]:
        """Add a new ComfyUI instance."""
        err = await self._check_api()
        if err:
            return err

        body = {"gpu_device": gpu_device, "vram_mode": vram_mode}
        if port is not None:
            body["port"] = port

        try:
            data = await self.manager.proxy("POST", "/api/instances", json=body)
            return {
                "success": True,
                "instance": data,
                "message": f"Instance added on GPU {gpu_device}",
            }
        except Exception as e:
            logger.error(f"comfyui_add_instance failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_start_instance(self, instance_id: str) -> Dict[str, Any]:
        """Start a ComfyUI instance."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", f"/api/instances/{instance_id}/start")
            return {
                "success": True,
                "result": data,
                "message": f"Instance {instance_id} started",
            }
        except Exception as e:
            logger.error(f"comfyui_start_instance failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_stop_instance(self, instance_id: str) -> Dict[str, Any]:
        """Stop a ComfyUI instance."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", f"/api/instances/{instance_id}/stop")
            return {
                "success": True,
                "result": data,
                "message": f"Instance {instance_id} stopped",
            }
        except Exception as e:
            logger.error(f"comfyui_stop_instance failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_list_models(self) -> Dict[str, Any]:
        """List locally installed ComfyUI models."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/models/local")
            models = data if isinstance(data, list) else data.get("models", data)

            # Enrich each model with 'filename' (the exact name ComfyUI needs)
            if isinstance(models, dict):
                import os
                for folder, model_list in models.items():
                    if not isinstance(model_list, list):
                        continue
                    for m in model_list:
                        if isinstance(m, dict) and "path" in m and "filename" not in m:
                            m["filename"] = os.path.basename(m["path"])

            return {
                "success": True,
                "models": models,
                "count": sum(len(v) for v in models.values() if isinstance(v, list)) if isinstance(models, dict) else len(models),
                "hint": "Use the 'filename' field (with extension) as the checkpoint parameter for comfyui_generate_image and comfyui_pool_generate. To research what models a workflow requires, use comfyui_list_templates(search='...') to find templates, then comfyui_analyze_workflow(workflow_path='...') to check installed vs missing models.",
            }
        except Exception as e:
            logger.error(f"comfyui_list_models failed: {e}")
            return {"success": False, "error": str(e)}

    # ======================== Helper Methods ========================

    async def _get_instance_port(self, instance_id: str) -> Optional[int]:
        """Resolve an instance ID to its port number via the management API."""
        try:
            data = await self.manager.proxy("GET", "/api/instances")
            instances = data if isinstance(data, list) else data.get("instances", [])
            for inst in instances:
                if str(inst.get("id")) == str(instance_id) or str(inst.get("instance_id")) == str(instance_id):
                    return inst.get("port")
        except Exception as e:
            logger.error(f"Failed to resolve instance port: {e}")
        return None

    async def _instance_request(self, port: int, method: str, path: str,
                                json_body: dict = None, timeout: float = 60) -> Dict[str, Any]:
        """Make a direct HTTP request to a ComfyUI instance (bypasses management API)."""
        url = f"http://127.0.0.1:{port}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    r = await client.get(url)
                elif method == "POST":
                    r = await client.post(url, json=json_body)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if r.status_code >= 400:
                    return {"success": False, "error": f"ComfyUI returned {r.status_code}: {r.text[:500]}"}

                try:
                    return {"success": True, "data": r.json()}
                except Exception:
                    return {"success": True, "data": r.text}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Request to instance on port {port} timed out"}
        except httpx.ConnectError:
            return {"success": False, "error": f"Cannot connect to ComfyUI instance on port {port}. Is it running?"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_model_defaults(self, checkpoint: str) -> Dict[str, Any]:
        """Auto-detect generation defaults based on checkpoint filename."""
        from backend.comfyui_utils import detect_model_defaults
        return detect_model_defaults(checkpoint)

    def _build_workflow(self, checkpoint: str, prompt: str, negative_prompt: str,
                        width: int, height: int, steps: int, cfg: float,
                        seed: int, sampler_name: str, scheduler: str) -> Dict:
        """Build a standard txt2img ComfyUI workflow (API format)."""
        from backend.comfyui_utils import build_txt2img_workflow
        return build_txt2img_workflow(
            checkpoint, prompt, negative_prompt, width, height,
            steps, cfg, seed, sampler_name, scheduler,
        )

    # ======================== Generation Pipeline Tools ========================

    async def comfyui_search_models(self, query: str = None,
                                     category: str = None) -> Dict[str, Any]:
        """Search for downloadable models. Registry results have 'id' fields for easy download."""
        err = await self._check_api()
        if err:
            return err

        try:
            if query:
                # Search HuggingFace, but also check registry for matches
                data = await self.manager.proxy("GET", "/api/models/search",
                                                params={"q": query})
                hf_results = data.get("results", []) if isinstance(data, dict) else data
                # Also search registry for the same query
                try:
                    reg_data = await self.manager.proxy("GET", "/api/models/registry")
                    reg_models = reg_data.get("models", []) if isinstance(reg_data, dict) else reg_data
                    query_lower = query.lower()
                    matching_registry = [
                        m for m in reg_models
                        if query_lower in m.get("name", "").lower()
                        or query_lower in m.get("id", "").lower()
                        or query_lower in m.get("folder", "").lower()
                        or query_lower in m.get("repo", "").lower()
                    ]
                except Exception:
                    matching_registry = []

                return {
                    "success": True,
                    "registry_matches": matching_registry,
                    "registry_match_count": len(matching_registry),
                    "huggingface_results": hf_results,
                    "huggingface_count": len(hf_results),
                    "hint": "Use model 'id' from registry_matches with comfyui_download_model(model_ids=[...]). "
                            "For HuggingFace results, use comfyui_download_model(models=[{repo, filename, folder}]).",
                }
            elif category:
                data = await self.manager.proxy("GET", "/api/models/registry",
                                                params={"category": category})
            else:
                data = await self.manager.proxy("GET", "/api/models/registry")

            models = data.get("models", []) if isinstance(data, dict) else data
            return {
                "success": True,
                "models": models,
                "count": len(models),
                "source": "registry",
                "hint": "Use model 'id' field with comfyui_download_model(model_ids=[...]) to download.",
            }
        except Exception as e:
            logger.error(f"comfyui_search_models failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_download_model(self, model_ids: List[str] = None,
                                      models: List[Dict] = None) -> Dict[str, Any]:
        """Download models by registry ID and/or HuggingFace repo info."""
        err = await self._check_api()
        if err:
            return err

        if not model_ids and not models:
            return {"success": False, "error": "Provide model_ids (registry IDs) and/or models (HF repo defs)"}

        try:
            body = {}
            if model_ids:
                body["model_ids"] = model_ids
            if models:
                body["models"] = models

            total_count = len(model_ids or []) + len(models or [])
            data = await self.manager.proxy("POST", "/api/models/download", json=body)

            # The endpoint may return a job_id (async download) or an "ok" message (already installed)
            job_id = data.get("job_id")
            if job_id:
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": f"Download started for {total_count} model(s). "
                               f"Use comfyui_job_status with job_id '{job_id}' to track progress.",
                }
            else:
                return {
                    "success": True,
                    "message": data.get("message", "Models already installed."),
                }
        except Exception as e:
            logger.error(f"comfyui_download_model failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of an async job."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", f"/api/jobs/{job_id}")
            result = {
                "success": True,
                "job_id": job_id,
                "status": data.get("status", "unknown"),
                "progress": data.get("progress"),
                "message": data.get("message"),
                "error": data.get("error"),
                "result": data.get("result"),
            }

            # For download jobs stuck at 0%, check actual disk progress
            progress = data.get("progress", {}) or {}
            operation = data.get("operation", "")
            if (operation == "download_models"
                    and data.get("status") == "running"
                    and progress.get("current", 0) == 0):
                disk_info = self._check_download_disk_progress()
                if disk_info:
                    result["disk_progress"] = disk_info
                    result["note"] = (
                        "HuggingFace downloads don't report granular progress to the job tracker. "
                        "The download IS active — see disk_progress for real byte counts. "
                        "Large models (2-17 GB) may take 5-20 minutes. "
                        "Poll again in 60 seconds."
                    )

            return result
        except Exception as e:
            logger.error(f"comfyui_job_status failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_await_job(self, job_id: str, poll_interval: int = 15,
                                timeout: int = 1800) -> Dict[str, Any]:
        """Block until a long-running job completes, polling internally."""
        err = await self._check_api()
        if err:
            return err

        poll_interval = max(5, min(120, poll_interval))
        timeout = max(30, min(7200, timeout))
        start = time.time()
        last_status = {}

        try:
            while True:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    return {
                        "success": False,
                        "error": f"Timed out after {int(elapsed)}s waiting for job {job_id}",
                        "job_id": job_id,
                        "last_status": last_status,
                    }

                data = await self.manager.proxy("GET", f"/api/jobs/{job_id}")
                status = data.get("status", "unknown")
                last_status = data

                if status == "completed":
                    return {
                        "success": True,
                        "job_id": job_id,
                        "status": "completed",
                        "result": data.get("result"),
                        "message": data.get("message", "Job completed successfully"),
                        "elapsed_seconds": int(time.time() - start),
                    }

                if status in ("failed", "error", "cancelled"):
                    return {
                        "success": False,
                        "job_id": job_id,
                        "status": status,
                        "error": data.get("error", data.get("message", "Job failed")),
                        "elapsed_seconds": int(time.time() - start),
                    }

                # Still running — sleep then retry
                await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            # Agent was aborted by user — clean exit
            return {
                "success": False,
                "job_id": job_id,
                "status": "aborted",
                "error": "Await cancelled (agent aborted by user)",
                "elapsed_seconds": int(time.time() - start),
            }
        except Exception as e:
            logger.error(f"comfyui_await_job failed: {e}")
            return {"success": False, "error": str(e)}

    def _check_download_disk_progress(self) -> Optional[Dict]:
        """Check actual download progress by scanning .incomplete files on disk."""
        try:
            models_dir = self.manager.module_dir / "comfyui" / "models"
            if not models_dir.exists():
                return None

            total_bytes = 0
            incomplete_files = []
            for incomplete in models_dir.rglob("*.incomplete"):
                size = incomplete.stat().st_size
                total_bytes += size
                incomplete_files.append(size)

            if not incomplete_files:
                return None

            total_mb = round(total_bytes / (1024 * 1024))
            return {
                "active_chunks": len(incomplete_files),
                "downloaded_mb": total_mb,
                "downloading": True,
            }
        except Exception:
            return None

    async def comfyui_generate_image(self, instance_id: str, prompt: str,
                                      checkpoint: str,
                                      negative_prompt: str = "blurry, low quality, distorted",
                                      width: int = None, height: int = None,
                                      steps: int = None, cfg: float = None,
                                      seed: int = -1, sampler_name: str = None,
                                      scheduler: str = None) -> Dict[str, Any]:
        """Generate an image using a ComfyUI instance."""
        err = await self._check_api()
        if err:
            return err

        try:
            # Ensure checkpoint has file extension
            from backend.comfyui_utils import ensure_checkpoint_extension
            checkpoint = ensure_checkpoint_extension(checkpoint)

            port = await self._get_instance_port(instance_id)
            if not port:
                return {"success": False,
                        "error": f"Instance '{instance_id}' not found or has no port"}

            defaults = self._detect_model_defaults(checkpoint)
            width = width or defaults["width"]
            height = height or defaults["height"]
            steps = steps or defaults["steps"]
            cfg = cfg if cfg is not None else defaults["cfg"]
            sampler_name = sampler_name or defaults["sampler_name"]
            scheduler = scheduler or defaults["scheduler"]

            workflow = self._build_workflow(
                checkpoint=checkpoint, prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, steps=steps, cfg=cfg,
                seed=seed, sampler_name=sampler_name, scheduler=scheduler,
            )

            result = await self._instance_request(
                port, "POST", "/prompt", json_body={"prompt": workflow}
            )
            if not result.get("success"):
                return result

            data = result.get("data", {})
            prompt_id = data.get("prompt_id")

            # Stash workflow for auto-cataloging when get_result is called
            if prompt_id:
                self._last_workflows[prompt_id] = workflow
                # Consume pending parent lineage from comfyui_prepare_input
                if hasattr(self, '_next_parent_id') and self._next_parent_id:
                    self._pending_parent_ids[prompt_id] = self._next_parent_id
                    self._next_parent_id = None

            return {
                "success": True,
                "prompt_id": prompt_id,
                "instance_id": instance_id,
                "port": port,
                "parameters": {
                    "checkpoint": checkpoint, "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width, "height": height, "steps": steps,
                    "cfg": cfg, "seed": seed, "sampler_name": sampler_name,
                    "scheduler": scheduler,
                },
                "message": f"Image generation queued (prompt_id: {prompt_id}). "
                           f"Use comfyui_await_result(instance_id=\"{instance_id}\", prompt_id=\"{prompt_id}\") "
                           f"to wait for completion (recommended — blocks until done). "
                           f"Once complete, use comfyui_prepare_input to chain the output to another workflow.",
            }
        except Exception as e:
            logger.error(f"comfyui_generate_image failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_get_result(self, instance_id: str,
                                  prompt_id: str) -> Dict[str, Any]:
        """Get the result of an image generation job."""
        err = await self._check_api()
        if err:
            return err

        try:
            port = await self._get_instance_port(instance_id)
            if not port:
                return {"success": False,
                        "error": f"Instance '{instance_id}' not found or has no port"}

            result = await self._instance_request(port, "GET", f"/history/{prompt_id}")
            if not result.get("success"):
                return result

            data = result.get("data", {})

            if not data or prompt_id not in data:
                return {
                    "success": True,
                    "status": "running",
                    "message": "Image is still being generated. Try again in a few seconds.",
                }

            entry = data[prompt_id]
            status_info = entry.get("status", {})
            outputs = entry.get("outputs", {})

            if status_info.get("status_str") == "error":
                messages = status_info.get("messages", [])
                error_msg = "; ".join(str(m) for m in messages) if messages else "Unknown error"
                return {"success": False, "status": "failed", "error": error_msg}

            # Extract all media from outputs (images, videos/gifs, audio)
            images = []
            for node_id, node_output in outputs.items():
                # ComfyUI uses different keys for different media types:
                # "images" → image outputs (SaveImage, PreviewImage)
                # "gifs"   → animated outputs (VHS/AnimateDiff video, animated WebP/GIF)
                # "audio"  → audio outputs (SaveAudio nodes)
                for output_key in ("images", "gifs", "audio"):
                    if output_key not in node_output:
                        continue
                    for item in node_output[output_key]:
                        filename = item.get("filename", "")
                        subfolder = item.get("subfolder", "")
                        item_type = item.get("type", "output")
                        view_url = f"http://127.0.0.1:{port}/view?filename={filename}&type={item_type}"
                        if subfolder:
                            view_url += f"&subfolder={subfolder}"

                        # Detect media type from extension
                        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                        video_exts = {"mp4", "webm", "gif", "mkv", "avi", "mov"}
                        audio_exts = {"wav", "mp3", "flac", "ogg", "m4a", "aac"}
                        if ext in video_exts:
                            media_type = "video"
                        elif ext in audio_exts:
                            media_type = "audio"
                        else:
                            media_type = "image"

                        images.append({
                            "filename": filename,
                            "subfolder": subfolder,
                            "url": view_url,
                            "media_type": media_type,
                            "disk_path": f"modules/comfyui/comfyui/output/{filename}",
                        })

            # Auto-catalog the generation
            generation_id = None
            if images and self.catalog:
                try:
                    workflow_json = self._last_workflows.pop(prompt_id, None)
                    parent_id = self._pending_parent_ids.pop(prompt_id, None)
                    output_dir = str(self.manager.module_dir / "comfyui" / "output") if self.manager else ""

                    generation_id = self.catalog.record_generation(
                        prompt_id=prompt_id,
                        instance_id=instance_id,
                        instance_port=port,
                        workflow_json=workflow_json or {},
                        images=images,
                        parent_id=parent_id,
                        comfyui_dir=output_dir,
                    )
                    for img in images:
                        img["generation_id"] = generation_id
                except Exception as cat_err:
                    logger.warning(f"Failed to catalog generation: {cat_err}")

            # Build chaining hint
            chain_hint = ""
            if images:
                first_file = images[0].get("filename", "")
                if first_file:
                    chain_hint = (
                        f" To use in another workflow, call "
                        f"comfyui_prepare_input(source=\"{first_file}\", instance_id=\"{instance_id}\") "
                        f"to copy it to the input/ directory, then pass the returned filename "
                        f"as image_path in comfyui_build_workflow overrides."
                    )

            result_dict = {
                "success": True,
                "status": "completed",
                "prompt_id": prompt_id,
                "images": images,
                "count": len(images),
                "message": f"Generation complete. {len(images)} file(s) produced." + chain_hint,
            }
            if generation_id:
                result_dict["generation_id"] = generation_id
                result_dict["message"] = (
                    f"Generation complete. {len(images)} file(s) cataloged "
                    f"(ID: {generation_id[:8]}...)." + chain_hint
                )
            return result_dict
        except Exception as e:
            logger.error(f"comfyui_get_result failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_await_result(
        self, instance_id: str, prompt_id: str,
        poll_interval: int = 10, timeout: int = 600,
    ) -> Dict[str, Any]:
        """Block until a ComfyUI generation completes, polling internally."""
        poll_interval = max(5, min(60, poll_interval))
        timeout = max(30, min(1800, timeout))
        start = time.time()

        try:
            while True:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    return {
                        "success": False,
                        "status": "timeout",
                        "error": f"Timed out after {int(elapsed)}s waiting for prompt {prompt_id}. "
                                 f"The generation may still be running — use comfyui_get_result to check.",
                        "instance_id": instance_id,
                        "prompt_id": prompt_id,
                    }

                result = await self.comfyui_get_result(instance_id, prompt_id)

                status = result.get("status", "")
                if status == "completed":
                    # comfyui_get_result already includes chaining hints in its message
                    result["elapsed_seconds"] = int(time.time() - start)
                    return result

                if status == "failed":
                    result["elapsed_seconds"] = int(time.time() - start)
                    return result

                if not result.get("success"):
                    # API error, not just "still running"
                    return result

                # Still running — sleep then retry
                await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            return {
                "success": False,
                "status": "aborted",
                "error": "Await cancelled (agent aborted by user)",
                "instance_id": instance_id,
                "prompt_id": prompt_id,
                "elapsed_seconds": int(time.time() - start),
            }
        except Exception as e:
            logger.error(f"comfyui_await_result failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_install_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """Install custom node packs."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", "/api/nodes/install",
                                             json={"node_ids": node_ids})
            job_id = data.get("job_id")
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Installation started for {len(node_ids)} node pack(s). "
                           f"Use comfyui_job_status with job_id '{job_id}' to track progress.",
            }
        except Exception as e:
            logger.error(f"comfyui_install_nodes failed: {e}")
            return {"success": False, "error": str(e)}

    # ======================== Workflow Building Tools ========================

    async def comfyui_describe_nodes(
        self,
        class_types: List[str] = None,
        category: str = None,
        search: str = None,
        instance_id: str = None,
    ) -> Dict[str, Any]:
        """Discover available ComfyUI node types and ready-made workflow templates."""
        from backend.comfyui_workflow_templates import (
            get_node_catalog, get_node_info, list_ready_made, COMFYUI_NODE_CATALOG,
        )

        try:
            # Normalize: models often send comma-separated strings instead of arrays
            if isinstance(class_types, str):
                class_types = [c.strip() for c in class_types.split(",") if c.strip()]

            # Special: list templates
            if class_types and class_types == ["templates"]:
                templates = list_ready_made()
                return {
                    "success": True,
                    "templates": templates,
                    "count": len(templates),
                    "hint": "Use comfyui_build_workflow(template_id='...', overrides={...}) to build any template.",
                }

            # Special: full catalog overview
            if class_types and class_types == ["all"]:
                # Group by category
                by_category = {}
                for name, info in COMFYUI_NODE_CATALOG.items():
                    cat = info.get("category", "other")
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append({
                        "class_type": name,
                        "description": info.get("description", ""),
                        "outputs": info.get("outputs", []),
                    })

                templates = list_ready_made()
                return {
                    "success": True,
                    "node_catalog": by_category,
                    "total_nodes": len(COMFYUI_NODE_CATALOG),
                    "templates": templates,
                    "template_count": len(templates),
                    "hint": "Use class_types=['NodeName'] for full schema. Use template_id with comfyui_build_workflow for easy generation.",
                }

            # Category or search filter
            if category or search:
                nodes = get_node_catalog(category=category, search=search)

                # If search returned nothing from static catalog, query live instance
                if not nodes and search and instance_id:
                    live_nodes = await self._query_object_info_search(instance_id, search)
                    if live_nodes:
                        return {
                            "success": True,
                            "nodes": live_nodes,
                            "count": len(live_nodes),
                            "source": "live_instance",
                            "hint": "These nodes were found on the running instance (not in static catalog).",
                        }

                return {
                    "success": True,
                    "nodes": nodes,
                    "count": len(nodes),
                }

            # Specific class_types requested
            if class_types:
                # Handle comma-separated string (models often send strings instead of arrays)
                if isinstance(class_types, str):
                    class_types = [c.strip() for c in class_types.split(",") if c.strip()]

                results = {}
                not_found = []

                for ct in class_types:
                    # Try static catalog first
                    info = get_node_info(ct)
                    if info:
                        results[ct] = info
                        continue

                    # Try live instance query if instance_id provided
                    if instance_id:
                        live_info = await self._query_object_info(instance_id, ct)
                        if live_info:
                            results[ct] = live_info
                            continue

                    not_found.append(ct)

                response = {"success": True, "nodes": results, "count": len(results)}
                if not_found:
                    response["not_found"] = not_found
                    if not instance_id:
                        response["hint"] = (
                            f"Node(s) {not_found} not in static catalog. "
                            "Provide instance_id to query a running instance's /object_info."
                        )
                return response

            # No filters — return overview
            templates = list_ready_made()
            categories = {}
            for name, info in COMFYUI_NODE_CATALOG.items():
                cat = info.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1

            return {
                "success": True,
                "catalog_summary": categories,
                "total_nodes": len(COMFYUI_NODE_CATALOG),
                "templates": templates,
                "hint": "Use class_types=['all'] for full catalog, category='sampler' to filter, class_types=['templates'] for templates. For custom nodes (LTX, IP-Adapter, etc.), use search='keyword' with instance_id to query the live instance.",
            }

        except Exception as e:
            logger.error(f"comfyui_describe_nodes failed: {e}")
            return {"success": False, "error": str(e)}

    async def _query_object_info(self, instance_id: str, class_type: str) -> Optional[Dict]:
        """Query a running instance's /object_info for a specific node type."""
        try:
            port = await self._get_instance_port(instance_id)
            if not port:
                return None

            # Check cache
            cached = self._object_info_cache.get(port, class_type)
            if cached is not None:
                return cached

            result = await self._instance_request(port, "GET", f"/object_info/{class_type}")
            if not result.get("success"):
                return None

            data = result.get("data", {})
            node_data = data.get(class_type, data)
            if not node_data:
                return None

            # Simplify the raw /object_info format
            simplified = self._simplify_object_info(class_type, node_data)
            self._object_info_cache.put(port, class_type, simplified)
            return simplified

        except Exception as e:
            logger.error(f"_query_object_info failed: {e}")
            return None

    async def _query_object_info_search(self, instance_id: str, search: str) -> Dict[str, Dict]:
        """Query a running instance's full /object_info and search by keyword."""
        try:
            port = await self._get_instance_port(instance_id)
            if not port:
                return {}

            # Check cache for the full catalog
            cached = self._object_info_cache.get(port, "_all_nodes")
            if cached is not None:
                all_nodes = cached
            else:
                result = await self._instance_request(port, "GET", "/object_info")
                if not result.get("success"):
                    return {}
                all_nodes = result.get("data", {})
                self._object_info_cache.put(port, "_all_nodes", all_nodes)

            # Search through all node types
            search_lower = search.lower()
            matches = {}
            for class_type, raw in all_nodes.items():
                display = (raw.get("display_name") or "").lower()
                cat = (raw.get("category") or "").lower()
                desc = (raw.get("description") or "").lower()
                if (search_lower in class_type.lower()
                        or search_lower in display
                        or search_lower in cat
                        or search_lower in desc):
                    matches[class_type] = self._simplify_object_info(class_type, raw)
                    if len(matches) >= 50:  # cap results
                        break

            return matches

        except Exception as e:
            logger.error(f"_query_object_info_search failed: {e}")
            return {}

    def _simplify_object_info(self, class_type: str, raw: Dict) -> Dict:
        """Simplify raw /object_info data into our catalog format."""
        inputs = {}
        required_inputs = raw.get("input", {}).get("required", {})
        optional_inputs = raw.get("input", {}).get("optional", {})

        for name, spec in required_inputs.items():
            inputs[name] = self._parse_input_spec(name, spec, required=True)
        for name, spec in optional_inputs.items():
            inputs[name] = self._parse_input_spec(name, spec, required=False)

        return {
            "category": raw.get("category", "unknown"),
            "description": raw.get("description", raw.get("display_name", class_type)),
            "inputs": inputs,
            "outputs": raw.get("output", []),
            "output_names": raw.get("output_name", []),
            "source": "live_instance",
        }

    def _parse_input_spec(self, name: str, spec: Any, required: bool) -> Dict:
        """Parse a single input spec from /object_info format."""
        if isinstance(spec, list) and len(spec) >= 1:
            type_info = spec[0]
            extra = spec[1] if len(spec) > 1 else {}

            if isinstance(type_info, list):
                # COMBO type — list of options
                return {
                    "type": "COMBO",
                    "required": required,
                    "options": type_info[:20],  # Truncate long lists
                    "default": extra.get("default") if isinstance(extra, dict) else None,
                }
            elif isinstance(type_info, str):
                result = {"type": type_info, "required": required}
                if isinstance(extra, dict):
                    if "default" in extra:
                        result["default"] = extra["default"]
                    if "min" in extra:
                        result["min"] = extra["min"]
                    if "max" in extra:
                        result["max"] = extra["max"]
                return result

        return {"type": "unknown", "required": required}

    async def comfyui_build_workflow(
        self,
        template_id: str = None,
        native_template: str = None,
        instance_id: str = None,
        overrides: Dict = None,
        execute: bool = True,
        nodes: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Build a ComfyUI workflow from a template, native template, or custom node specs.
        When execute=True and instance_id is provided, immediately executes on that instance."""
        from backend.comfyui_workflow_templates import (
            build_ready_made, build_comfyui_workflow, READY_MADE_WORKFLOWS,
        )

        try:
            if not template_id and not native_template and not nodes:
                available = list(READY_MADE_WORKFLOWS.keys())
                return {
                    "success": False,
                    "error": "Provide template_id (built-in), native_template (from comfyui_list_templates), or nodes (custom).",
                    "available_templates": available,
                    "hint": "Use comfyui_list_templates to find native templates like 'ltxv_text_to_video'.",
                }

            # Native template: convert graph-format JSON to API format
            if native_template:
                if not instance_id:
                    return {
                        "success": False,
                        "error": "instance_id is required for native template conversion (needed for /object_info).",
                    }
                result = await self._build_from_native_template(
                    native_template, instance_id, overrides or {}
                )
                if execute and result.get("success") and instance_id:
                    return await self._execute_built_workflow(result, instance_id)
                return result

            workflow = None
            warnings = []
            source = None

            if template_id:
                if template_id not in READY_MADE_WORKFLOWS:
                    # Check if it's actually a native template name
                    native_path = self._find_native_template(template_id)
                    if native_path and instance_id:
                        result = await self._build_from_native_template(
                            template_id, instance_id, overrides or {}
                        )
                        if execute and result.get("success") and instance_id:
                            return await self._execute_built_workflow(result, instance_id)
                        return result
                    available = list(READY_MADE_WORKFLOWS.keys())
                    hint = ""
                    if native_path:
                        hint = (f" '{template_id}' is a native template. "
                                f"Use native_template='{template_id}' with instance_id to convert it.")
                    return {
                        "success": False,
                        "error": f"Unknown built-in template: {template_id}. Available: {available}",
                        "hint": hint or "Use comfyui_list_templates to find native templates.",
                    }

                # Build from built-in template
                workflow, warnings = build_ready_made(template_id, overrides or {})
                source = template_id

            elif nodes:
                # Build from custom node specs
                workflow, warnings = build_comfyui_workflow(nodes, validate=True)
                source = "custom"

            if workflow is None:
                return {"success": False, "error": "No workflow was built."}

            # Auto-execute if requested
            if execute and instance_id:
                err = await self._check_api()
                if err:
                    return err
                port = await self._get_instance_port(instance_id)
                if not port:
                    return {"success": False,
                            "error": f"Instance '{instance_id}' not found or has no port"}
                result = await self._instance_request(
                    port, "POST", "/prompt", json_body={"prompt": workflow}
                )
                if not result.get("success"):
                    return result
                data = result.get("data", {})
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    self._last_workflows[prompt_id] = workflow
                    if hasattr(self, '_next_parent_id') and self._next_parent_id:
                        self._pending_parent_ids[prompt_id] = self._next_parent_id
                        self._next_parent_id = None
                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "instance_id": instance_id,
                    "template": source,
                    "node_count": len(workflow),
                    "validation_warnings": warnings if warnings else None,
                    "message": f"Built '{source}' workflow ({len(workflow)} nodes) and queued on '{instance_id}'. "
                               f"Use comfyui_await_result(instance_id=\"{instance_id}\", prompt_id=\"{prompt_id}\") "
                               f"to wait for completion (recommended — blocks until done).",
                }

            # Build only (no execute)
            return {
                "success": True,
                "workflow": workflow,
                "template": source,
                "node_count": len(workflow),
                "validation_warnings": warnings if warnings else None,
                "message": f"Built '{source}' workflow with {len(workflow)} nodes. "
                           f"Use comfyui_execute_workflow to run it.",
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"comfyui_build_workflow failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_built_workflow(self, build_result: Dict, instance_id: str) -> Dict[str, Any]:
        """Execute a workflow that was built by _build_from_native_template."""
        workflow = build_result.get("workflow")
        if not workflow:
            return build_result  # No workflow to execute, return as-is

        err = await self._check_api()
        if err:
            return err
        port = await self._get_instance_port(instance_id)
        if not port:
            return {"success": False,
                    "error": f"Instance '{instance_id}' not found or has no port"}
        result = await self._instance_request(
            port, "POST", "/prompt", json_body={"prompt": workflow}
        )
        if not result.get("success"):
            return result
        data = result.get("data", {})
        prompt_id = data.get("prompt_id")
        if prompt_id:
            self._last_workflows[prompt_id] = workflow
            if hasattr(self, '_next_parent_id') and self._next_parent_id:
                self._pending_parent_ids[prompt_id] = self._next_parent_id
                self._next_parent_id = None
        return {
            "success": True,
            "prompt_id": prompt_id,
            "instance_id": instance_id,
            "node_count": len(workflow),
            "message": f"Built and queued workflow ({len(workflow)} nodes) on '{instance_id}'. "
                       f"Use comfyui_await_result(instance_id=\"{instance_id}\", prompt_id=\"{prompt_id}\") "
                       f"to wait for completion (recommended — blocks until done).",
        }

    def _find_native_template(self, template_name: str) -> Optional[str]:
        """Find a native template file by name. Returns file path or None."""
        import os
        if self.manager and hasattr(self.manager, 'module_dir'):
            comfyui_dir = str(self.manager.module_dir)
        else:
            comfyui_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "modules", "comfyui"
            )
        site_packages = os.path.join(comfyui_dir, "python_embedded", "Lib", "site-packages")
        if not os.path.isdir(site_packages):
            return None

        for pkg_dir in os.listdir(site_packages):
            if not pkg_dir.startswith("comfyui_workflow_templates"):
                continue
            templates_dir = os.path.join(site_packages, pkg_dir, "templates")
            if not os.path.isdir(templates_dir):
                continue
            target = os.path.join(templates_dir, f"{template_name}.json")
            if os.path.isfile(target):
                return target
        return None

    async def _build_from_native_template(
        self, template_name: str, instance_id: str, overrides: Dict
    ) -> Dict[str, Any]:
        """Load a native graph-format template and convert to API format."""
        import json as json_mod

        template_path = self._find_native_template(template_name)
        if not template_path:
            return {
                "success": False,
                "error": f"Native template '{template_name}' not found. Use comfyui_list_templates to see available templates.",
            }

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                graph = json_mod.load(f)
        except Exception as e:
            return {"success": False, "error": f"Failed to read template: {e}"}

        if "nodes" not in graph or "links" not in graph:
            return {"success": False, "error": "Template is not in graph format (missing nodes/links)."}

        # Convert graph to API format
        workflow, warnings = await self._convert_graph_to_api(graph, instance_id)
        if not workflow:
            return {
                "success": False,
                "error": "Conversion failed — no executable nodes found.",
                "warnings": warnings,
            }

        # Apply overrides: match by input name across all nodes
        applied_overrides = []
        if overrides:
            # Identify positive/negative CLIPTextEncode by tracing links
            positive_clip_id = negative_clip_id = None
            for node_id, node_data in workflow.items():
                inputs = node_data.get("inputs", {})
                if "positive" in inputs and isinstance(inputs["positive"], list):
                    positive_clip_id = inputs["positive"][0]
                if "negative" in inputs and isinstance(inputs["negative"], list):
                    negative_clip_id = inputs["negative"][0]

            for key, value in overrides.items():
                applied = False

                # Special handling for prompt/negative_prompt using link tracing
                if key == "prompt" and positive_clip_id and positive_clip_id in workflow:
                    node_data = workflow[positive_clip_id]
                    if "text" in node_data.get("inputs", {}):
                        node_data["inputs"]["text"] = value
                        applied_overrides.append(f"prompt → node {positive_clip_id} CLIPTextEncode:text")
                        applied = True
                elif key == "negative_prompt" and negative_clip_id and negative_clip_id in workflow:
                    node_data = workflow[negative_clip_id]
                    if "text" in node_data.get("inputs", {}):
                        node_data["inputs"]["text"] = value
                        applied_overrides.append(f"negative_prompt → node {negative_clip_id} CLIPTextEncode:text")
                        applied = True
                elif key in ("checkpoint", "model"):
                    for nid, nd in workflow.items():
                        if nd.get("class_type") == "CheckpointLoaderSimple":
                            nd["inputs"]["ckpt_name"] = value
                            applied_overrides.append(f"{key} → CheckpointLoaderSimple:ckpt_name")
                            applied = True
                            break
                elif key == "clip_name":
                    for nid, nd in workflow.items():
                        if nd.get("class_type") == "CLIPLoader":
                            nd["inputs"]["clip_name"] = value
                            applied_overrides.append(f"clip_name → CLIPLoader:clip_name")
                            applied = True
                            break

                # Generic: search all nodes for matching input name
                if not applied:
                    for node_id, node_data in workflow.items():
                        inputs = node_data.get("inputs", {})
                        if key in inputs and not isinstance(inputs[key], list):
                            inputs[key] = value
                            applied_overrides.append(f"{key} → {node_data['class_type']}:{key}")
                            applied = True
                            break

        return {
            "success": True,
            "workflow": workflow,
            "template": template_name,
            "source": "native_template",
            "node_count": len(workflow),
            "applied_overrides": applied_overrides if applied_overrides else None,
            "warnings": warnings if warnings else None,
            "message": f"Converted native template '{template_name}' ({len(workflow)} nodes). "
                       f"Use comfyui_execute_workflow to run it.",
        }

    async def _convert_graph_to_api(
        self, graph: Dict, instance_id: str
    ) -> tuple:
        """Convert ComfyUI graph format to API format using /object_info."""
        warnings = []

        # 1. Build link map: link_id -> (from_node_id, from_slot)
        link_map = {}
        for link in graph.get("links", []):
            if len(link) >= 4:
                link_id, from_node, from_slot = link[0], link[1], link[2]
                link_map[link_id] = (from_node, from_slot)

        # 2. Get /object_info for all node types
        port = await self._get_instance_port(instance_id)
        object_info_cache = {}
        if port:
            # Try to get full /object_info (cached)
            cached = self._object_info_cache.get(port, "_all_nodes")
            if cached:
                object_info_cache = cached
            else:
                result = await self._instance_request(port, "GET", "/object_info")
                if result.get("success"):
                    object_info_cache = result.get("data", {})
                    self._object_info_cache.put(port, "_all_nodes", object_info_cache)

        # Known non-executable node types
        SKIP_TYPES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "GroupNode"}
        # Known frontend-only widget values (e.g. seed control_after_generate)
        SEED_CONTROL_VALUES = {"randomize", "fixed", "increment", "decrement", "last"}

        # 3. Convert each node
        api_workflow = {}
        for node in graph.get("nodes", []):
            class_type = node.get("type", "")
            if class_type in SKIP_TYPES or not class_type:
                continue

            node_id = str(node["id"])
            inputs = {}

            # Map linked inputs from graph connections
            linked_names = set()
            for inp in node.get("inputs", []):
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    from_node, from_slot = link_map[link_id]
                    inputs[inp["name"]] = [str(from_node), from_slot]
                    linked_names.add(inp["name"])

            # Map widgets_values to non-linked inputs using /object_info
            info = object_info_cache.get(class_type)
            widgets = node.get("widgets_values", [])

            if info and widgets:
                required = info.get("input", {}).get("required", {})
                optional = info.get("input", {}).get("optional", {})

                # Collect non-linked inputs in definition order
                non_linked = []
                for name, spec in required.items():
                    if name not in linked_names:
                        input_type = spec[0] if isinstance(spec, list) else str(spec)
                        non_linked.append((name, input_type))
                for name, spec in optional.items():
                    if name not in linked_names:
                        input_type = spec[0] if isinstance(spec, list) else str(spec)
                        non_linked.append((name, input_type))

                # Walk widgets_values and assign to non-linked inputs
                wi = 0  # widget value index
                ni = 0  # non-linked input index
                while wi < len(widgets) and ni < len(non_linked):
                    val = widgets[wi]
                    name, input_type = non_linked[ni]

                    # Skip frontend-only values (seed control_after_generate)
                    if isinstance(val, str) and val.lower() in SEED_CONTROL_VALUES:
                        wi += 1
                        continue

                    # Skip link-type inputs that somehow aren't linked
                    if input_type in ("MODEL", "CONDITIONING", "LATENT", "VAE",
                                      "CLIP", "CLIP_VISION", "IMAGE", "MASK",
                                      "SAMPLER", "SIGMAS", "CONTROL_NET",
                                      "UPSCALE_MODEL", "VIDEO"):
                        ni += 1
                        continue

                    inputs[name] = val
                    wi += 1
                    ni += 1

            elif widgets and not info:
                warnings.append(f"No /object_info for '{class_type}' — widgets_values not mapped")

            api_workflow[node_id] = {
                "class_type": class_type,
                "inputs": inputs,
            }

        return api_workflow, warnings

    async def comfyui_execute_workflow(
        self,
        instance_id: str,
        workflow: Optional[Dict] = None,
        workflow_json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute any ComfyUI workflow on an instance."""
        err = await self._check_api()
        if err:
            return err

        try:
            workflow = workflow or workflow_json
            if not workflow:
                return {
                    "success": False,
                    "error": "Missing workflow payload. Provide 'workflow' or 'workflow_json'."
                }

            port = await self._get_instance_port(instance_id)
            if not port:
                return {"success": False,
                        "error": f"Instance '{instance_id}' not found or has no port"}

            result = await self._instance_request(
                port, "POST", "/prompt", json_body={"prompt": workflow}
            )
            if not result.get("success"):
                result["hint"] = (
                    "Workflow validation failed. Instead of hand-writing workflow JSON, "
                    "use comfyui_build_workflow(template_id='...', instance_id='...', overrides={...}) "
                    "which handles correct node wiring and model filenames automatically. "
                    "Available templates: txt2img, img2img, upscale, inpaint, txt2img_hires, "
                    "controlnet_pose, svd_img2video, ltxv_img2video."
                )
                return result

            data = result.get("data", {})
            prompt_id = data.get("prompt_id")

            # Stash workflow for auto-cataloging when get_result is called
            if prompt_id:
                self._last_workflows[prompt_id] = workflow
                if hasattr(self, '_next_parent_id') and self._next_parent_id:
                    self._pending_parent_ids[prompt_id] = self._next_parent_id
                    self._next_parent_id = None

            return {
                "success": True,
                "prompt_id": prompt_id,
                "instance_id": instance_id,
                "port": port,
                "message": f"Workflow queued (prompt_id: {prompt_id}). "
                           f"Use comfyui_get_result(instance_id=\"{instance_id}\", prompt_id=\"{prompt_id}\") to check when complete.",
            }
        except Exception as e:
            logger.error(f"comfyui_execute_workflow failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Generation Catalog Tools ====================

    async def comfyui_search_generations(
        self, query: str = "", checkpoint: str = "", tags: str = "",
        favorite: bool = False, min_rating: int = 0, limit: int = 10,
    ) -> Dict[str, Any]:
        """Search the generation catalog."""
        if not self.catalog:
            return {"success": False, "error": "Media catalog not initialized."}

        try:
            results, total = self.catalog.search_generations(
                query=query or None,
                checkpoint=checkpoint or None,
                tags=tags or None,
                favorite=favorite or None,
                min_rating=min_rating or None,
                limit=limit,
            )

            # Compact representation for the LLM
            generations = []
            for gen in results:
                generations.append({
                    "id": gen["id"],
                    "prompt_text": (gen.get("prompt_text") or "")[:120],
                    "checkpoint": gen.get("checkpoint"),
                    "seed": gen.get("seed"),
                    "steps": gen.get("steps"),
                    "cfg": gen.get("cfg"),
                    "width": gen.get("width"),
                    "height": gen.get("height"),
                    "workflow_type": gen.get("workflow_type"),
                    "rating": gen.get("rating", 0),
                    "favorite": bool(gen.get("favorite")),
                    "tags": gen.get("tags", ""),
                    "title": gen.get("title", ""),
                    "created_at": gen.get("created_at"),
                })

            return {
                "success": True,
                "generations": generations,
                "total": total,
                "message": f"Found {total} generation(s), showing {len(generations)}.",
            }
        except Exception as e:
            logger.error(f"comfyui_search_generations failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_prepare_input(
        self, source: str, instance_id: str, file_index: int = 0,
    ) -> Dict[str, Any]:
        """Prepare an image as input for a ComfyUI workflow."""
        err = self._check_manager()
        if err:
            return err

        try:
            port = await self._get_instance_port(instance_id)
            if not port:
                return {"success": False,
                        "error": f"Instance '{instance_id}' not found or has no port"}

            # Resolve the ComfyUI input directory
            comfyui_dir = self.manager.module_dir / "comfyui"
            if not comfyui_dir.exists():
                return {"success": False, "error": "ComfyUI directory not found. Is ComfyUI installed?"}
            input_dir = comfyui_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)

            filename = None

            # Check if source looks like a UUID (generation ID)
            import re
            uuid_pattern = re.compile(
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
            )

            if uuid_pattern.match(source):
                # Source is a generation ID
                if not self.catalog:
                    return {"success": False, "error": "Media catalog not initialized."}
                filename = self.catalog.copy_to_input(source, str(input_dir), file_index)
                if not filename:
                    return {"success": False,
                            "error": f"Generation '{source}' not found or file index {file_index} out of range"}
                # Track lineage: the NEXT queued generation should be a child of this one.
                # Store as "next parent" — consumed by comfyui_generate_image or build_workflow.
                self._next_parent_id = source

            elif source.startswith("http://") or source.startswith("https://"):
                # Source is a URL
                if not self.catalog:
                    return {"success": False, "error": "Media catalog not initialized."}
                filename = self.catalog.download_to_input(source, str(input_dir))
                if not filename:
                    return {"success": False, "error": f"Failed to download image from URL"}

            else:
                # Source is a file path or filename
                src_path = Path(source)
                if not src_path.exists():
                    # Try resolving as a filename in the ComfyUI output directory
                    output_dir = comfyui_dir / "output"
                    output_path = output_dir / source
                    if output_path.exists():
                        src_path = output_path
                    else:
                        # Also try ComfyUI input dir (already there)
                        input_path = input_dir / source
                        if input_path.exists():
                            # Already in input dir, just return the filename
                            return {
                                "success": True,
                                "filename": source,
                                "input_dir": str(input_dir),
                                "message": f"Image '{source}' already in ComfyUI input/. "
                                           f"Now use comfyui_build_workflow with overrides={{\"image_path\": \"{source}\"}} "
                                           f"to build the next workflow step (e.g. template_id='svd_img2video' or 'img2img').",
                            }
                        return {"success": False,
                                "error": f"File not found: {source}. Checked: {source}, {output_path}, {input_path}"}
                import shutil
                dest = input_dir / src_path.name
                shutil.copy2(str(src_path), str(dest))
                filename = src_path.name

            return {
                "success": True,
                "filename": filename,
                "input_dir": str(input_dir),
                "message": f"Image prepared as '{filename}' in ComfyUI input/. "
                           f"Now use comfyui_build_workflow with overrides={{\"image_path\": \"{filename}\"}} "
                           f"to build the next workflow step (e.g. template_id='svd_img2video' or 'img2img').",
            }
        except Exception as e:
            logger.error(f"comfyui_prepare_input failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Node Management Tools ====================

    async def comfyui_list_node_packs(self) -> Dict[str, Any]:
        """Browse the curated registry of installable custom node packs."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/nodes/registry")
            nodes = data if isinstance(data, list) else data.get("nodes", data)
            return {
                "success": True,
                "node_packs": nodes,
                "count": len(nodes) if isinstance(nodes, list) else None,
            }
        except Exception as e:
            logger.error(f"comfyui_list_node_packs failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_list_installed_nodes(self) -> Dict[str, Any]:
        """List all currently installed custom node packs."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/nodes/installed")
            nodes = data if isinstance(data, list) else data.get("nodes", data)
            return {
                "success": True,
                "installed_nodes": nodes,
                "count": len(nodes) if isinstance(nodes, list) else None,
            }
        except Exception as e:
            logger.error(f"comfyui_list_installed_nodes failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_update_nodes(self) -> Dict[str, Any]:
        """Update all installed custom node packs."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", "/api/nodes/update-all")
            job_id = data.get("job_id")
            if job_id:
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": f"Node update started (job: {job_id}). "
                               f"Use comfyui_job_status to track progress.",
                }
            return {"success": True, "result": data}
        except Exception as e:
            logger.error(f"comfyui_update_nodes failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_remove_node(self, node_name: str) -> Dict[str, Any]:
        """Remove an installed custom node pack."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("DELETE", f"/api/nodes/{node_name}")
            return {
                "success": True,
                "result": data,
                "message": f"Node pack '{node_name}' removed.",
            }
        except Exception as e:
            logger.error(f"comfyui_remove_node failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Instance Management Tools ====================

    async def comfyui_remove_instance(self, instance_id: str) -> Dict[str, Any]:
        """Remove a ComfyUI instance permanently."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("DELETE", f"/api/instances/{instance_id}")
            return {
                "success": True,
                "result": data,
                "message": f"Instance {instance_id} removed.",
            }
        except Exception as e:
            logger.error(f"comfyui_remove_instance failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_start_all_instances(self) -> Dict[str, Any]:
        """Start all ComfyUI instances at once."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", "/api/instances/start-all")
            return {
                "success": True,
                "result": data,
                "message": "All instances starting.",
            }
        except Exception as e:
            logger.error(f"comfyui_start_all_instances failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_stop_all_instances(self) -> Dict[str, Any]:
        """Stop all running ComfyUI instances at once."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", "/api/instances/stop-all")
            return {
                "success": True,
                "result": data,
                "message": "All instances stopping.",
            }
        except Exception as e:
            logger.error(f"comfyui_stop_all_instances failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Model Categories ====================

    async def comfyui_model_categories(self) -> Dict[str, Any]:
        """List model folder categories."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/models/categories")
            categories = data if isinstance(data, list) else data.get("categories", data)
            return {
                "success": True,
                "categories": categories,
                "count": len(categories) if isinstance(categories, list) else None,
            }
        except Exception as e:
            logger.error(f"comfyui_model_categories failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Settings ====================

    async def comfyui_get_settings(self) -> Dict[str, Any]:
        """Get current ComfyUI module settings."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/settings")
            return {
                "success": True,
                "settings": data,
            }
        except Exception as e:
            logger.error(f"comfyui_get_settings failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_update_settings(self, settings: Dict) -> Dict[str, Any]:
        """Update ComfyUI module settings."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("PUT", "/api/settings", json=settings)
            return {
                "success": True,
                "result": data,
                "message": "Settings updated.",
            }
        except Exception as e:
            logger.error(f"comfyui_update_settings failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Update / Purge ====================

    async def comfyui_update_comfyui(self) -> Dict[str, Any]:
        """Update ComfyUI to the latest version via git pull."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", "/api/update")
            job_id = data.get("job_id")
            if job_id:
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": f"ComfyUI update started (job: {job_id}). "
                               f"Use comfyui_job_status to track progress.",
                }
            return {"success": True, "result": data}
        except Exception as e:
            logger.error(f"comfyui_update_comfyui failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_purge(self) -> Dict[str, Any]:
        """Purge ComfyUI installation (keeps models and Python)."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("POST", "/api/purge")
            return {
                "success": True,
                "result": data,
                "message": "ComfyUI purged. Models and Python environment kept. Re-install with comfyui_install.",
            }
        except Exception as e:
            logger.error(f"comfyui_purge failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== External ComfyUI ====================

    async def comfyui_manage_external(self, action: str,
                                       directory: str = None,
                                       name: str = None) -> Dict[str, Any]:
        """Manage external ComfyUI installations."""
        err = await self._check_api()
        if err:
            return err

        try:
            if action == "list":
                data = await self.manager.proxy("GET", "/api/comfyui/saved")
                return {
                    "success": True,
                    "saved_directories": data if isinstance(data, list) else data.get("saved", data),
                }

            if action == "add":
                if not directory:
                    return {"success": False, "error": "directory is required for 'add' action"}
                body = {"directory": directory}
                if name:
                    body["name"] = name
                data = await self.manager.proxy("POST", "/api/comfyui/saved", json=body)
                return {
                    "success": True,
                    "result": data,
                    "message": f"External directory added: {directory}",
                }

            if action == "remove":
                if not directory:
                    return {"success": False, "error": "directory is required for 'remove' action"}
                data = await self.manager.proxy("DELETE", "/api/comfyui/saved",
                                                 params={"directory": directory})
                return {
                    "success": True,
                    "result": data,
                    "message": f"External directory removed: {directory}",
                }

            if action == "switch":
                if not directory:
                    return {"success": False, "error": "directory is required for 'switch' action"}
                data = await self.manager.proxy("PUT", "/api/comfyui/target",
                                                 json={"directory": directory})
                return {
                    "success": True,
                    "result": data,
                    "message": f"Active target switched to: {directory}",
                }

            return {"success": False, "error": f"Unknown action: {action}. Use list, add, remove, or switch."}

        except Exception as e:
            logger.error(f"comfyui_manage_external failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== GPUs ====================

    async def comfyui_list_gpus(self) -> Dict[str, Any]:
        """List available GPUs with VRAM information."""
        err = await self._check_api()
        if err:
            return err

        try:
            data = await self.manager.proxy("GET", "/api/gpus")
            gpus = data if isinstance(data, list) else data.get("gpus", data)
            return {
                "success": True,
                "gpus": gpus,
                "count": len(gpus) if isinstance(gpus, list) else None,
            }
        except Exception as e:
            logger.error(f"comfyui_list_gpus failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_list_templates(
        self, category: str = "all", search: str = ""
    ) -> Dict[str, Any]:
        """List native ComfyUI workflow templates from installed template packages."""
        import os

        try:
            # Scan template packages in ComfyUI's site-packages
            if self.manager and hasattr(self.manager, 'module_dir'):
                comfyui_dir = str(self.manager.module_dir)
            else:
                comfyui_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "modules", "comfyui"
                )

            site_packages = os.path.join(comfyui_dir, "python_embedded", "Lib", "site-packages")
            if not os.path.isdir(site_packages):
                return {"success": False, "error": "ComfyUI site-packages not found. Is ComfyUI installed?"}

            # Validate category parameter
            valid_categories = {"all", "video", "image", "api", "other", "core"}
            if category not in valid_categories:
                # The agent likely meant to use 'search' instead of 'category'
                search = search or category  # Treat invalid category as search term
                category = "all"

            # Map package directories to categories
            pkg_category_map = {
                "comfyui_workflow_templates_media_video": "video",
                "comfyui_workflow_templates_media_image": "image",
                "comfyui_workflow_templates_media_api": "api",
                "comfyui_workflow_templates_media_other": "other",
                "comfyui_workflow_templates_core": "core",
            }

            templates = []
            for pkg_name, cat in pkg_category_map.items():
                if category != "all" and cat != category:
                    continue

                templates_dir = os.path.join(site_packages, pkg_name, "templates")
                if not os.path.isdir(templates_dir):
                    continue

                for fname in sorted(os.listdir(templates_dir)):
                    if not fname.endswith(".json"):
                        continue
                    name = fname[:-5]  # Strip .json

                    if search and search.lower() not in name.lower():
                        continue

                    # Determine if API-based (cloud) or local
                    is_api = name.startswith("api_")
                    has_preview = os.path.exists(
                        os.path.join(templates_dir, f"{name}-1.webp")
                    )

                    templates.append({
                        "name": name,
                        "category": cat,
                        "is_api": is_api,
                        "has_preview": has_preview,
                        "file": fname,
                        "package": pkg_name,
                    })

            # Also include our built-in ready-made templates
            from backend.comfyui_workflow_templates import list_ready_made
            built_in = list_ready_made()
            for t in built_in:
                if search and search.lower() not in t["id"].lower():
                    continue
                templates.append({
                    "name": t["id"],
                    "category": "built-in",
                    "is_api": False,
                    "description": t["description"],
                    "note": "Use comfyui_build_workflow(template_id='{}') to build".format(t["id"]),
                })

            # Scan custom_nodes for example workflows
            comfyui_root = os.path.join(comfyui_dir, "ComfyUI") if "ComfyUI" not in comfyui_dir else comfyui_dir
            custom_nodes_dir = os.path.join(comfyui_root, "custom_nodes")
            if os.path.isdir(custom_nodes_dir):
                for node_pkg in sorted(os.listdir(custom_nodes_dir)):
                    examples_dir = os.path.join(custom_nodes_dir, node_pkg, "example_workflows")
                    if not os.path.isdir(examples_dir):
                        continue
                    for fname in sorted(os.listdir(examples_dir)):
                        if not fname.endswith(".json"):
                            continue
                        name = fname[:-5]
                        if search and search.lower() not in name.lower():
                            continue
                        rel_path = os.path.join("custom_nodes", node_pkg, "example_workflows", fname)
                        templates.append({
                            "name": name,
                            "category": "example",
                            "is_api": False,
                            "source": node_pkg,
                            "file": fname,
                            "path": rel_path,
                            "note": f"Use comfyui_analyze_workflow(workflow_path='{rel_path}') to check required models",
                        })

            return {
                "success": True,
                "templates": templates,
                "count": len(templates),
                "hint": (
                    "Built-in templates: use comfyui_build_workflow(template_id=...). "
                    "Native/example templates: use comfyui_analyze_workflow(workflow_path=...) to check required models first, "
                    "then comfyui_build_workflow(native_template=...) or comfyui_execute_workflow(workflow_path=...) to run."
                ),
            }
        except Exception as e:
            logger.error(f"comfyui_list_templates failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Workflow analysis
    # ------------------------------------------------------------------ #

    async def comfyui_analyze_workflow(
        self,
        workflow_path: str = None,
        workflow_json: Dict = None,
        native_template: str = None,
        template_name: str = None,
    ) -> Dict[str, Any]:
        """Analyze a workflow JSON — extract all model references, compare against local."""
        import json as _json
        import os

        # Accept template name aliases
        if not workflow_path and (native_template or template_name):
            workflow_path = native_template or template_name

        # --- Load workflow ---
        if workflow_json:
            data = workflow_json if isinstance(workflow_json, dict) else _json.loads(workflow_json)
        elif workflow_path:
            # Resolve relative paths against ComfyUI dir
            comfyui_dir = ""
            if self.manager and hasattr(self.manager, "module_dir"):
                comfyui_dir = os.path.join(str(self.manager.module_dir), "ComfyUI")
            if not comfyui_dir or not os.path.isdir(comfyui_dir):
                comfyui_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "modules", "comfyui", "ComfyUI",
                )

            if not os.path.isabs(workflow_path):
                workflow_path = os.path.join(comfyui_dir, workflow_path)

            if not os.path.isfile(workflow_path):
                # Try resolving as a native template name (e.g. "ltxv_text_to_video")
                resolved = self._resolve_native_template(comfyui_dir, workflow_path)
                if resolved:
                    workflow_path = resolved
                else:
                    return {"success": False, "error": f"Workflow file not found: {workflow_path}. Use comfyui_list_templates(search='...') to find valid template paths."}
            with open(workflow_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        else:
            return {"success": False, "error": "Provide workflow_path or workflow_json"}

        # --- Extract models ---
        found = {}  # lowercase filename -> info dict

        def _add(filename, folder, node_type, url=None):
            key = filename.lower()
            if key not in found:
                entry = {"filename": filename, "folder": folder, "node_type": node_type}
                if url:
                    entry["url"] = url
                found[key] = entry

        # Strategy 1: Graph-format properties.models (richest — has URLs)
        for node in self._iter_graph_nodes(data):
            for m in node.get("properties", {}).get("models", []):
                name = m.get("name", "")
                if name:
                    _add(name, m.get("directory", "unknown"), node.get("type", ""), m.get("url"))

        # Strategy 2: API-format inputs with known field names
        api_data = data.get("prompt", data)  # hybrid files have "prompt" key
        if self._is_api_format(api_data):
            for _nid, node in api_data.items():
                if not isinstance(node, dict) or "class_type" not in node:
                    continue
                ct = node["class_type"]
                inputs = node.get("inputs", {})
                for field, value in inputs.items():
                    if not isinstance(value, str):
                        continue
                    if field in _INPUT_FIELD_MAP:
                        _add(value, _INPUT_FIELD_MAP[field], ct)
                    elif _is_model_filename(value) and "loader" in ct.lower():
                        _add(value, _infer_folder(ct), ct)

        # Strategy 3: Graph-format widgets_values on Loader nodes
        for node in self._iter_graph_nodes(data):
            ntype = node.get("type", "")
            if "loader" not in ntype.lower():
                continue
            for val in node.get("widgets_values", []):
                if isinstance(val, str) and _is_model_filename(val):
                    _add(val, _infer_folder(ntype), ntype)

        # --- Compare against local models ---
        installed_set = set()  # lowercase basenames
        api_ok = False
        try:
            local = await self.manager.proxy("GET", "/api/models/local")
            models_by_folder = local if isinstance(local, dict) else {}
            if "models" in models_by_folder and isinstance(models_by_folder["models"], dict):
                models_by_folder = models_by_folder["models"]
            for _folder, model_list in models_by_folder.items():
                if not isinstance(model_list, list):
                    continue
                for m in model_list:
                    name = m.get("name", "")
                    path = m.get("path", "")
                    if name:
                        installed_set.add(name.lower())
                    if path:
                        installed_set.add(os.path.basename(path).lower())
            api_ok = True
        except Exception:
            pass  # API not running — still return extraction results

        installed = []
        missing = []
        for _key, info in found.items():
            basename = info["filename"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            is_local = basename.lower() in installed_set
            (installed if is_local else missing).append(info)

        hint_parts = []
        if not api_ok:
            hint_parts.append("ComfyUI API not running — could not verify local models.")
        if missing:
            with_url = sum(1 for m in missing if m.get("url"))
            hint_parts.append(
                f"{len(missing)} model(s) missing. Use comfyui_search_models to find and comfyui_download_model to install."
            )
            if with_url:
                hint_parts.append(f"{with_url} have direct download URLs in the 'url' field.")
        else:
            hint_parts.append("All models are installed. The workflow is ready to run.")

        return {
            "success": True,
            "total_models": len(found),
            "installed_count": len(installed),
            "missing_count": len(missing),
            "installed": installed,
            "missing": missing,
            "hint": " ".join(hint_parts),
        }

    @staticmethod
    def _iter_graph_nodes(data):
        """Yield all nodes from a graph-format workflow, including subgraph/definition nodes."""
        for node in data.get("nodes", []):
            yield node
            for sub in node.get("nodes", []):
                yield sub
        # Native templates store loader nodes inside definitions.subgraphs
        for sg in data.get("definitions", {}).get("subgraphs", []):
            for node in sg.get("nodes", []):
                yield node
                for sub in node.get("nodes", []):
                    yield sub

    @staticmethod
    def _resolve_native_template(comfyui_dir: str, original_path: str) -> str:
        """Try to find a native template or example workflow matching a name."""
        import os
        # Extract the name from the path (strip directories and extension)
        name = os.path.basename(original_path)
        name = name.rsplit(".", 1)[0] if "." in name else name
        name_lower = name.lower().replace("-", "_").replace(" ", "_")

        # Search native template packages
        site_packages = os.path.join(
            os.path.dirname(comfyui_dir), "python_embedded", "Lib", "site-packages"
        )
        pkg_dirs = [
            "comfyui_workflow_templates_media_video",
            "comfyui_workflow_templates_media_image",
            "comfyui_workflow_templates_media_api",
            "comfyui_workflow_templates_media_other",
            "comfyui_workflow_templates_core",
        ]
        for pkg in pkg_dirs:
            templates_dir = os.path.join(site_packages, pkg, "templates")
            if not os.path.isdir(templates_dir):
                continue
            for fname in os.listdir(templates_dir):
                if not fname.endswith(".json"):
                    continue
                fbase = fname[:-5].lower().replace("-", "_").replace(" ", "_")
                if fbase == name_lower or name_lower in fbase or fbase in name_lower:
                    return os.path.join(templates_dir, fname)

        # Search custom_nodes example workflows
        custom_nodes_dir = os.path.join(comfyui_dir, "custom_nodes")
        if os.path.isdir(custom_nodes_dir):
            for node_pkg in os.listdir(custom_nodes_dir):
                examples_dir = os.path.join(custom_nodes_dir, node_pkg, "example_workflows")
                if not os.path.isdir(examples_dir):
                    continue
                for fname in os.listdir(examples_dir):
                    if not fname.endswith(".json"):
                        continue
                    fbase = fname[:-5].lower().replace("-", "_").replace(" ", "_")
                    if fbase == name_lower or name_lower in fbase or fbase in name_lower:
                        return os.path.join(examples_dir, fname)

        return ""

    @staticmethod
    def _is_api_format(data):
        """Check if data looks like API format (flat dict of node_id -> {class_type, inputs})."""
        if not isinstance(data, dict):
            return False
        return any(
            isinstance(v, dict) and "class_type" in v
            for v in data.values()
        )

    # ------------------------------------------------------------------ #
    # Pool tools (multi-instance dispatch)
    # ------------------------------------------------------------------ #

    def _check_pool(self) -> Optional[Dict[str, Any]]:
        """Return error dict if pool is not available."""
        if not self.pool:
            return {
                "success": False,
                "error": "ComfyUI pool not initialized. The pool requires the ComfyUI manager to be configured."
            }
        return None

    async def comfyui_pool_status(self) -> Dict[str, Any]:
        """Get pool overview: instances, queues, loaded checkpoints, active jobs."""
        err = self._check_pool()
        if err:
            return err
        try:
            return await self.pool.get_pool_status()
        except Exception as e:
            logger.error(f"comfyui_pool_status failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_pool_generate(self, checkpoint: str, prompt: str,
                                     negative_prompt: str = "blurry, low quality, distorted",
                                     width: int = None, height: int = None,
                                     steps: int = None, cfg: float = None,
                                     seed: int = -1, sampler_name: str = None,
                                     scheduler: str = None) -> Dict[str, Any]:
        """Submit a single generation job to the pool."""
        err = self._check_pool()
        if err:
            return err
        try:
            return await self.pool.submit_job(
                checkpoint=checkpoint, prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height, steps=steps,
                cfg=cfg, seed=seed, sampler_name=sampler_name,
                scheduler=scheduler,
            )
        except Exception as e:
            logger.error(f"comfyui_pool_generate failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_pool_batch(self, checkpoint: str, jobs: List,
                                  steps: int = None, cfg: float = None,
                                  width: int = None, height: int = None,
                                  sampler_name: str = None,
                                  scheduler: str = None) -> Dict[str, Any]:
        """Submit N jobs distributed across pool instances."""
        err = self._check_pool()
        if err:
            return err
        try:
            return await self.pool.submit_batch(
                checkpoint=checkpoint, jobs=jobs,
                steps=steps, cfg=cfg,
                width=width, height=height,
                sampler_name=sampler_name, scheduler=scheduler,
            )
        except Exception as e:
            logger.error(f"comfyui_pool_batch failed: {e}")
            return {"success": False, "error": str(e)}

    async def comfyui_pool_results(self, job_id: str) -> Dict[str, Any]:
        """Get results for a pool job or batch."""
        err = self._check_pool()
        if err:
            return err
        try:
            # Try as single job first, then as batch
            result = await self.pool.get_job_status(job_id)
            if result is not None:
                return result
            result = await self.pool.get_batch_status(job_id)
            if result is not None:
                return result
            return {"success": False, "error": f"Job or batch '{job_id}' not found."}
        except Exception as e:
            logger.error(f"comfyui_pool_results failed: {e}")
            return {"success": False, "error": str(e)}
