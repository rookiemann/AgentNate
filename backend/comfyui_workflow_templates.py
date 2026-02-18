"""
ComfyUI Workflow Templates Library

Contains a curated node catalog, ready-made workflow templates, and a
workflow builder — the ComfyUI equivalent of workflow_templates.py for n8n.

The agent uses this to:
1. Discover available node types (describe_nodes)
2. Build workflows from templates or custom node specs (build_workflow)
3. Execute them via comfyui_execute_workflow
"""

import random
from typing import Dict, List, Any, Optional, Tuple


# ============================================================
# COMFYUI DATA TYPE SYSTEM
# ============================================================

COMFYUI_DATA_TYPES = {
    "MODEL": "Diffusion model weights",
    "CLIP": "CLIP text encoder",
    "VAE": "Variational autoencoder",
    "CONDITIONING": "Encoded text conditioning",
    "LATENT": "Latent space tensor",
    "IMAGE": "Image tensor (B,H,W,C)",
    "MASK": "Mask tensor",
    "INT": "Integer value",
    "FLOAT": "Float value",
    "STRING": "Text string",
    "COMBO": "Selection from options",
    "CONTROL_NET": "ControlNet model",
    "CLIP_VISION": "CLIP vision model",
    "CLIP_VISION_OUTPUT": "CLIP vision output",
    "UPSCALE_MODEL": "Upscale model",
    "SIGMAS": "Noise schedule (sigma values for sampling)",
    "SAMPLER": "Sampler function object",
    "VIDEO": "Video tensor (frames)",
}


# ============================================================
# CURATED NODE CATALOG (~40 commonly-used nodes)
# ============================================================

COMFYUI_NODE_CATALOG = {
    # === LOADERS ===
    "CheckpointLoaderSimple": {
        "category": "loader",
        "description": "Load a checkpoint model file",
        "inputs": {
            "ckpt_name": {"type": "COMBO", "required": True, "description": "Checkpoint filename from models/checkpoints/"},
        },
        "outputs": ["MODEL", "CLIP", "VAE"],
    },
    "VAELoader": {
        "category": "loader",
        "description": "Load a standalone VAE model",
        "inputs": {
            "vae_name": {"type": "COMBO", "required": True, "description": "VAE filename from models/vae/"},
        },
        "outputs": ["VAE"],
    },
    "LoraLoader": {
        "category": "loader",
        "description": "Load a LoRA and apply it to MODEL and CLIP",
        "inputs": {
            "model": {"type": "MODEL", "required": True, "description": "Input model"},
            "clip": {"type": "CLIP", "required": True, "description": "Input CLIP"},
            "lora_name": {"type": "COMBO", "required": True, "description": "LoRA filename from models/loras/"},
            "strength_model": {"type": "FLOAT", "required": False, "default": 1.0, "description": "LoRA strength for model"},
            "strength_clip": {"type": "FLOAT", "required": False, "default": 1.0, "description": "LoRA strength for CLIP"},
        },
        "outputs": ["MODEL", "CLIP"],
    },
    "ControlNetLoader": {
        "category": "loader",
        "description": "Load a ControlNet model",
        "inputs": {
            "control_net_name": {"type": "COMBO", "required": True, "description": "ControlNet filename"},
        },
        "outputs": ["CONTROL_NET"],
    },
    "CLIPVisionLoader": {
        "category": "loader",
        "description": "Load a CLIP vision model",
        "inputs": {
            "clip_name": {"type": "COMBO", "required": True, "description": "CLIP vision model filename"},
        },
        "outputs": ["CLIP_VISION"],
    },
    "UpscaleModelLoader": {
        "category": "loader",
        "description": "Load an upscale model (ESRGAN, etc.)",
        "inputs": {
            "model_name": {"type": "COMBO", "required": True, "description": "Upscale model filename"},
        },
        "outputs": ["UPSCALE_MODEL"],
    },
    "UNETLoader": {
        "category": "loader",
        "description": "Load a standalone UNET model (for Flux, etc.)",
        "inputs": {
            "unet_name": {"type": "COMBO", "required": True, "description": "UNET filename"},
            "weight_dtype": {"type": "COMBO", "required": False, "default": "default",
                             "description": "Weight precision", "options": ["default", "fp8_e4m3fn", "fp8_e5m2"]},
        },
        "outputs": ["MODEL"],
    },
    "DualCLIPLoader": {
        "category": "loader",
        "description": "Load two CLIP models (for SDXL, Flux, etc.)",
        "inputs": {
            "clip_name1": {"type": "COMBO", "required": True, "description": "First CLIP model"},
            "clip_name2": {"type": "COMBO", "required": True, "description": "Second CLIP model"},
            "type": {"type": "COMBO", "required": False, "default": "sdxl",
                     "description": "CLIP type", "options": ["sdxl", "sd3", "flux"]},
        },
        "outputs": ["CLIP"],
    },

    # === CONDITIONING ===
    "CLIPTextEncode": {
        "category": "conditioning",
        "description": "Encode text into conditioning using CLIP",
        "inputs": {
            "text": {"type": "STRING", "required": True, "description": "Text prompt to encode"},
            "clip": {"type": "CLIP", "required": True, "description": "CLIP model"},
        },
        "outputs": ["CONDITIONING"],
    },
    "ConditioningCombine": {
        "category": "conditioning",
        "description": "Combine two conditionings (add together)",
        "inputs": {
            "conditioning_1": {"type": "CONDITIONING", "required": True},
            "conditioning_2": {"type": "CONDITIONING", "required": True},
        },
        "outputs": ["CONDITIONING"],
    },
    "ConditioningSetArea": {
        "category": "conditioning",
        "description": "Set area for regional conditioning (inpainting)",
        "inputs": {
            "conditioning": {"type": "CONDITIONING", "required": True},
            "width": {"type": "INT", "required": True}, "height": {"type": "INT", "required": True},
            "x": {"type": "INT", "required": True}, "y": {"type": "INT", "required": True},
            "strength": {"type": "FLOAT", "required": False, "default": 1.0},
        },
        "outputs": ["CONDITIONING"],
    },
    "ControlNetApplyAdvanced": {
        "category": "conditioning",
        "description": "Apply ControlNet to positive and negative conditioning",
        "inputs": {
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "control_net": {"type": "CONTROL_NET", "required": True},
            "image": {"type": "IMAGE", "required": True, "description": "Control image (pose, depth, etc.)"},
            "strength": {"type": "FLOAT", "required": False, "default": 1.0},
            "start_percent": {"type": "FLOAT", "required": False, "default": 0.0},
            "end_percent": {"type": "FLOAT", "required": False, "default": 1.0},
        },
        "outputs": ["CONDITIONING", "CONDITIONING"],
    },

    # === LATENT ===
    "EmptyLatentImage": {
        "category": "latent",
        "description": "Create an empty latent image (starting point for txt2img)",
        "inputs": {
            "width": {"type": "INT", "required": False, "default": 1024, "description": "Image width"},
            "height": {"type": "INT", "required": False, "default": 1024, "description": "Image height"},
            "batch_size": {"type": "INT", "required": False, "default": 1},
        },
        "outputs": ["LATENT"],
    },
    "VAEEncode": {
        "category": "latent",
        "description": "Encode an image into latent space (for img2img)",
        "inputs": {
            "pixels": {"type": "IMAGE", "required": True, "description": "Image to encode"},
            "vae": {"type": "VAE", "required": True},
        },
        "outputs": ["LATENT"],
    },
    "VAEDecode": {
        "category": "latent",
        "description": "Decode latent back to image",
        "inputs": {
            "samples": {"type": "LATENT", "required": True, "description": "Latent to decode"},
            "vae": {"type": "VAE", "required": True},
        },
        "outputs": ["IMAGE"],
    },
    "LatentUpscale": {
        "category": "latent",
        "description": "Upscale latent space (for hires fix)",
        "inputs": {
            "samples": {"type": "LATENT", "required": True},
            "upscale_method": {"type": "COMBO", "required": False, "default": "nearest-exact",
                               "options": ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]},
            "width": {"type": "INT", "required": True}, "height": {"type": "INT", "required": True},
            "crop": {"type": "COMBO", "required": False, "default": "disabled",
                     "options": ["disabled", "center"]},
        },
        "outputs": ["LATENT"],
    },

    # === SAMPLING ===
    "KSampler": {
        "category": "sampling",
        "description": "Standard sampler — the core generation node",
        "inputs": {
            "model": {"type": "MODEL", "required": True},
            "positive": {"type": "CONDITIONING", "required": True, "description": "Positive conditioning"},
            "negative": {"type": "CONDITIONING", "required": True, "description": "Negative conditioning"},
            "latent_image": {"type": "LATENT", "required": True, "description": "Input latent"},
            "seed": {"type": "INT", "required": False, "default": -1, "description": "Random seed (-1 = random)"},
            "steps": {"type": "INT", "required": False, "default": 20, "description": "Sampling steps"},
            "cfg": {"type": "FLOAT", "required": False, "default": 7.0, "description": "CFG scale / guidance"},
            "sampler_name": {"type": "COMBO", "required": False, "default": "euler",
                             "options": ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2",
                                         "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
                                         "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                                         "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                                         "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "uni_pc", "uni_pc_bh2"]},
            "scheduler": {"type": "COMBO", "required": False, "default": "normal",
                          "options": ["normal", "karras", "exponential", "sgm_uniform", "simple",
                                      "ddim_uniform", "beta"]},
            "denoise": {"type": "FLOAT", "required": False, "default": 1.0,
                        "description": "Denoise strength (1.0=full, <1.0=img2img)"},
        },
        "outputs": ["LATENT"],
    },
    "KSamplerAdvanced": {
        "category": "sampling",
        "description": "Advanced sampler with start/end step control",
        "inputs": {
            "model": {"type": "MODEL", "required": True},
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "latent_image": {"type": "LATENT", "required": True},
            "add_noise": {"type": "COMBO", "required": False, "default": "enable",
                          "options": ["enable", "disable"]},
            "noise_seed": {"type": "INT", "required": False, "default": -1},
            "steps": {"type": "INT", "required": False, "default": 20},
            "cfg": {"type": "FLOAT", "required": False, "default": 7.0},
            "sampler_name": {"type": "COMBO", "required": False, "default": "euler"},
            "scheduler": {"type": "COMBO", "required": False, "default": "normal"},
            "start_at_step": {"type": "INT", "required": False, "default": 0},
            "end_at_step": {"type": "INT", "required": False, "default": 10000},
            "return_with_leftover_noise": {"type": "COMBO", "required": False, "default": "disable",
                                           "options": ["disable", "enable"]},
        },
        "outputs": ["LATENT"],
    },

    # === IMAGE ===
    "SaveImage": {
        "category": "image",
        "description": "Save image to disk (output/ directory)",
        "inputs": {
            "images": {"type": "IMAGE", "required": True},
            "filename_prefix": {"type": "STRING", "required": False, "default": "ComfyUI",
                                "description": "Filename prefix"},
        },
        "outputs": [],
    },
    "PreviewImage": {
        "category": "image",
        "description": "Preview image (temporary, not saved to output/)",
        "inputs": {
            "images": {"type": "IMAGE", "required": True},
        },
        "outputs": [],
    },
    "LoadImage": {
        "category": "image",
        "description": "Load an image from ComfyUI's input/ directory",
        "inputs": {
            "image": {"type": "COMBO", "required": True, "description": "Image filename from input/ directory"},
        },
        "outputs": ["IMAGE", "MASK"],
    },
    "ImageScale": {
        "category": "image",
        "description": "Resize an image",
        "inputs": {
            "image": {"type": "IMAGE", "required": True},
            "upscale_method": {"type": "COMBO", "required": False, "default": "nearest-exact",
                               "options": ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]},
            "width": {"type": "INT", "required": True}, "height": {"type": "INT", "required": True},
            "crop": {"type": "COMBO", "required": False, "default": "disabled",
                     "options": ["disabled", "center"]},
        },
        "outputs": ["IMAGE"],
    },
    "ImageUpscaleWithModel": {
        "category": "image",
        "description": "Upscale image using an upscale model (ESRGAN, etc.)",
        "inputs": {
            "upscale_model": {"type": "UPSCALE_MODEL", "required": True},
            "image": {"type": "IMAGE", "required": True},
        },
        "outputs": ["IMAGE"],
    },
    "ImageInvert": {
        "category": "image",
        "description": "Invert image colors",
        "inputs": {
            "image": {"type": "IMAGE", "required": True},
        },
        "outputs": ["IMAGE"],
    },
    "ImageBatch": {
        "category": "image",
        "description": "Batch two images together",
        "inputs": {
            "image1": {"type": "IMAGE", "required": True},
            "image2": {"type": "IMAGE", "required": True},
        },
        "outputs": ["IMAGE"],
    },

    # === MASK ===
    "LoadImageMask": {
        "category": "mask",
        "description": "Load an image as a mask (uses alpha or color channel)",
        "inputs": {
            "image": {"type": "COMBO", "required": True, "description": "Image filename"},
            "channel": {"type": "COMBO", "required": False, "default": "alpha",
                        "options": ["alpha", "red", "green", "blue"]},
        },
        "outputs": ["MASK"],
    },
    "MaskComposite": {
        "category": "mask",
        "description": "Combine two masks",
        "inputs": {
            "destination": {"type": "MASK", "required": True},
            "source": {"type": "MASK", "required": True},
            "x": {"type": "INT", "required": False, "default": 0},
            "y": {"type": "INT", "required": False, "default": 0},
            "operation": {"type": "COMBO", "required": False, "default": "multiply",
                          "options": ["multiply", "add", "subtract", "and", "or", "xor"]},
        },
        "outputs": ["MASK"],
    },
    "InvertMask": {
        "category": "mask",
        "description": "Invert a mask",
        "inputs": {
            "mask": {"type": "MASK", "required": True},
        },
        "outputs": ["MASK"],
    },

    # === UTILITY / ADVANCED ===
    "CLIPSetLastLayer": {
        "category": "utility",
        "description": "Set CLIP skip (which layer to use)",
        "inputs": {
            "clip": {"type": "CLIP", "required": True},
            "stop_at_clip_layer": {"type": "INT", "required": False, "default": -1,
                                   "description": "Layer to stop at (-1 = last, -2 = second to last, etc.)"},
        },
        "outputs": ["CLIP"],
    },
    "ConditioningZeroOut": {
        "category": "utility",
        "description": "Zero out conditioning (for CFG=1 / no guidance workflows)",
        "inputs": {
            "conditioning": {"type": "CONDITIONING", "required": True},
        },
        "outputs": ["CONDITIONING"],
    },
    "FluxGuidance": {
        "category": "utility",
        "description": "Flux-specific guidance control",
        "inputs": {
            "conditioning": {"type": "CONDITIONING", "required": True},
            "guidance": {"type": "FLOAT", "required": False, "default": 3.5, "description": "Guidance strength"},
        },
        "outputs": ["CONDITIONING"],
    },
    "SetLatentNoiseMask": {
        "category": "utility",
        "description": "Set noise mask on latent (for inpainting)",
        "inputs": {
            "samples": {"type": "LATENT", "required": True},
            "mask": {"type": "MASK", "required": True},
        },
        "outputs": ["LATENT"],
    },
    "RepeatLatentBatch": {
        "category": "utility",
        "description": "Repeat a latent to create a batch",
        "inputs": {
            "samples": {"type": "LATENT", "required": True},
            "amount": {"type": "INT", "required": False, "default": 1},
        },
        "outputs": ["LATENT"],
    },

    # === ADVANCED LOADERS ===
    "CLIPLoader": {
        "category": "loader",
        "description": "Load a standalone CLIP/text encoder model (T5, etc.)",
        "inputs": {
            "clip_name": {"type": "COMBO", "required": True, "description": "CLIP model filename from models/text_encoders/"},
            "type": {"type": "COMBO", "required": False, "default": "stable_diffusion",
                     "description": "CLIP type", "options": ["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "cosmos", "wan2.1"]},
        },
        "outputs": ["CLIP"],
    },

    # === ADVANCED SAMPLING ===
    "KSamplerSelect": {
        "category": "sampling",
        "description": "Select a sampler by name (returns SAMPLER object for SamplerCustom)",
        "inputs": {
            "sampler_name": {"type": "COMBO", "required": True, "default": "euler",
                             "options": ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2",
                                         "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
                                         "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                                         "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                                         "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                                         "uni_pc", "uni_pc_bh2"]},
        },
        "outputs": ["SAMPLER"],
    },
    "SamplerCustom": {
        "category": "sampling",
        "description": "Custom sampler with explicit sigmas input (for advanced scheduling)",
        "inputs": {
            "model": {"type": "MODEL", "required": True},
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "sampler": {"type": "SAMPLER", "required": True, "description": "Sampler from KSamplerSelect"},
            "sigmas": {"type": "SIGMAS", "required": True, "description": "Noise schedule from a scheduler node"},
            "latent_image": {"type": "LATENT", "required": True},
            "add_noise": {"type": "BOOLEAN", "required": False, "default": True},
            "noise_seed": {"type": "INT", "required": False, "default": -1},
            "cfg": {"type": "FLOAT", "required": False, "default": 1.0},
        },
        "outputs": ["LATENT", "LATENT"],
    },

    # === LTX VIDEO ===
    "LTXVImgToVideo": {
        "category": "video",
        "description": "Prepare image for LTX Video generation (sets up conditioning + latent)",
        "inputs": {
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "vae": {"type": "VAE", "required": True},
            "image": {"type": "IMAGE", "required": True, "description": "Input image to animate"},
            "width": {"type": "INT", "required": False, "default": 768},
            "height": {"type": "INT", "required": False, "default": 512},
            "length": {"type": "INT", "required": False, "default": 97, "description": "Number of frames"},
            "batch_size": {"type": "INT", "required": False, "default": 1},
            "strength": {"type": "FLOAT", "required": False, "default": 0.15,
                         "description": "How much to deviate from input image (0.0=identical, 1.0=full generation)"},
        },
        "outputs": ["CONDITIONING", "CONDITIONING", "LATENT"],
    },
    "LTXVConditioning": {
        "category": "video",
        "description": "Apply LTX Video conditioning with frame rate",
        "inputs": {
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "frame_rate": {"type": "FLOAT", "required": False, "default": 25.0},
        },
        "outputs": ["CONDITIONING", "CONDITIONING"],
    },
    "LTXVScheduler": {
        "category": "video",
        "description": "Generate noise schedule (sigmas) for LTX Video sampling",
        "inputs": {
            "latent": {"type": "LATENT", "required": True},
            "steps": {"type": "INT", "required": False, "default": 30},
            "max_shift": {"type": "FLOAT", "required": False, "default": 2.05},
            "base_shift": {"type": "FLOAT", "required": False, "default": 0.95},
            "stretch": {"type": "FLOAT", "required": False, "default": 0.1},
            "invert": {"type": "BOOLEAN", "required": False, "default": True},
        },
        "outputs": ["SIGMAS"],
    },

    # === VIDEO OUTPUT ===
    "CreateVideo": {
        "category": "video",
        "description": "Convert image sequence to video",
        "inputs": {
            "images": {"type": "IMAGE", "required": True, "description": "Batch of frames"},
            "fps": {"type": "FLOAT", "required": False, "default": 24.0},
        },
        "outputs": ["VIDEO"],
    },
    "SaveVideo": {
        "category": "video",
        "description": "Save video to disk (output/ directory)",
        "inputs": {
            "video": {"type": "VIDEO", "required": True},
            "filename_prefix": {"type": "STRING", "required": False, "default": "ComfyUI",
                                "description": "Filename prefix for saved video"},
            "format": {"type": "COMBO", "required": False, "default": "auto",
                       "options": ["auto", "mp4"]},
            "codec": {"type": "COMBO", "required": False, "default": "auto",
                      "options": ["auto", "h264", "h265"]},
        },
        "outputs": [],
    },

    # === FLUX ===
    "CLIPTextEncodeFlux": {
        "category": "conditioning",
        "description": "Flux.1 dual text encoding (clip_l + t5xxl with guidance). For Flux 2 use CLIPTextEncode + FluxGuidance instead.",
        "inputs": {
            "clip": {"type": "CLIP", "required": True, "description": "CLIP model (from DualCLIPLoader type=flux)"},
            "clip_l": {"type": "STRING", "required": True, "description": "CLIP-L prompt text"},
            "t5xxl": {"type": "STRING", "required": True, "description": "T5-XXL prompt text"},
            "guidance": {"type": "FLOAT", "required": False, "default": 3.5, "description": "Guidance strength"},
        },
        "outputs": ["CONDITIONING"],
    },
    "Flux2Scheduler": {
        "category": "sampling",
        "description": "Generate noise schedule (sigmas) for Flux 2 sampling",
        "inputs": {
            "steps": {"type": "INT", "required": False, "default": 20, "description": "Number of sampling steps"},
            "width": {"type": "INT", "required": False, "default": 1024},
            "height": {"type": "INT", "required": False, "default": 1024},
        },
        "outputs": ["SIGMAS"],
    },
    "EmptyFlux2LatentImage": {
        "category": "latent",
        "description": "Create empty latent for Flux 2 generation",
        "inputs": {
            "width": {"type": "INT", "required": False, "default": 1024},
            "height": {"type": "INT", "required": False, "default": 1024},
            "batch_size": {"type": "INT", "required": False, "default": 1},
        },
        "outputs": ["LATENT"],
    },
    "ModelSamplingFlux": {
        "category": "utility",
        "description": "Apply Flux shift parameters to model for proper sampling",
        "inputs": {
            "model": {"type": "MODEL", "required": True},
            "max_shift": {"type": "FLOAT", "required": False, "default": 1.15},
            "base_shift": {"type": "FLOAT", "required": False, "default": 0.5},
            "width": {"type": "INT", "required": False, "default": 1024},
            "height": {"type": "INT", "required": False, "default": 1024},
        },
        "outputs": ["MODEL"],
    },

    # === WAN VIDEO ===
    "WanFirstLastFrameToVideo": {
        "category": "video",
        "description": "WAN 2.1 first-to-last frame video conditioning. Takes two images and creates video transition.",
        "inputs": {
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "vae": {"type": "VAE", "required": True},
            "width": {"type": "INT", "required": False, "default": 832},
            "height": {"type": "INT", "required": False, "default": 480},
            "length": {"type": "INT", "required": False, "default": 81, "description": "Number of video frames"},
            "batch_size": {"type": "INT", "required": False, "default": 1},
        },
        "optional_inputs": {
            "start_image": {"type": "IMAGE", "description": "First frame image"},
            "end_image": {"type": "IMAGE", "description": "Last frame image"},
        },
        "outputs": ["CONDITIONING", "CONDITIONING", "LATENT"],
    },
    "WanImageToVideo": {
        "category": "video",
        "description": "WAN 2.1 image-to-video conditioning. Animates a single image into video.",
        "inputs": {
            "positive": {"type": "CONDITIONING", "required": True},
            "negative": {"type": "CONDITIONING", "required": True},
            "vae": {"type": "VAE", "required": True},
            "width": {"type": "INT", "required": False, "default": 832},
            "height": {"type": "INT", "required": False, "default": 480},
            "length": {"type": "INT", "required": False, "default": 81},
            "batch_size": {"type": "INT", "required": False, "default": 1},
        },
        "optional_inputs": {
            "start_image": {"type": "IMAGE", "description": "Image to animate"},
        },
        "outputs": ["CONDITIONING", "CONDITIONING", "LATENT"],
    },

    # === QWEN IMAGE EDIT ===
    "TextEncodeQwenImageEdit": {
        "category": "conditioning",
        "description": "Qwen image edit text encoding with optional source image conditioning",
        "inputs": {
            "clip": {"type": "CLIP", "required": True},
            "prompt": {"type": "STRING", "required": True, "description": "Edit instruction prompt"},
        },
        "optional_inputs": {
            "vae": {"type": "VAE", "description": "VAE for image encoding"},
            "image": {"type": "IMAGE", "description": "Source image to edit"},
        },
        "outputs": ["CONDITIONING"],
    },
    "EmptyQwenImageLayeredLatentImage": {
        "category": "latent",
        "description": "Create layered latent for Qwen image generation/editing",
        "inputs": {
            "width": {"type": "INT", "required": False, "default": 640},
            "height": {"type": "INT", "required": False, "default": 640},
            "layers": {"type": "INT", "required": False, "default": 3, "description": "Number of layers (3 for edit)"},
            "batch_size": {"type": "INT", "required": False, "default": 1},
        },
        "outputs": ["LATENT"],
    },

    # === VIDEO SAVE (WEBP/ANIMATED) ===
    "SaveAnimatedWEBP": {
        "category": "video",
        "description": "Save image batch as animated WEBP",
        "inputs": {
            "images": {"type": "IMAGE", "required": True, "description": "Batch of frames"},
            "filename_prefix": {"type": "STRING", "required": False, "default": "ComfyUI"},
            "fps": {"type": "FLOAT", "required": False, "default": 6.0},
            "lossless": {"type": "BOOLEAN", "required": False, "default": False},
            "quality": {"type": "INT", "required": False, "default": 80},
            "method": {"type": "COMBO", "required": False, "default": "default",
                       "options": ["default"]},
        },
        "outputs": [],
    },
}


# ============================================================
# HELPER: MODEL DEFAULTS DETECTION
# ============================================================

def detect_model_defaults(checkpoint: str) -> Dict[str, Any]:
    """Auto-detect generation defaults based on checkpoint filename."""
    name_lower = checkpoint.lower()

    if "flux" in name_lower:
        return {
            "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple",
            "steps": 25, "width": 1024, "height": 1024,
        }
    elif "xl" in name_lower or "sdxl" in name_lower or "turbo" in name_lower:
        return {
            "cfg": 8.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "steps": 30, "width": 1024, "height": 1024,
        }
    else:
        return {
            "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "steps": 20, "width": 512, "height": 512,
        }


def _resolve_seed(seed: int) -> int:
    """Resolve seed (-1 means random)."""
    if seed == -1:
        return random.randint(0, 2**63 - 1)
    return seed


# ============================================================
# READY-MADE WORKFLOW TEMPLATES
# ============================================================

def _build_txt2img(p: Dict) -> Dict:
    """Standard text-to-image workflow."""
    defaults = detect_model_defaults(p.get("checkpoint", ""))
    seed = _resolve_seed(p.get("seed", -1))

    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p["checkpoint"]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", ""), "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, low quality, distorted"), "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": p.get("width", defaults["width"]),
                         "height": p.get("height", defaults["height"]),
                         "batch_size": p.get("batch_size", 1)}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": seed,
                         "steps": p.get("steps", defaults["steps"]),
                         "cfg": p.get("cfg", defaults["cfg"]),
                         "sampler_name": p.get("sampler_name", defaults["sampler_name"]),
                         "scheduler": p.get("scheduler", defaults["scheduler"]),
                         "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0],
                         "filename_prefix": p.get("filename_prefix", "AgentNate")}},
    }


def _build_img2img(p: Dict) -> Dict:
    """Image-to-image workflow."""
    defaults = detect_model_defaults(p.get("checkpoint", ""))
    seed = _resolve_seed(p.get("seed", -1))

    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p["checkpoint"]}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        "3": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", ""), "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, low quality, distorted"), "clip": ["1", 1]}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["3", 0], "seed": seed,
                         "steps": p.get("steps", defaults["steps"]),
                         "cfg": p.get("cfg", defaults["cfg"]),
                         "sampler_name": p.get("sampler_name", defaults["sampler_name"]),
                         "scheduler": p.get("scheduler", defaults["scheduler"]),
                         "denoise": p.get("denoise", 0.7)}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0],
                         "filename_prefix": p.get("filename_prefix", "AgentNate_img2img")}},
    }


def _build_upscale(p: Dict) -> Dict:
    """Upscale with model workflow."""
    return {
        "1": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        "2": {"class_type": "UpscaleModelLoader",
              "inputs": {"model_name": p.get("upscale_model", "RealESRGAN_x4plus.pth")}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "SaveImage",
              "inputs": {"images": ["3", 0],
                         "filename_prefix": p.get("filename_prefix", "AgentNate_upscale")}},
    }


def _build_inpaint(p: Dict) -> Dict:
    """Inpainting workflow."""
    defaults = detect_model_defaults(p.get("checkpoint", ""))
    seed = _resolve_seed(p.get("seed", -1))

    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p["checkpoint"]}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        "3": {"class_type": "LoadImageMask",
              "inputs": {"image": p["mask_path"], "channel": "alpha"}},
        "4": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "5": {"class_type": "SetLatentNoiseMask",
              "inputs": {"samples": ["4", 0], "mask": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", ""), "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, low quality, distorted"), "clip": ["1", 1]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["5", 0], "seed": seed,
                         "steps": p.get("steps", defaults["steps"]),
                         "cfg": p.get("cfg", defaults["cfg"]),
                         "sampler_name": p.get("sampler_name", defaults["sampler_name"]),
                         "scheduler": p.get("scheduler", defaults["scheduler"]),
                         "denoise": p.get("denoise", 0.8)}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0],
                          "filename_prefix": p.get("filename_prefix", "AgentNate_inpaint")}},
    }


def _build_txt2img_hires(p: Dict) -> Dict:
    """Text-to-image with hires fix (two-pass generation)."""
    defaults = detect_model_defaults(p.get("checkpoint", ""))
    seed = _resolve_seed(p.get("seed", -1))
    w = p.get("width", defaults["width"])
    h = p.get("height", defaults["height"])
    factor = p.get("upscale_factor", 1.5)

    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p["checkpoint"]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", ""), "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, low quality, distorted"), "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        # First pass
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": seed,
                         "steps": p.get("steps", defaults["steps"]),
                         "cfg": p.get("cfg", defaults["cfg"]),
                         "sampler_name": p.get("sampler_name", defaults["sampler_name"]),
                         "scheduler": p.get("scheduler", defaults["scheduler"]),
                         "denoise": 1.0}},
        # Upscale latent
        "6": {"class_type": "LatentUpscale",
              "inputs": {"samples": ["5", 0], "upscale_method": "nearest-exact",
                         "width": int(w * factor), "height": int(h * factor),
                         "crop": "disabled"}},
        # Second pass (hires fix)
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["6", 0], "seed": seed,
                         "steps": p.get("hires_steps", 15),
                         "cfg": p.get("cfg", defaults["cfg"]),
                         "sampler_name": p.get("sampler_name", defaults["sampler_name"]),
                         "scheduler": p.get("scheduler", defaults["scheduler"]),
                         "denoise": p.get("hires_denoise", 0.5)}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0],
                         "filename_prefix": p.get("filename_prefix", "AgentNate_hires")}},
    }


def _build_controlnet_pose(p: Dict) -> Dict:
    """ControlNet-guided generation (pose, depth, canny, etc.)."""
    defaults = detect_model_defaults(p.get("checkpoint", ""))
    seed = _resolve_seed(p.get("seed", -1))

    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p["checkpoint"]}},
        "2": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": p["controlnet_model"]}},
        "3": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", ""), "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, low quality, distorted"), "clip": ["1", 1]}},
        "6": {"class_type": "ControlNetApplyAdvanced",
              "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                         "control_net": ["2", 0], "image": ["3", 0],
                         "strength": p.get("strength", 0.8),
                         "start_percent": p.get("start_percent", 0.0),
                         "end_percent": p.get("end_percent", 1.0)}},
        "7": {"class_type": "EmptyLatentImage",
              "inputs": {"width": p.get("width", defaults["width"]),
                         "height": p.get("height", defaults["height"]),
                         "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1],
                         "latent_image": ["7", 0], "seed": seed,
                         "steps": p.get("steps", defaults["steps"]),
                         "cfg": p.get("cfg", defaults["cfg"]),
                         "sampler_name": p.get("sampler_name", defaults["sampler_name"]),
                         "scheduler": p.get("scheduler", defaults["scheduler"]),
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0],
                          "filename_prefix": p.get("filename_prefix", "AgentNate_controlnet")}},
    }


def _build_svd_img2vid(p: Dict) -> Dict:
    """Stable Video Diffusion XT image-to-video workflow.

    Uses ImageOnlyCheckpointLoader for svd_xt.safetensors which outputs
    MODEL, CLIP_VISION, and VAE.  SVD_img2vid_Conditioning creates the
    positive/negative/latent triple.  VideoLinearCFGGuidance wraps the
    model for smooth temporal coherence.
    """
    seed = _resolve_seed(p.get("seed", -1))

    return {
        # Load SVD checkpoint → MODEL (0), CLIP_VISION (1), VAE (2)
        "1": {"class_type": "ImageOnlyCheckpointLoader",
              "inputs": {"ckpt_name": p.get("checkpoint", "svd_xt.safetensors")}},
        # Load input image
        "2": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        # SVD conditioning → positive (0), negative (1), latent (2)
        "3": {"class_type": "SVD_img2vid_Conditioning",
              "inputs": {"clip_vision": ["1", 1],
                         "init_image": ["2", 0],
                         "vae": ["1", 2],
                         "width": p.get("width", 1024),
                         "height": p.get("height", 576),
                         "video_frames": p.get("video_frames", 25),
                         "motion_bucket_id": p.get("motion_bucket_id", 127),
                         "fps": p.get("fps", 6),
                         "augmentation_level": p.get("augmentation_level", 0.0)}},
        # Wrap model for video CFG guidance
        "4": {"class_type": "VideoLinearCFGGuidance",
              "inputs": {"model": ["1", 0],
                         "min_cfg": p.get("min_cfg", 1.0)}},
        # Sample
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0],
                         "positive": ["3", 0],
                         "negative": ["3", 1],
                         "latent_image": ["3", 2],
                         "seed": seed,
                         "steps": p.get("steps", 20),
                         "cfg": p.get("cfg", 2.5),
                         "sampler_name": p.get("sampler_name", "euler_ancestral"),
                         "scheduler": p.get("scheduler", "karras"),
                         "denoise": p.get("denoise", 1.0)}},
        # Decode latent frames → images
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        # Save as animated WEBP
        "7": {"class_type": "SaveAnimatedWEBP",
              "inputs": {"images": ["6", 0],
                         "filename_prefix": p.get("filename_prefix", "AgentNate_svd"),
                         "fps": p.get("output_fps", p.get("fps", 6.0)),
                         "lossless": False,
                         "quality": p.get("quality", 80),
                         "method": "default"}},
    }


def _build_ltxv_img2vid(p: Dict) -> Dict:
    """LTX Video image-to-video workflow.

    Uses separate CLIPLoader for T5 text encoder (the LTX Video checkpoint
    does NOT include a usable CLIP encoder — output slot 1 is null).
    """
    seed = _resolve_seed(p.get("seed", -1))

    return {
        # Loaders
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p.get("checkpoint", "ltx-2-19b-dev-fp8.safetensors")}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.get("text_encoder", "t5xxl_fp8_e4m3fn.safetensors"),
                         "type": "ltxv"}},
        "3": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        # Text encoding (using CLIPLoader output, NOT checkpoint CLIP)
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", "cinematic motion, smooth animation"),
                         "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, distorted, low quality, static"),
                         "clip": ["2", 0]}},
        # LTX Video conditioning pipeline
        "6": {"class_type": "LTXVImgToVideo",
              "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                         "vae": ["1", 2], "image": ["3", 0],
                         "width": p.get("width", 768),
                         "height": p.get("height", 512),
                         "length": p.get("length", 97),
                         "batch_size": 1,
                         "strength": p.get("strength", 0.15)}},
        "7": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["6", 0], "negative": ["6", 1],
                         "frame_rate": p.get("fps", 25)}},
        # Scheduling
        "8": {"class_type": "LTXVScheduler",
              "inputs": {"latent": ["6", 2],
                         "steps": p.get("steps", 30),
                         "max_shift": p.get("max_shift", 2.05),
                         "base_shift": p.get("base_shift", 0.95),
                         "stretch": p.get("stretch", 0.1),
                         "invert": True}},
        "9": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": p.get("sampler_name", "euler")}},
        # Sampling
        "10": {"class_type": "SamplerCustom",
               "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
                          "sampler": ["9", 0], "sigmas": ["8", 0],
                          "latent_image": ["6", 2],
                          "add_noise": True, "noise_seed": seed,
                          "cfg": p.get("cfg", 3.0)}},
        # Decode & save
        "11": {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "fps": p.get("fps", 24)}},
        "13": {"class_type": "SaveVideo",
               "inputs": {"video": ["12", 0],
                          "filename_prefix": p.get("filename_prefix", "AgentNate_ltxv"),
                          "format": "auto", "codec": "auto"}},
    }


def _build_flux2_txt2img(p: Dict) -> Dict:
    """Flux 2 text-to-image using UNET + single CLIPLoader(type=flux2) + SamplerCustom.

    Flux 2 uses a SINGLE text encoder (Mistral 3 Small 24B), NOT dual CLIP.
    The CLIPLoader with type='flux2' handles the Mistral model internally.
    Standard CLIPTextEncode + FluxGuidance replaces CLIPTextEncodeFlux.

    VRAM notes (RTX 3090 24GB):
    - UNET fp8: ~34GB on disk, loaded as fp8 tensors → ~12GB VRAM with fp8_e4m3fn
    - CLIP (Mistral 3 Small): ~17GB on disk → offload to CPU with clip_device="cpu"
    - VAE: ~321MB → trivial
    - Default: weight_dtype=fp8_e4m3fn + clip_device=cpu fits in 24GB
    """
    seed = _resolve_seed(p.get("seed", -1))
    w = p.get("width", 1024)
    h = p.get("height", 1024)
    # Accept both template param names AND ComfyUI node input names
    unet = p.get("unet") or p.get("unet_name") or "flux2_dev_fp8mixed.safetensors"
    clip = p.get("clip") or p.get("clip_name") or "mistral_3_small_flux2_fp8.safetensors"
    vae = p.get("vae") or p.get("vae_name") or "flux2-vae.safetensors"

    # Build CLIPLoader inputs — add device param only if not "default" (optional input)
    clip_inputs = {"clip_name": clip, "type": "flux2"}
    clip_device = p.get("clip_device", "cpu")
    if clip_device and clip_device != "default":
        clip_inputs["device"] = clip_device

    return {
        # Load Flux 2 UNET model (fp8 quantized by default to fit in VRAM)
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet,
                         "weight_dtype": p.get("weight_dtype", "fp8_e4m3fn")}},
        # Load single CLIP (Mistral 3 Small for Flux 2) — CPU offload by default
        "2": {"class_type": "CLIPLoader",
              "inputs": clip_inputs},
        # Load VAE
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae}},
        # Positive text encoding
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0],
                         "text": p.get("prompt", "")}},
        # Apply guidance to positive conditioning
        "5": {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["4", 0],
                         "guidance": p.get("guidance", 3.5)}},
        # Negative text encoding (empty for Flux 2)
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0],
                         "text": p.get("negative_prompt", "")}},
        # Apply Flux shift parameters
        "7": {"class_type": "ModelSamplingFlux",
              "inputs": {"model": ["1", 0],
                         "max_shift": p.get("max_shift", 1.15),
                         "base_shift": p.get("base_shift", 0.5),
                         "width": w, "height": h}},
        # Empty Flux 2 latent (128 channels)
        "8": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": w, "height": h, "batch_size": p.get("batch_size", 1)}},
        # Flux 2 scheduler → sigmas
        "9": {"class_type": "Flux2Scheduler",
              "inputs": {"steps": p.get("steps", 25), "width": w, "height": h}},
        # Sampler select
        "10": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": p.get("sampler_name", "euler")}},
        # Custom sampler with sigmas
        "11": {"class_type": "SamplerCustom",
               "inputs": {"model": ["7", 0], "positive": ["5", 0], "negative": ["6", 0],
                          "sampler": ["10", 0], "sigmas": ["9", 0],
                          "latent_image": ["8", 0],
                          "add_noise": True, "noise_seed": seed,
                          "cfg": p.get("cfg", 1.0)}},
        # Decode & save
        "12": {"class_type": "VAEDecode",
               "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0],
                          "filename_prefix": p.get("filename_prefix", "AgentNate_flux2")}},
    }


def _build_qwen_image_edit(p: Dict) -> Dict:
    """Qwen Image Edit workflow — edit a source image with a text prompt."""
    seed = _resolve_seed(p.get("seed", -1))
    w = p.get("width", 640)
    h = p.get("height", 640)

    return {
        # Load Qwen checkpoint → MODEL, CLIP, VAE
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": p.get("checkpoint", "qwen_image_edit_fp8.safetensors")}},
        # Load source image
        "2": {"class_type": "LoadImage",
              "inputs": {"image": p["image_path"]}},
        # Positive: Qwen edit encoding with source image
        "3": {"class_type": "TextEncodeQwenImageEdit",
              "inputs": {"clip": ["1", 1],
                         "prompt": p.get("prompt", ""),
                         "image": ["2", 0],
                         "vae": ["1", 2]}},
        # Negative conditioning
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, low quality, distorted"),
                         "clip": ["1", 1]}},
        # Layered latent for Qwen edit
        "5": {"class_type": "EmptyQwenImageLayeredLatentImage",
              "inputs": {"width": w, "height": h,
                         "layers": p.get("layers", 3),
                         "batch_size": 1}},
        # Sample
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
                         "latent_image": ["5", 0], "seed": seed,
                         "steps": p.get("steps", 25),
                         "cfg": p.get("cfg", 7.0),
                         "sampler_name": p.get("sampler_name", "euler"),
                         "scheduler": p.get("scheduler", "normal"),
                         "denoise": p.get("denoise", 1.0)}},
        # Decode & save
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0],
                         "filename_prefix": p.get("filename_prefix", "AgentNate_qwen_edit")}},
    }


def _build_wan_first_last_to_video(p: Dict) -> Dict:
    """WAN 2.1 first-to-last frame video generation.

    Takes two images (first frame + last frame) and generates a video
    transitioning between them using WAN's FirstLastFrameToVideo node.
    """
    seed = _resolve_seed(p.get("seed", -1))
    w = p.get("width", 1024)
    h = p.get("height", 576)

    return {
        # Load WAN UNET model
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": p.get("unet", "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"),
                         "weight_dtype": p.get("weight_dtype", "default")}},
        # Load WAN text encoder
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.get("text_encoder", "umt5_xxl_fp8_e4m3fn.safetensors"),
                         "type": "wan2.1"}},
        # Load WAN VAE
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": p.get("vae", "wan_2.1_vae.safetensors")}},
        # Load first frame image
        "4": {"class_type": "LoadImage",
              "inputs": {"image": p["first_frame"]}},
        # Load last frame image
        "5": {"class_type": "LoadImage",
              "inputs": {"image": p["last_frame"]}},
        # Text encoding
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("prompt", "smooth cinematic motion between frames"),
                         "clip": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p.get("negative_prompt", "blurry, distorted, low quality, static, jittery"),
                         "clip": ["2", 0]}},
        # WAN first-last frame conditioning → positive, negative, latent
        "8": {"class_type": "WanFirstLastFrameToVideo",
              "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                         "vae": ["3", 0],
                         "width": w, "height": h,
                         "length": p.get("length", 81),
                         "batch_size": 1,
                         "start_image": ["4", 0],
                         "end_image": ["5", 0]}},
        # Sample
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0],
                         "positive": ["8", 0], "negative": ["8", 1],
                         "latent_image": ["8", 2],
                         "seed": seed,
                         "steps": p.get("steps", 30),
                         "cfg": p.get("cfg", 5.0),
                         "sampler_name": p.get("sampler_name", "euler"),
                         "scheduler": p.get("scheduler", "normal"),
                         "denoise": 1.0}},
        # Decode frames
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        # Save as animated WEBP (ComfyUI standard video output)
        "11": {"class_type": "SaveAnimatedWEBP",
               "inputs": {"images": ["10", 0],
                          "filename_prefix": p.get("filename_prefix", "AgentNate_wan_video"),
                          "fps": p.get("fps", 16.0),
                          "lossless": False,
                          "quality": p.get("quality", 80),
                          "method": "default"}},
    }


READY_MADE_WORKFLOWS = {
    "txt2img": {
        "name": "Text to Image",
        "description": "Generate an image from a text prompt using a checkpoint model",
        "category": "generation",
        "required_models": ["checkpoints"],
        "overridable_params": {
            "checkpoint": {"type": "string", "required": True, "description": "Checkpoint filename"},
            "prompt": {"type": "string", "required": True, "description": "Positive text prompt"},
            "negative_prompt": {"type": "string", "default": "blurry, low quality, distorted"},
            "width": {"type": "int", "default": "auto"}, "height": {"type": "int", "default": "auto"},
            "steps": {"type": "int", "default": "auto"}, "cfg": {"type": "float", "default": "auto"},
            "seed": {"type": "int", "default": -1},
            "sampler_name": {"type": "string", "default": "auto"},
            "scheduler": {"type": "string", "default": "auto"},
        },
        "builder": _build_txt2img,
    },
    "img2img": {
        "name": "Image to Image",
        "description": "Transform an existing image guided by a text prompt",
        "category": "generation",
        "required_models": ["checkpoints"],
        "overridable_params": {
            "checkpoint": {"type": "string", "required": True},
            "image_path": {"type": "string", "required": True, "description": "Image filename in ComfyUI input/ dir"},
            "prompt": {"type": "string", "required": True},
            "negative_prompt": {"type": "string", "default": "blurry, low quality, distorted"},
            "denoise": {"type": "float", "default": 0.7, "description": "Denoise strength (0.0-1.0)"},
            "steps": {"type": "int", "default": "auto"}, "cfg": {"type": "float", "default": "auto"},
            "seed": {"type": "int", "default": -1},
        },
        "builder": _build_img2img,
    },
    "upscale": {
        "name": "Upscale Image",
        "description": "Upscale an image using an AI upscale model (ESRGAN, etc.)",
        "category": "upscaling",
        "required_models": ["upscale_models"],
        "overridable_params": {
            "image_path": {"type": "string", "required": True, "description": "Image filename in ComfyUI input/ dir"},
            "upscale_model": {"type": "string", "default": "RealESRGAN_x4plus.pth"},
        },
        "builder": _build_upscale,
    },
    "inpaint": {
        "name": "Inpainting",
        "description": "Fill in masked areas of an image guided by a text prompt",
        "category": "editing",
        "required_models": ["checkpoints"],
        "overridable_params": {
            "checkpoint": {"type": "string", "required": True},
            "image_path": {"type": "string", "required": True, "description": "Image filename"},
            "mask_path": {"type": "string", "required": True, "description": "Mask image filename (white=inpaint)"},
            "prompt": {"type": "string", "required": True},
            "negative_prompt": {"type": "string", "default": "blurry, low quality, distorted"},
            "denoise": {"type": "float", "default": 0.8},
            "steps": {"type": "int", "default": "auto"}, "cfg": {"type": "float", "default": "auto"},
            "seed": {"type": "int", "default": -1},
        },
        "builder": _build_inpaint,
    },
    "txt2img_hires": {
        "name": "Text to Image (Hires Fix)",
        "description": "Generate an image with a two-pass hires fix for sharper details at larger sizes",
        "category": "generation",
        "required_models": ["checkpoints"],
        "overridable_params": {
            "checkpoint": {"type": "string", "required": True},
            "prompt": {"type": "string", "required": True},
            "negative_prompt": {"type": "string", "default": "blurry, low quality, distorted"},
            "width": {"type": "int", "default": "auto"}, "height": {"type": "int", "default": "auto"},
            "steps": {"type": "int", "default": "auto"}, "cfg": {"type": "float", "default": "auto"},
            "seed": {"type": "int", "default": -1},
            "upscale_factor": {"type": "float", "default": 1.5, "description": "Upscale multiplier for hires pass"},
            "hires_denoise": {"type": "float", "default": 0.5, "description": "Denoise strength for hires pass"},
            "hires_steps": {"type": "int", "default": 15, "description": "Steps for hires pass"},
        },
        "builder": _build_txt2img_hires,
    },
    "controlnet_pose": {
        "name": "ControlNet Guided",
        "description": "Generate an image guided by a ControlNet control image (pose, depth, canny, etc.)",
        "category": "generation",
        "required_models": ["checkpoints", "controlnet"],
        "overridable_params": {
            "checkpoint": {"type": "string", "required": True},
            "controlnet_model": {"type": "string", "required": True, "description": "ControlNet model filename"},
            "image_path": {"type": "string", "required": True, "description": "Control image filename"},
            "prompt": {"type": "string", "required": True},
            "negative_prompt": {"type": "string", "default": "blurry, low quality, distorted"},
            "strength": {"type": "float", "default": 0.8, "description": "ControlNet strength"},
            "width": {"type": "int", "default": "auto"}, "height": {"type": "int", "default": "auto"},
            "steps": {"type": "int", "default": "auto"}, "cfg": {"type": "float", "default": "auto"},
            "seed": {"type": "int", "default": -1},
        },
        "builder": _build_controlnet_pose,
    },
    "ltxv_img2video": {
        "name": "LTX Video (Image to Video)",
        "description": "Animate a still image into a video using LTX Video 2. Requires ltx-2-19b checkpoint + T5 text encoder.",
        "category": "video",
        "required_models": ["checkpoints", "text_encoders"],
        "overridable_params": {
            "checkpoint": {"type": "string", "default": "ltx-2-19b-dev-fp8.safetensors",
                           "description": "LTX Video checkpoint in models/checkpoints/"},
            "text_encoder": {"type": "string", "default": "t5xxl_fp8_e4m3fn.safetensors",
                             "description": "T5 text encoder in models/text_encoders/"},
            "image_path": {"type": "string", "required": True,
                           "description": "Image filename in ComfyUI input/ directory"},
            "prompt": {"type": "string", "default": "cinematic motion, smooth animation",
                       "description": "Motion/animation description"},
            "negative_prompt": {"type": "string", "default": "blurry, distorted, low quality, static"},
            "width": {"type": "int", "default": 768}, "height": {"type": "int", "default": 512},
            "length": {"type": "int", "default": 97, "description": "Number of frames (97 ≈ 4s at 25fps)"},
            "fps": {"type": "int", "default": 25, "description": "Frame rate for conditioning and output"},
            "strength": {"type": "float", "default": 0.15,
                         "description": "How much to deviate from input (0.0=identical, 1.0=full generation)"},
            "steps": {"type": "int", "default": 30}, "cfg": {"type": "float", "default": 3.0},
            "seed": {"type": "int", "default": -1},
        },
        "builder": _build_ltxv_img2vid,
    },
    "svd_img2video": {
        "name": "SVD XT (Image to Video)",
        "description": "Animate a still image into a short video using Stable Video Diffusion XT. Fits in 12GB VRAM.",
        "category": "video",
        "required_models": ["checkpoints"],
        "overridable_params": {
            "checkpoint": {"type": "string", "default": "svd_xt.safetensors",
                           "description": "SVD checkpoint in models/checkpoints/"},
            "image_path": {"type": "string", "required": True,
                           "description": "Image filename in ComfyUI input/ directory (use comfyui_prepare_input first)"},
            "width": {"type": "int", "default": 1024}, "height": {"type": "int", "default": 576},
            "video_frames": {"type": "int", "default": 25, "description": "Number of video frames to generate"},
            "motion_bucket_id": {"type": "int", "default": 127,
                                 "description": "Motion intensity (1-1023, higher=more motion)"},
            "fps": {"type": "int", "default": 6, "description": "Conditioning FPS (affects motion speed)"},
            "augmentation_level": {"type": "float", "default": 0.0,
                                   "description": "Noise added to conditioning image (0.0-10.0)"},
            "min_cfg": {"type": "float", "default": 1.0, "description": "Minimum CFG for video guidance"},
            "steps": {"type": "int", "default": 20}, "cfg": {"type": "float", "default": 2.5},
            "seed": {"type": "int", "default": -1},
            "sampler_name": {"type": "string", "default": "euler_ancestral"},
            "scheduler": {"type": "string", "default": "karras"},
        },
        "builder": _build_svd_img2vid,
    },
    "flux2_txt2img": {
        "name": "Flux 2 Text to Image",
        "description": "Generate an image using Flux 2 Dev with single Mistral 3 Small text encoder. High quality 1024x1024 generation. WARNING: fp8mixed UNET is 34GB, requires >32GB VRAM or aggressive offloading. Use weight_dtype=fp8_e4m3fn and clip_device=cpu to reduce VRAM. For 24GB GPUs, consider a fully quantized (NF4/Q4) model variant.",
        "category": "generation",
        "required_models": ["diffusion_models", "text_encoders", "vae"],
        "overridable_params": {
            "unet": {"type": "string", "default": "flux2_dev_fp8mixed.safetensors",
                     "description": "Flux 2 UNET in models/diffusion_models/"},
            "clip": {"type": "string", "default": "mistral_3_small_flux2_fp8.safetensors",
                     "description": "Mistral 3 Small text encoder in models/text_encoders/"},
            "vae": {"type": "string", "default": "flux2-vae.safetensors",
                    "description": "Flux 2 VAE in models/vae/"},
            "prompt": {"type": "string", "required": True, "description": "Text prompt"},
            "negative_prompt": {"type": "string", "default": ""},
            "width": {"type": "int", "default": 1024}, "height": {"type": "int", "default": 1024},
            "steps": {"type": "int", "default": 25}, "cfg": {"type": "float", "default": 1.0},
            "guidance": {"type": "float", "default": 3.5, "description": "Flux guidance strength"},
            "seed": {"type": "int", "default": -1},
            "sampler_name": {"type": "string", "default": "euler"},
            "weight_dtype": {"type": "string", "default": "fp8_e4m3fn",
                             "description": "UNET precision: default, fp8_e4m3fn, fp8_e4m3fn_fast, fp8_e5m2"},
            "clip_device": {"type": "string", "default": "cpu",
                            "description": "Where to load text encoder: default (GPU) or cpu (RAM offload)"},
        },
        "builder": _build_flux2_txt2img,
    },
    "qwen_image_edit": {
        "name": "Qwen Image Edit",
        "description": "Edit an existing image using Qwen Image Edit model with text instruction. Creates modified versions of input images.",
        "category": "editing",
        "required_models": ["checkpoints"],
        "overridable_params": {
            "checkpoint": {"type": "string", "default": "qwen_image_edit_fp8.safetensors",
                           "description": "Qwen Image Edit checkpoint in models/checkpoints/"},
            "image_path": {"type": "string", "required": True,
                           "description": "Source image filename in ComfyUI input/ directory"},
            "prompt": {"type": "string", "required": True,
                       "description": "Edit instruction (e.g. 'change viewing angle to from below looking up')"},
            "negative_prompt": {"type": "string", "default": "blurry, low quality, distorted"},
            "width": {"type": "int", "default": 640}, "height": {"type": "int", "default": 640},
            "layers": {"type": "int", "default": 3, "description": "Number of latent layers"},
            "steps": {"type": "int", "default": 25}, "cfg": {"type": "float", "default": 7.0},
            "seed": {"type": "int", "default": -1},
        },
        "builder": _build_qwen_image_edit,
    },
    "wan_first_last_video": {
        "name": "WAN 2.1 First-Last Frame Video",
        "description": "Generate a video that transitions between two images (first frame to last frame) using WAN 2.1 14B. Great for morphing/transition effects.",
        "category": "video",
        "required_models": ["diffusion_models", "text_encoders", "vae"],
        "overridable_params": {
            "unet": {"type": "string", "default": "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
                     "description": "WAN UNET model in models/diffusion_models/"},
            "text_encoder": {"type": "string", "default": "umt5_xxl_fp8_e4m3fn.safetensors",
                             "description": "UMT5-XXL encoder in models/text_encoders/"},
            "vae": {"type": "string", "default": "wan_2.1_vae.safetensors",
                    "description": "WAN VAE in models/vae/"},
            "first_frame": {"type": "string", "required": True,
                            "description": "First frame image filename in ComfyUI input/ directory"},
            "last_frame": {"type": "string", "required": True,
                           "description": "Last frame image filename in ComfyUI input/ directory"},
            "prompt": {"type": "string", "default": "smooth cinematic motion between frames",
                       "description": "Motion/animation description"},
            "negative_prompt": {"type": "string", "default": "blurry, distorted, low quality, static, jittery"},
            "width": {"type": "int", "default": 1024}, "height": {"type": "int", "default": 576},
            "length": {"type": "int", "default": 81, "description": "Number of video frames (81 ≈ 5s at 16fps)"},
            "fps": {"type": "float", "default": 16.0, "description": "Output frame rate"},
            "steps": {"type": "int", "default": 30}, "cfg": {"type": "float", "default": 5.0},
            "seed": {"type": "int", "default": -1},
        },
        "builder": _build_wan_first_last_to_video,
    },
}


# ============================================================
# NODE CATALOG QUERY FUNCTIONS
# ============================================================

def get_node_catalog(
    category: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Dict]:
    """Get filtered view of the node catalog."""
    results = {}

    for name, info in COMFYUI_NODE_CATALOG.items():
        if category and info.get("category") != category:
            continue
        if search:
            search_lower = search.lower()
            if (search_lower not in name.lower()
                    and search_lower not in info.get("description", "").lower()
                    and search_lower not in info.get("category", "").lower()):
                continue
        results[name] = info

    return results


def get_node_info(class_type: str) -> Optional[Dict]:
    """Get schema for a single node type from the static catalog."""
    return COMFYUI_NODE_CATALOG.get(class_type)


def list_ready_made() -> List[Dict]:
    """List all ready-made workflow templates."""
    return [
        {
            "id": tid,
            "name": info["name"],
            "description": info["description"],
            "category": info["category"],
            "required_models": info["required_models"],
            "params": {
                k: {pk: pv for pk, pv in v.items() if pk != "description"}
                for k, v in info["overridable_params"].items()
            },
        }
        for tid, info in READY_MADE_WORKFLOWS.items()
    ]


def build_ready_made(template_id: str, overrides: Dict = None) -> Tuple[Dict, List[str]]:
    """
    Instantiate a ready-made workflow template.

    Returns (workflow_dict, warnings_list).
    """
    if template_id not in READY_MADE_WORKFLOWS:
        available = list(READY_MADE_WORKFLOWS.keys())
        raise ValueError(f"Unknown template: {template_id}. Available: {available}")

    template = READY_MADE_WORKFLOWS[template_id]
    params = dict(overrides or {})

    # Validate required params
    warnings = []
    for pname, pinfo in template["overridable_params"].items():
        if pinfo.get("required") and pname not in params:
            raise ValueError(f"Template '{template_id}' requires parameter: {pname}")

    workflow = template["builder"](params)
    return workflow, warnings


# ============================================================
# CUSTOM WORKFLOW BUILDER
# ============================================================

def build_comfyui_workflow(
    nodes_spec: List[Dict],
    validate: bool = True,
) -> Tuple[Dict, List[str]]:
    """
    Build a ComfyUI workflow from node specifications.

    Each node spec: {"id": "ref_name", "class_type": "NodeType", "inputs": {...}}

    Input values can be:
    - Primitives (str, int, float): used as-is
    - Connection refs: {"node_ref": "ref_name", "output_index": N}
      → converted to ["numeric_id", N] in ComfyUI API format

    Returns (workflow_dict, warnings_list).
    """
    if not nodes_spec:
        raise ValueError("No nodes provided")

    # Step 1: Assign sequential IDs and build ref map
    ref_to_id = {}
    for idx, spec in enumerate(nodes_spec, start=1):
        ref_name = spec.get("id", f"node_{idx}")
        numeric_id = str(idx)
        ref_to_id[ref_name] = numeric_id

    # Step 2: Build workflow with resolved connections
    workflow = {}
    warnings = []

    for idx, spec in enumerate(nodes_spec, start=1):
        class_type = spec.get("class_type")
        if not class_type:
            raise ValueError(f"Node spec #{idx} missing 'class_type'")

        numeric_id = str(idx)
        raw_inputs = spec.get("inputs", {})
        resolved_inputs = {}

        for input_name, input_value in raw_inputs.items():
            if isinstance(input_value, dict) and "node_ref" in input_value:
                # Connection reference — resolve to ["numeric_id", output_index]
                ref_name = input_value["node_ref"]
                output_index = input_value.get("output_index", 0)
                if ref_name not in ref_to_id:
                    available = list(ref_to_id.keys())
                    raise ValueError(
                        f"Node '{spec.get('id', idx)}' input '{input_name}' references "
                        f"unknown node '{ref_name}'. Available: {available}"
                    )
                resolved_inputs[input_name] = [ref_to_id[ref_name], output_index]
            elif input_name == "seed" and input_value == -1:
                resolved_inputs[input_name] = _resolve_seed(-1)
            else:
                resolved_inputs[input_name] = input_value

        workflow[numeric_id] = {
            "class_type": class_type,
            "inputs": resolved_inputs,
        }

    # Step 3: Validate connections
    if validate:
        warnings = _validate_workflow(workflow)

    return workflow, warnings


def _validate_workflow(workflow: Dict) -> List[str]:
    """Validate connection types in a built workflow. Returns list of warnings."""
    warnings = []

    for node_id, node in workflow.items():
        class_type = node["class_type"]
        node_schema = COMFYUI_NODE_CATALOG.get(class_type)

        if not node_schema:
            continue  # Can't validate unknown nodes

        for input_name, input_value in node["inputs"].items():
            if not isinstance(input_value, list) or len(input_value) != 2:
                continue  # Not a connection, skip

            source_id, output_index = input_value
            source_node = workflow.get(str(source_id))
            if not source_node:
                warnings.append(f"Node {node_id} ({class_type}): input '{input_name}' references non-existent node {source_id}")
                continue

            source_schema = COMFYUI_NODE_CATALOG.get(source_node["class_type"])
            if not source_schema:
                continue  # Can't validate unknown source

            source_outputs = source_schema.get("outputs", [])
            if output_index >= len(source_outputs):
                warnings.append(
                    f"Node {node_id} ({class_type}): input '{input_name}' uses output[{output_index}] "
                    f"of {source_node['class_type']}, but it only has {len(source_outputs)} outputs"
                )
                continue

            # Check type compatibility
            source_type = source_outputs[output_index]
            input_schema = node_schema.get("inputs", {}).get(input_name, {})
            expected_type = input_schema.get("type")

            if expected_type and source_type != expected_type:
                # Wildcard types don't need to match exactly
                if expected_type not in ("*", "ANY") and source_type not in ("*", "ANY"):
                    warnings.append(
                        f"Node {node_id} ({class_type}): input '{input_name}' expects {expected_type} "
                        f"but receives {source_type} from {source_node['class_type']}[{output_index}]"
                    )

    return warnings


def validate_comfyui_workflow(workflow: Dict) -> Tuple[bool, List[str]]:
    """Public validation function. Returns (is_valid, warnings)."""
    warnings = _validate_workflow(workflow)
    return len(warnings) == 0, warnings
