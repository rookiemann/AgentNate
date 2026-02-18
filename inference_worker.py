#!/usr/bin/env python3
"""
llama.cpp subprocess worker for AgentNate.

Handles model loading and chat inference in an isolated process.
Receives JSON commands via stdin, outputs JSON responses via stdout.

Features:
- Model loading with GPU layer control
- Vision model support (mmproj) with multiple handlers
- Auto-detection of model family and chat format
- Streaming chat generation
- Full sampling parameter support
- Reasoning token stripping
- Speculative decoding support
- Robust error handling
"""
import sys
import os
import json
import re
import traceback

# Ensure unbuffered output for proper streaming
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# CUDA optimization
os.environ["LLAMA_CUDA_FORCE_MMQ"] = "1"


def log(msg: str):
    """Log to stderr (doesn't interfere with JSON protocol on stdout)."""
    print(f"[WORKER] {msg}", file=sys.stderr, flush=True)


def send(data: dict):
    """Send JSON response to stdout."""
    print(json.dumps(data), flush=True)


def strip_reasoning_tokens(text: str) -> str:
    """
    Strip common reasoning/thinking tokens from model output.
    Many models output internal reasoning that users don't want to see.
    """
    patterns = [
        # Channel-based thinking (OSS models)
        r'<\|channel\|>analysis<\|message\|>.*?<\|end\|>',
        r'<\|channel\|>thinking<\|message\|>.*?<\|end\|>',
        r'<\|channel\|>internal<\|message\|>.*?<\|end\|>',
        r'<\|start\|>assistant<\|channel\|>.*?<\|channel\|>final<\|message\|>',

        # DeepSeek thinking
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',

        # Generic reasoning blocks
        r'<reasoning>.*?</reasoning>',
        r'<internal>.*?</internal>',
        r'\[THINKING\].*?\[/THINKING\]',
        r'\[INTERNAL\].*?\[/INTERNAL\]',

        # QwQ-style
        r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)

    return result.strip()


def detect_model_family(model_path: str, mmproj_path: str = None) -> dict:
    """
    Detect model family and return appropriate settings.
    Returns dict with 'chat_format', 'chat_handler_type', 'use_embedded_template' keys.
    """
    path_lower = model_path.lower()
    mmproj_lower = (mmproj_path or "").lower()
    combined = path_lower + " " + mmproj_lower

    result = {
        "chat_format": None,
        "chat_handler_type": None,  # 'llava15', 'moondream', 'gemma3', None
        "use_embedded_template": False,
        "model_family": "unknown",
        "notes": "",
        "is_vision": mmproj_path is not None,
    }

    # === VISION MODELS (check first - more specific) ===

    if "moondream" in combined:
        result["model_family"] = "moondream"
        result["chat_handler_type"] = "moondream"
        result["chat_format"] = "moondream"
        result["is_vision"] = True
        return result

    # Gemma 3 Vision
    if ("gemma-3" in combined or "gemma3" in combined) and mmproj_path:
        result["model_family"] = "gemma3-vision"
        result["chat_handler_type"] = "gemma3"  # Will fallback to llava15 if not available
        result["is_vision"] = True
        return result

    # LLaVA variants
    if any(x in combined for x in ["llava-v1.6", "llava-1.6", "llava1.6"]):
        result["model_family"] = "llava-1.6"
        result["chat_format"] = "llava-1-6"
        result["chat_handler_type"] = "llava15"
        result["is_vision"] = True
        return result

    if any(x in combined for x in ["llava-v1.5", "llava-1.5", "llava1.5"]):
        result["model_family"] = "llava-1.5"
        result["chat_format"] = "llava-1-5"
        result["chat_handler_type"] = "llava15"
        result["is_vision"] = True
        return result

    if "llava-llama-3" in combined or "llama3-llava" in combined:
        result["model_family"] = "llava-llama3"
        result["chat_format"] = "llama-3"
        result["chat_handler_type"] = "llava15"
        result["is_vision"] = True
        return result

    if "llava-phi-3" in combined or ("phi-3" in combined and mmproj_path):
        result["model_family"] = "llava-phi3"
        result["chat_handler_type"] = "llava15"
        result["is_vision"] = True
        return result

    if any(x in combined for x in ["llava-mistral", "bakllava"]) or ("llava" in combined and "mistral" in combined):
        result["model_family"] = "llava-mistral"
        result["chat_format"] = "mistral-instruct"
        result["chat_handler_type"] = "llava15"
        result["is_vision"] = True
        return result

    # Generic LLaVA (catch-all for vision)
    if "llava" in combined and mmproj_path:
        result["model_family"] = "llava-generic"
        result["chat_handler_type"] = "llava15"
        result["is_vision"] = True
        return result

    # Other vision model patterns
    if any(x in combined for x in ["cogvlm", "minicpm-v", "qwen-vl", "internvl", "yi-vl", "deepseek-vl", "pixtral", "molmo"]):
        result["model_family"] = "vision-generic"
        result["chat_handler_type"] = "llava15"  # Fallback handler
        result["is_vision"] = True
        return result

    # === TEXT-ONLY MODELS ===

    # Gemma 3 (text) - use embedded template, it's strict about format
    if "gemma-3" in combined or "gemma3" in combined:
        result["model_family"] = "gemma3"
        result["use_embedded_template"] = True
        result["notes"] = "Gemma 3 works best with embedded template"
        return result

    # Gemma 2 - can use 'gemma' format
    if "gemma-2" in combined or "gemma2" in combined:
        result["model_family"] = "gemma2"
        result["chat_format"] = "gemma"
        return result

    # Gemma 1
    if "gemma" in combined:
        result["model_family"] = "gemma"
        result["chat_format"] = "gemma"
        return result

    # Llama 3.x
    if any(x in combined for x in ["llama-3.3", "llama-3.2", "llama-3.1", "llama-3-", "llama3"]):
        result["model_family"] = "llama3"
        result["chat_format"] = "llama-3"
        return result

    # Llama 2
    if "llama-2" in combined or "llama2" in combined:
        result["model_family"] = "llama2"
        result["chat_format"] = "llama-2"
        return result

    # Qwen / Qwen2
    if "qwen" in combined:
        result["model_family"] = "qwen"
        result["chat_format"] = "qwen"
        return result

    # Mistral / Mixtral / Devstral
    if "mistral" in combined or "mixtral" in combined or "devstral" in combined:
        result["model_family"] = "mistral"
        result["chat_format"] = "mistral-instruct"
        return result

    # Phi-3 / Phi-4
    if "phi-4" in combined or "phi4" in combined:
        result["model_family"] = "phi4"
        result["use_embedded_template"] = True
        return result

    if "phi-3" in combined or "phi3" in combined:
        result["model_family"] = "phi3"
        result["use_embedded_template"] = True
        return result

    # DeepSeek
    if "deepseek" in combined:
        result["model_family"] = "deepseek"
        result["use_embedded_template"] = True
        return result

    # IBM Granite
    if "granite" in combined:
        result["model_family"] = "granite"
        result["use_embedded_template"] = True
        return result

    # Command-R (Cohere)
    if "command-r" in combined or "command_r" in combined:
        result["model_family"] = "command-r"
        result["use_embedded_template"] = True
        return result

    # Yi models
    if "yi-" in combined or "/yi" in combined:
        result["model_family"] = "yi"
        result["chat_format"] = "chatml"
        return result

    # Zephyr
    if "zephyr" in combined:
        result["model_family"] = "zephyr"
        result["chat_format"] = "zephyr"
        return result

    # OpenChat
    if "openchat" in combined:
        result["model_family"] = "openchat"
        result["chat_format"] = "openchat"
        return result

    # Vicuna
    if "vicuna" in combined:
        result["model_family"] = "vicuna"
        result["chat_format"] = "vicuna"
        return result

    # Alpaca
    if "alpaca" in combined:
        result["model_family"] = "alpaca"
        result["chat_format"] = "alpaca"
        return result

    # ChatGLM
    if "chatglm" in combined or "glm-4" in combined or "glm4" in combined:
        result["model_family"] = "chatglm"
        result["use_embedded_template"] = True
        return result

    # Internlm
    if "internlm" in combined:
        result["model_family"] = "internlm"
        result["use_embedded_template"] = True
        return result

    # Generic chatml models (many finetunes use this)
    if any(x in combined for x in ["hermes", "openhermes", "nous", "dolphin", "neural"]):
        result["model_family"] = "chatml-finetune"
        result["chat_format"] = "chatml"
        return result

    # Fallback - trust embedded template if available
    result["model_family"] = "unknown"
    result["use_embedded_template"] = True
    result["notes"] = "Unknown model - trusting embedded template"
    return result


def main():
    log("Starting worker process")

    # Import llama_cpp after environment is set
    try:
        from llama_cpp import Llama
        log("llama_cpp imported successfully")
    except ImportError as e:
        send({"error": f"Failed to import llama_cpp: {e}"})
        return 1

    # Vision chat handlers - import with fallbacks
    HAS_LLAVA = False
    HAS_MOONDREAM = False
    HAS_GEMMA3 = False

    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        HAS_LLAVA = True
    except ImportError:
        pass

    try:
        from llama_cpp.llama_chat_format import MoondreamChatHandler
        HAS_MOONDREAM = True
    except ImportError:
        pass

    try:
        from llama_cpp.llama_chat_format import Gemma3ChatHandler
        HAS_GEMMA3 = True
    except ImportError:
        pass

    log(f"Vision handlers: Llava={HAS_LLAVA}, Moondream={HAS_MOONDREAM}, Gemma3={HAS_GEMMA3}")

    llm = None
    is_vision = False
    model_name = None
    strip_reasoning = False  # Can be enabled per-request or globally

    # Signal ready
    send({"status": "ready", "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            command = data.get("command")

            if command == "ping":
                send({"status": "pong", "loaded": llm is not None, "model": model_name, "is_vision": is_vision})

            elif command == "load":
                model_path = data["model_path"]
                n_ctx = data.get("n_ctx", 4096)
                n_gpu_layers = data.get("n_gpu_layers", 99)
                use_mmap = data.get("use_mmap", True)
                flash_attn = data.get("flash_attn", True)
                use_mlock = data.get("use_mlock", True)
                mmproj_path = data.get("mmproj_path")
                is_cpu_only = data.get("is_cpu_only", False)

                # Unload existing model
                if llm is not None:
                    log("Unloading existing model")
                    del llm
                    llm = None
                    import gc
                    gc.collect()

                log(f"Loading: {os.path.basename(model_path)}")
                if mmproj_path:
                    log(f"  With vision projector: {os.path.basename(mmproj_path)}")

                # === AUTO-DETECT MODEL FAMILY AND CHAT FORMAT ===
                detection = detect_model_family(model_path, mmproj_path)
                log(f"  Model family: {detection['model_family']}")
                if detection['notes']:
                    log(f"  Note: {detection['notes']}")

                # Build load kwargs
                load_kwargs = {
                    "model_path": model_path,
                    "n_ctx": max(n_ctx, 512),
                    "n_gpu_layers": n_gpu_layers,
                    "use_mmap": use_mmap,
                    "flash_attn": flash_attn,
                    "use_mlock": use_mlock,
                    "verbose": False,
                    "logits_all": True,
                    "n_batch": 1024,
                    "n_threads": os.cpu_count() or 8,
                }

                # Apply chat format if detected (and not using embedded template)
                if detection['chat_format'] and not detection['use_embedded_template']:
                    load_kwargs["chat_format"] = detection['chat_format']
                    log(f"  Using chat_format: {detection['chat_format']}")
                elif detection['use_embedded_template']:
                    log("  Using model's embedded chat template")

                # === SET UP CHAT HANDLER FOR VISION MODELS ===
                try:
                    handler_type = detection['chat_handler_type']

                    if handler_type == 'moondream' and HAS_MOONDREAM:
                        log("  Using Moondream chat handler")
                        chat_handler = MoondreamChatHandler(clip_model_path=mmproj_path)
                        load_kwargs["chat_handler"] = chat_handler
                        is_vision = True

                    elif handler_type == 'gemma3' and HAS_GEMMA3:
                        log("  Using Gemma3 chat handler")
                        chat_handler = Gemma3ChatHandler(clip_model_path=mmproj_path)
                        load_kwargs["chat_handler"] = chat_handler
                        is_vision = True

                    elif handler_type == 'gemma3' and not HAS_GEMMA3 and HAS_LLAVA:
                        log("  Gemma3 handler not available, falling back to Llava15")
                        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                        load_kwargs["chat_handler"] = chat_handler
                        is_vision = True

                    elif handler_type == 'llava15' and HAS_LLAVA:
                        log("  Using Llava15 chat handler")
                        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                        load_kwargs["chat_handler"] = chat_handler
                        is_vision = True

                    elif mmproj_path and HAS_LLAVA:
                        # Fallback for any vision model with mmproj
                        log("  Using Llava15 chat handler (fallback)")
                        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                        load_kwargs["chat_handler"] = chat_handler
                        is_vision = True

                    else:
                        is_vision = False
                        if mmproj_path:
                            log("  WARNING: mmproj provided but no vision handler available")

                except Exception as handler_err:
                    log(f"  Vision handler error: {handler_err}")
                    is_vision = False

                # === SPECULATIVE DECODING (optional) ===
                try:
                    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                    num_pred_tokens = 2 if is_cpu_only else 10
                    draft = LlamaPromptLookupDecoding(num_pred_tokens=num_pred_tokens)
                    load_kwargs["draft_model"] = draft
                    log(f"  Speculative decoding enabled (pred_tokens={num_pred_tokens})")
                except ImportError:
                    pass  # Not available in this version
                except Exception as spec_err:
                    log(f"  Speculative decoding disabled: {spec_err}")

                # Load the model
                try:
                    llm = Llama(**load_kwargs)
                    model_name = os.path.basename(model_path)
                    log("Model loaded successfully")
                    send({
                        "status": "loaded",
                        "model": model_name,
                        "model_family": detection['model_family'],
                        "n_ctx": n_ctx,
                        "n_gpu_layers": n_gpu_layers,
                        "is_vision": is_vision,
                        "chat_format": detection.get('chat_format'),
                    })
                except Exception as load_err:
                    log(f"Load failed: {load_err}")
                    send({"error": f"Load failed: {load_err}"})
                    llm = None
                    is_vision = False

            elif command == "chat":
                if llm is None:
                    send({"error": "No model loaded"})
                    continue

                messages = data["messages"]
                max_tokens = data.get("max_tokens", 256)
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.95)
                top_k = data.get("top_k", 40)
                repeat_penalty = data.get("repeat_penalty", 1.1)
                presence_penalty = data.get("presence_penalty", 0.0)
                frequency_penalty = data.get("frequency_penalty", 0.0)
                mirostat = data.get("mirostat", 0)
                mirostat_tau = data.get("mirostat_tau", 5.0)
                mirostat_eta = data.get("mirostat_eta", 0.1)
                typical_p = data.get("typical_p", 1.0)
                tfs_z = data.get("tfs_z", 1.0)
                stop = data.get("stop", None)
                request_id = data.get("request_id", "unknown")
                do_strip_reasoning = data.get("strip_reasoning", strip_reasoning)

                # Count images in messages for logging
                total_images = 0
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image_url":
                                total_images += 1

                if total_images > 0:
                    log(f"Chat request {request_id[:8]} with {total_images} image(s)")
                else:
                    log(f"Chat request {request_id[:8]}")

                try:
                    # Retry logic for transient CUDA errors
                    max_retries = 3

                    for attempt in range(max_retries):
                        try:
                            if attempt > 0:
                                log(f"Retry attempt {attempt + 1}/{max_retries}")
                                import time as time_mod
                                time_mod.sleep(0.1 * (attempt + 1))

                            stream = llm.create_chat_completion(
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=max(temperature, 0.01),
                                top_p=top_p,
                                top_k=top_k,
                                repeat_penalty=repeat_penalty,
                                presence_penalty=presence_penalty,
                                frequency_penalty=frequency_penalty,
                                mirostat_mode=mirostat,
                                mirostat_tau=mirostat_tau,
                                mirostat_eta=mirostat_eta,
                                typical_p=typical_p,
                                tfs_z=tfs_z,
                                stop=stop,
                                stream=True
                            )

                            prompt_tokens = 0
                            completion_tokens = 0
                            finish_reason = None

                            for chunk in stream:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                text = delta.get("content", "")

                                if text:
                                    # Optionally strip reasoning tokens
                                    if do_strip_reasoning:
                                        text = strip_reasoning_tokens(text)
                                    if text:  # May be empty after stripping
                                        send({"text": text, "request_id": request_id})
                                        completion_tokens += 1

                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                                    break

                            usage = chunk.get("usage", {}) if 'chunk' in dir() else {}
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)

                            log(f"Chat complete: {completion_tokens} tokens")
                            send({
                                "done": True,
                                "request_id": request_id,
                                "finish_reason": finish_reason,
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                }
                            })
                            break  # Success

                        except Exception as retry_e:
                            error_str = str(retry_e)
                            if "llama_decode returned -1" in error_str and attempt < max_retries - 1:
                                log(f"Retryable error: {error_str}")
                                continue
                            else:
                                raise

                except Exception as gen_e:
                    error_msg = str(gen_e)
                    log(f"Chat error: {error_msg}")

                    is_context_error = (
                        "exceed" in error_msg.lower() or
                        "context" in error_msg.lower() or
                        "requested tokens" in error_msg.lower() or
                        "too many tokens" in error_msg.lower()
                    )

                    if is_context_error:
                        send({
                            "error": f"Context overflow: {error_msg}",
                            "request_id": request_id,
                            "recoverable": True
                        })
                    else:
                        send({
                            "error": error_msg,
                            "request_id": request_id,
                            "recoverable": False
                        })
                    send({"done": True, "request_id": request_id})

            elif command == "generate":
                # Raw text generation (non-chat)
                if llm is None:
                    send({"error": "No model loaded"})
                    continue

                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 256)
                temperature = data.get("temperature", 0.7)
                request_id = data.get("request_id", "unknown")

                try:
                    stream = llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=max(temperature, 0.01),
                        stream=True
                    )

                    for chunk in stream:
                        text = chunk["choices"][0]["text"]
                        if text:
                            send({"text": text, "request_id": request_id})

                    send({"done": True, "request_id": request_id})

                except Exception as e:
                    send({"error": str(e), "request_id": request_id})
                    send({"done": True, "request_id": request_id})

            elif command == "unload":
                if llm is not None:
                    log("Unloading model")
                    del llm
                    llm = None
                    model_name = None
                    is_vision = False
                    import gc
                    gc.collect()
                    send({"status": "unloaded"})
                else:
                    send({"status": "already_unloaded"})

            elif command == "exit":
                log("Exit command received")
                break

            else:
                send({"error": f"Unknown command: {command}"})

        except json.JSONDecodeError as e:
            send({"error": f"Invalid JSON: {e}"})
        except Exception as e:
            log(f"Unexpected error: {traceback.format_exc()}")
            send({"error": str(e)})

    # Cleanup
    if llm is not None:
        log("Final cleanup - unloading model")
        del llm

    log("Worker exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
