"""
Model Management Routes

REST API for loading, unloading, and querying models.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class LoadModelRequest(BaseModel):
    provider: str  # llama_cpp, lm_studio, ollama, openrouter
    model_id: str
    options: Optional[Dict[str, Any]] = None


class LoadModelResponse(BaseModel):
    success: bool
    instance_id: Optional[str] = None
    error: Optional[str] = None


@router.get("/list")
async def list_all_models(request: Request):
    """List all available models from all providers."""
    orchestrator = request.app.state.orchestrator
    models = await orchestrator.list_all_models()
    return models


@router.get("/list/{provider}")
async def list_provider_models(request: Request, provider: str):
    """List models from a specific provider only."""
    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    try:
        provider_type = ProviderType(provider)
    except ValueError:
        return {"error": f"Invalid provider: {provider}", "models": []}

    if provider_type not in orchestrator.providers:
        return {"error": f"Provider {provider} not enabled", "models": []}

    try:
        prov = orchestrator.providers[provider_type]
        models = await prov.list_models()
        return {"models": models}
    except Exception as e:
        return {"error": str(e), "models": []}


@router.get("/loaded")
async def list_loaded_models(request: Request):
    """List all currently loaded model instances."""
    orchestrator = request.app.state.orchestrator
    instances = orchestrator.get_loaded_instances()
    return [
        {
            "id": i.id,
            "provider": i.provider_type.value,
            "model": i.model_identifier,
            "name": i.display_name,
            "status": i.status.value,
            "gpu_index": i.gpu_index,
            "context_length": i.context_length,
            "request_count": i.request_count,
            "has_vision": i.metadata.get("has_vision", False),
            # Include metadata for error display (Ollama pre-warm failures, etc.)
            "metadata": {
                "error": i.metadata.get("error"),
            } if i.metadata.get("error") else None,
        }
        for i in instances
    ]


@router.post("/load", response_model=LoadModelResponse)
async def load_model(request: Request, body: LoadModelRequest):
    """Load a model from a provider."""
    import logging
    logger = logging.getLogger(__name__)

    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    logger.info(f"=== Load model request ===")
    logger.info(f"Provider: {body.provider}")
    logger.info(f"Model ID: {body.model_id}")
    logger.info(f"Options: {body.options}")

    try:
        provider_type = ProviderType(body.provider)
        options = body.options or {}

        instance = await orchestrator.load_model(
            provider_type, body.model_id, **options
        )

        logger.info(f"Model loaded successfully: instance_id={instance.id}")
        return LoadModelResponse(
            success=True,
            instance_id=instance.id
        )

    except ValueError as e:
        logger.error(f"Invalid provider error: {e}")
        return LoadModelResponse(success=False, error=f"Invalid provider: {body.provider}")
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return LoadModelResponse(success=False, error=str(e))


@router.post("/load-async")
async def load_model_async(request: Request, body: LoadModelRequest):
    """Load a model asynchronously (returns immediately with pending ID)."""
    import logging
    logger = logging.getLogger(__name__)

    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    logger.info(f"=== Async load model request ===")
    logger.info(f"Provider: {body.provider}")
    logger.info(f"Model ID: {body.model_id}")
    logger.info(f"Options: {body.options}")

    try:
        provider_type = ProviderType(body.provider)
        options = body.options or {}

        # This returns immediately with a pending ID
        pending_id = await orchestrator.load_model_async(
            provider_type, body.model_id, **options
        )

        logger.info(f"Model load started: pending_id={pending_id}")
        return {"success": True, "pending_id": pending_id}

    except ValueError as e:
        logger.error(f"Invalid provider error: {e}")
        return {"success": False, "error": f"Invalid provider: {body.provider}"}
    except Exception as e:
        logger.error(f"Async load error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/load-jit")
async def load_model_jit(request: Request, body: LoadModelRequest):
    """Load model with JIT (returns existing if already loaded)."""
    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    try:
        provider_type = ProviderType(body.provider)
        options = body.options or {}

        instance_id = await orchestrator.load_model_jit(
            provider_type, body.model_id, **options
        )

        return {"success": True, "instance_id": instance_id}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/{instance_id}")
async def unload_model(request: Request, instance_id: str):
    """Unload a model instance."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"=== Unload model request: {instance_id} ===")
    orchestrator = request.app.state.orchestrator

    try:
        success = await orchestrator.unload_model(instance_id)
        if not success:
            logger.warning(f"Unload failed - instance not found: {instance_id}")
            return {"success": False, "error": f"Model instance not found: {instance_id}"}
        logger.info(f"Unload successful: {instance_id}")
        return {"success": success}
    except Exception as e:
        logger.error(f"Unload error for {instance_id}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/pending")
async def get_pending_loads(request: Request):
    """Get list of pending model loads."""
    orchestrator = request.app.state.orchestrator

    try:
        if hasattr(orchestrator, 'get_pending_loads'):
            pending = orchestrator.get_pending_loads()
            # Serialize ModelInstance objects
            return {"pending": [
                {
                    "id": p.id,
                    "provider": p.provider_type.value,
                    "model": p.model_identifier,
                    "name": p.display_name,
                    "status": p.status.value,
                }
                for p in pending
            ]}
        else:
            return {"pending": []}
    except Exception as e:
        return {"pending": [], "error": str(e)}


@router.get("/health/all")
async def check_all_health(request: Request):
    """Check health of all providers."""
    orchestrator = request.app.state.orchestrator
    health = await orchestrator.check_all_health()
    return health


@router.get("/providers")
async def get_enabled_providers(request: Request):
    """Get list of enabled providers."""
    orchestrator = request.app.state.orchestrator
    providers = orchestrator.get_enabled_providers()
    return [p.value for p in providers]


@router.post("/cancel/{instance_id}")
async def cancel_model_load(request: Request, instance_id: str):
    """Cancel a pending model load."""
    import logging
    logger = logging.getLogger(__name__)

    orchestrator = request.app.state.orchestrator
    logger.info(f"=== Cancel model load request: {instance_id} ===")

    try:
        # Check if orchestrator has cancel_load method
        if hasattr(orchestrator, 'cancel_load'):
            success = await orchestrator.cancel_load(instance_id)
            if success:
                logger.info(f"Model load cancelled: {instance_id}")
            else:
                logger.warning(f"Cancel returned False for: {instance_id}")
            return {"success": success}
        else:
            logger.warning("Orchestrator does not have cancel_load method")
            return {"success": False, "error": "Cancel not supported"}
    except Exception as e:
        logger.error(f"Cancel error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/providers/reload/{provider}")
async def reload_provider(request: Request, provider: str):
    """Reload a provider after settings change."""
    import logging
    logger = logging.getLogger(__name__)

    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    try:
        provider_type = ProviderType(provider)
        logger.info(f"Reloading provider: {provider}")
        await orchestrator.reload_provider(provider_type)
        logger.info(f"Provider reloaded: {provider}")
        return {"success": True, "provider": provider}
    except ValueError:
        return {"success": False, "error": f"Invalid provider: {provider}"}
    except Exception as e:
        logger.error(f"Provider reload error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/providers/openrouter/credits")
async def get_openrouter_credits(request: Request):
    """Get OpenRouter account balance/credits info."""
    import logging
    logger = logging.getLogger(__name__)

    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    # Check if OpenRouter provider is enabled
    if ProviderType.OPENROUTER not in orchestrator.providers:
        return {"success": False, "error": "OpenRouter provider not enabled"}

    provider = orchestrator.providers[ProviderType.OPENROUTER]

    try:
        credits = await provider.get_credits()
        if "error" in credits:
            return {"success": False, "error": credits["error"]}
        return {"success": True, "credits": credits}
    except Exception as e:
        logger.error(f"Error fetching OpenRouter credits: {e}")
        return {"success": False, "error": str(e)}


@router.get("/context-length/{provider}/{model_id:path}")
async def get_model_context_length(request: Request, provider: str, model_id: str):
    """
    Get exact context length for a specific model.

    Called when user selects a model in the load modal dropdown.
    Makes a single API call to the provider to get the exact value.
    """
    import logging
    logger = logging.getLogger(__name__)

    from providers.base import ProviderType

    orchestrator = request.app.state.orchestrator

    try:
        provider_type = ProviderType(provider)

        if provider_type not in orchestrator.providers:
            return {"success": False, "error": f"Provider {provider} not available"}

        prov = orchestrator.providers[provider_type]

        # Provider-specific context length lookup
        if provider == "ollama":
            if hasattr(prov, 'get_model_context_length'):
                context_length = await prov.get_model_context_length(model_id)
                return {"success": True, "context_length": context_length, "model_id": model_id}

        elif provider == "lm_studio":
            # LM Studio: try native API for specific model info
            if hasattr(prov, '_get_session'):
                try:
                    session = await prov._get_session()
                    native_url = prov.base_url.replace("/v1", "") + f"/api/v0/models/{model_id}"
                    async with session.get(native_url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            context_length = data.get("max_context_length")
                            if context_length:
                                return {"success": True, "context_length": context_length, "model_id": model_id}
                except Exception as e:
                    logger.debug(f"LM Studio model info lookup failed: {e}")

            # Fallback to estimation
            if hasattr(prov, '_estimate_context_length'):
                context_length = prov._estimate_context_length(model_id)
                return {"success": True, "context_length": context_length, "model_id": model_id, "estimated": True}

        elif provider == "llama_cpp" or provider == "vllm":
            # llama_cpp / vLLM: read from GGUF metadata
            try:
                from backend.utils.gguf_utils import get_model_context_length
                context_length = get_model_context_length(model_id)
                if context_length:
                    return {"success": True, "context_length": context_length, "model_id": model_id}
            except Exception as e:
                logger.debug(f"GGUF context length read failed: {e}")

        elif provider == "openrouter":
            # OpenRouter: already has context_length in list, just return from cache
            models = await prov.list_models()
            for m in models:
                if m.get("id") == model_id:
                    return {"success": True, "context_length": m.get("context_length"), "model_id": model_id}

        # Fallback: return None and let UI use default
        return {"success": True, "context_length": None, "model_id": model_id}

    except ValueError:
        return {"success": False, "error": f"Invalid provider: {provider}"}
    except Exception as e:
        logger.error(f"Context length lookup error: {e}")
        return {"success": False, "error": str(e)}


# NOTE: This wildcard route must come AFTER all specific routes like /pending, /providers, /health/all
@router.get("/{instance_id}")
async def get_instance_info(request: Request, instance_id: str):
    """Get detailed info about a model instance."""
    orchestrator = request.app.state.orchestrator
    instance = orchestrator.get_instance(instance_id)

    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    return {
        "id": instance.id,
        "provider": instance.provider_type.value,
        "model": instance.model_identifier,
        "name": instance.display_name,
        "status": instance.status.value,
        "gpu_index": instance.gpu_index,
        "context_length": instance.context_length,
        "request_count": instance.request_count,
        "slot_name": instance.slot_name,
        "metadata": instance.metadata,
    }
