"""
Embedding Manager

Auto-manages embedding model lifecycle across multiple providers.
Cascades through: LM Studio → Ollama → llama.cpp → OpenRouter

Key feature: Auto-loads embedding models when needed, auto-unloads when done.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import aiohttp
except ImportError:
    aiohttp = None

# LM Studio SDK for model loading
try:
    import lmstudio
    HAS_LMSTUDIO_SDK = True
except ImportError:
    lmstudio = None
    HAS_LMSTUDIO_SDK = False

logger = logging.getLogger("utils.embedding_manager")


class EmbeddingProvider(Enum):
    LOCAL = "local"  # sentence-transformers, auto-downloads to app
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    OPENROUTER = "openrouter"
    NONE = "none"


# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class ProviderInfo:
    """Information about an available embedding provider."""
    provider: EmbeddingProvider
    available: bool
    model_id: str = ""
    base_url: str = ""
    error: str = ""


class EmbeddingManager:
    """
    Auto-manages embedding model lifecycle across providers.

    Features:
    - Cascading provider detection (LM Studio → Ollama → llama.cpp → OpenRouter)
    - Auto-load on first embed request
    - Auto-unload when session ends
    - Fallback to simple injection if no provider available
    """

    # Known embedding model patterns
    EMBEDDING_MODEL_PATTERNS = [
        "text-embedding",
        "nomic-embed",
        "bge-",
        "e5-",
        "all-minilm",
        "mxbai-embed",
        "snowflake-arctic-embed",
    ]

    # Default local embedding model (small, fast, good quality)
    DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.active_provider: Optional[EmbeddingProvider] = None
        self.active_model: Optional[str] = None
        self.active_base_url: Optional[str] = None
        self._auto_loaded: bool = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._embedding_dim: Optional[int] = None
        # LM Studio SDK handle for loaded model
        self._lm_studio_client = None
        self._lm_studio_model_handle = None
        self._lm_studio_identifier: Optional[str] = None
        # Local sentence-transformers model
        self._local_model: Optional["SentenceTransformer"] = None
        # App root for model storage
        self._app_root = self._get_app_root()

    def _get_app_root(self) -> str:
        """Get the app's root directory for model storage."""
        import os
        # Go up from backend/utils to app root
        current_file = os.path.abspath(__file__)
        utils_dir = os.path.dirname(current_file)
        backend_dir = os.path.dirname(utils_dir)
        app_root = os.path.dirname(backend_dir)
        return app_root

    def _get_models_dir(self) -> str:
        """Get the models directory, creating it if needed."""
        import os
        models_dir = os.path.join(self._app_root, "models", "embeddings")
        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for embedding manager")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _is_embedding_model(self, model_id: str) -> bool:
        """Check if model ID looks like an embedding model."""
        model_lower = model_id.lower()
        return any(p in model_lower for p in self.EMBEDDING_MODEL_PATTERNS)

    # ==================== Provider Detection ====================

    async def detect_providers(self) -> List[ProviderInfo]:
        """
        Check which embedding providers are available.

        Returns list in priority order: LM Studio, Local, Ollama, OpenRouter
        """
        providers = []

        # Check LM Studio (primary - if available)
        lm_studio = await self._check_lm_studio()
        providers.append(lm_studio)

        # Check Local sentence-transformers (fallback - always available if installed)
        local = self._check_local()
        providers.append(local)

        # Check Ollama
        ollama = await self._check_ollama()
        providers.append(ollama)

        # Check OpenRouter
        openrouter = await self._check_openrouter()
        providers.append(openrouter)

        return providers

    def _check_local(self) -> ProviderInfo:
        """Check if local sentence-transformers is available."""
        if not HAS_SENTENCE_TRANSFORMERS:
            return ProviderInfo(
                provider=EmbeddingProvider.LOCAL,
                available=False,
                error="sentence-transformers not installed (pip install sentence-transformers)"
            )

        # Local is always available if the package is installed
        # Model will be auto-downloaded on first use
        return ProviderInfo(
            provider=EmbeddingProvider.LOCAL,
            available=True,
            model_id=self.DEFAULT_LOCAL_MODEL,
            base_url=self._get_models_dir()  # Store path for reference
        )

    async def _check_lm_studio(self) -> ProviderInfo:
        """
        Check if LM Studio has embedding models available.

        Checks DOWNLOADED models (not just loaded), so we can auto-load them.
        """
        try:
            lm_config = self.settings.get("providers", {}).get("lm_studio", {})
            if not lm_config.get("enabled", False):
                return ProviderInfo(
                    provider=EmbeddingProvider.LM_STUDIO,
                    available=False,
                    error="LM Studio provider disabled"
                )

            base_url = lm_config.get("base_url", "http://localhost:1234/v1")

            # First check if LM Studio is reachable
            session = await self._get_session()
            try:
                async with session.get(
                    f"{base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        return ProviderInfo(
                            provider=EmbeddingProvider.LM_STUDIO,
                            available=False,
                            error=f"LM Studio not reachable (HTTP {response.status})"
                        )
            except Exception as e:
                return ProviderInfo(
                    provider=EmbeddingProvider.LM_STUDIO,
                    available=False,
                    error=f"LM Studio not reachable: {e}"
                )

            # Use SDK to get DOWNLOADED models (not just loaded)
            if HAS_LMSTUDIO_SDK:
                try:
                    embedding_models = await self._get_lm_studio_downloaded_embedding_models()
                    if embedding_models:
                        # Prefer nomic-embed if available
                        preferred = next(
                            (m for m in embedding_models if "nomic" in m.lower()),
                            embedding_models[0]
                        )
                        return ProviderInfo(
                            provider=EmbeddingProvider.LM_STUDIO,
                            available=True,
                            model_id=preferred,
                            base_url=base_url
                        )
                except Exception as e:
                    logger.warning(f"SDK model listing failed: {e}")

            # Fallback: check loaded models via API
            async with session.get(
                f"{base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                data = await response.json()
                models = data.get("data", [])

                embedding_models = [
                    m["id"] for m in models
                    if self._is_embedding_model(m.get("id", ""))
                ]

                if embedding_models:
                    preferred = next(
                        (m for m in embedding_models if "nomic" in m.lower()),
                        embedding_models[0]
                    )
                    return ProviderInfo(
                        provider=EmbeddingProvider.LM_STUDIO,
                        available=True,
                        model_id=preferred,
                        base_url=base_url
                    )

            return ProviderInfo(
                provider=EmbeddingProvider.LM_STUDIO,
                available=False,
                error="No embedding models found (download one in LM Studio)"
            )

        except asyncio.TimeoutError:
            return ProviderInfo(
                provider=EmbeddingProvider.LM_STUDIO,
                available=False,
                error="Connection timeout"
            )
        except Exception as e:
            return ProviderInfo(
                provider=EmbeddingProvider.LM_STUDIO,
                available=False,
                error=str(e)
            )

    async def _get_lm_studio_downloaded_embedding_models(self) -> List[str]:
        """Get list of downloaded embedding models from LM Studio via SDK."""
        if not HAS_LMSTUDIO_SDK:
            return []

        loop = asyncio.get_event_loop()

        try:
            # Get SDK client
            api_host = lmstudio.Client.find_default_local_api_host()
            if not api_host:
                return []

            client = lmstudio.Client(api_host=api_host)

            # List downloaded embedding models (use embedding domain)
            downloaded = await loop.run_in_executor(
                None,
                lambda: list(client.embedding.list_downloaded())
            )

            # All models from embedding.list_downloaded() are embedding models
            embedding_models = [m.path for m in downloaded]

            logger.info(f"Found {len(embedding_models)} downloaded embedding models in LM Studio")
            return embedding_models

        except Exception as e:
            logger.warning(f"Failed to list LM Studio downloaded models: {e}")
            return []

    async def _check_ollama(self) -> ProviderInfo:
        """Check if Ollama has embedding models available."""
        try:
            ollama_config = self.settings.get("providers", {}).get("ollama", {})
            if not ollama_config.get("enabled", False):
                return ProviderInfo(
                    provider=EmbeddingProvider.OLLAMA,
                    available=False,
                    error="Ollama provider disabled"
                )

            base_url = ollama_config.get("base_url", "http://localhost:11434")

            session = await self._get_session()
            async with session.get(
                f"{base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    return ProviderInfo(
                        provider=EmbeddingProvider.OLLAMA,
                        available=False,
                        error=f"HTTP {response.status}"
                    )

                data = await response.json()
                models = data.get("models", [])

                # Find embedding models
                embedding_models = [
                    m["name"] for m in models
                    if self._is_embedding_model(m.get("name", ""))
                ]

                if embedding_models:
                    preferred = next(
                        (m for m in embedding_models if "nomic" in m.lower()),
                        embedding_models[0]
                    )
                    return ProviderInfo(
                        provider=EmbeddingProvider.OLLAMA,
                        available=True,
                        model_id=preferred,
                        base_url=base_url
                    )
                else:
                    return ProviderInfo(
                        provider=EmbeddingProvider.OLLAMA,
                        available=False,
                        error="No embedding models found"
                    )

        except asyncio.TimeoutError:
            return ProviderInfo(
                provider=EmbeddingProvider.OLLAMA,
                available=False,
                error="Connection timeout"
            )
        except Exception as e:
            return ProviderInfo(
                provider=EmbeddingProvider.OLLAMA,
                available=False,
                error=str(e)
            )

    async def _check_openrouter(self) -> ProviderInfo:
        """Check if OpenRouter is configured (always available if API key set)."""
        try:
            or_config = self.settings.get("providers", {}).get("openrouter", {})
            if not or_config.get("enabled", False):
                return ProviderInfo(
                    provider=EmbeddingProvider.OPENROUTER,
                    available=False,
                    error="OpenRouter provider disabled"
                )

            api_key = or_config.get("api_key", "")
            if not api_key or api_key == "sk-or-...":
                return ProviderInfo(
                    provider=EmbeddingProvider.OPENROUTER,
                    available=False,
                    error="No API key configured"
                )

            # OpenRouter doesn't have a native embeddings endpoint,
            # but we could use their text-embedding models if available
            # For now, mark as unavailable for embeddings
            return ProviderInfo(
                provider=EmbeddingProvider.OPENROUTER,
                available=False,
                error="OpenRouter embeddings not yet implemented"
            )

        except Exception as e:
            return ProviderInfo(
                provider=EmbeddingProvider.OPENROUTER,
                available=False,
                error=str(e)
            )

    # ==================== Auto Load/Unload ====================

    async def auto_load(self) -> bool:
        """
        Load embedding model from first available provider.

        For LM Studio: Actually loads the model via SDK if not already loaded.
        Returns True if successfully loaded, False otherwise.
        """
        if self.active_provider is not None:
            logger.info(f"Embedding model already loaded: {self.active_model}")
            return True

        providers = await self.detect_providers()

        for provider_info in providers:
            if not provider_info.available:
                logger.debug(
                    f"Skipping {provider_info.provider.value}: {provider_info.error}"
                )
                continue

            try:
                logger.info(
                    f"Loading embedding model from {provider_info.provider.value}: "
                    f"{provider_info.model_id}"
                )

                # For LM Studio, actually load the model via SDK
                if provider_info.provider == EmbeddingProvider.LM_STUDIO:
                    loaded = await self._load_lm_studio_embedding_model(
                        provider_info.model_id
                    )
                    if not loaded:
                        logger.warning("Failed to load LM Studio embedding model")
                        continue

                # For Local, load sentence-transformers model
                if provider_info.provider == EmbeddingProvider.LOCAL:
                    loaded = await self._load_local_embedding_model(
                        provider_info.model_id
                    )
                    if not loaded:
                        logger.warning("Failed to load local embedding model")
                        continue

                self.active_provider = provider_info.provider
                self.active_model = provider_info.model_id
                self.active_base_url = provider_info.base_url
                self._auto_loaded = True

                # Test with a simple embed to verify it works
                test_result = await self._embed_single(
                    "test", provider_info.provider, provider_info.base_url,
                    provider_info.model_id
                )

                if test_result is not None:
                    self._embedding_dim = len(test_result)
                    logger.info(
                        f"Embedding model loaded successfully. "
                        f"Dimension: {self._embedding_dim}"
                    )
                    return True
                else:
                    # Reset if test failed
                    await self._cleanup_lm_studio()
                    self.active_provider = None
                    self.active_model = None
                    self.active_base_url = None
                    self._auto_loaded = False

            except Exception as e:
                logger.warning(
                    f"Failed to load from {provider_info.provider.value}: {e}"
                )
                await self._cleanup_lm_studio()
                self.active_provider = None
                self.active_model = None
                self.active_base_url = None
                self._auto_loaded = False
                continue

        logger.warning("No embedding provider available")
        return False

    async def _load_lm_studio_embedding_model(self, model_path: str) -> bool:
        """Load embedding model in LM Studio via SDK."""
        if not HAS_LMSTUDIO_SDK:
            logger.warning("LM Studio SDK not available, cannot auto-load model")
            return False

        loop = asyncio.get_event_loop()

        try:
            # Get SDK client
            api_host = lmstudio.Client.find_default_local_api_host()
            if not api_host:
                logger.error("Could not find LM Studio API host")
                return False

            self._lm_studio_client = lmstudio.Client(api_host=api_host)

            # Check if embedding model is already loaded (use embedding domain, not llm)
            loaded_models = await loop.run_in_executor(
                None,
                lambda: list(self._lm_studio_client.embedding.list_loaded())
            )

            # Check if our embedding model is already loaded
            for m in loaded_models:
                logger.info(f"Found loaded embedding model: {m.identifier}")
                self._lm_studio_identifier = m.identifier
                return True

            # Load the model using embedding domain (not llm)
            import time
            self._lm_studio_identifier = f"embedding-{int(time.time())}"

            logger.info(f"Loading embedding model: {model_path} as {self._lm_studio_identifier}")

            self._lm_studio_model_handle = await loop.run_in_executor(
                None,
                lambda: self._lm_studio_client.embedding.load_new_instance(
                    model_path,
                    self._lm_studio_identifier,
                    ttl=3600,  # 1 hour TTL
                )
            )

            logger.info(f"Embedding model loaded successfully: {self._lm_studio_identifier}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LM Studio embedding model: {e}")
            await self._cleanup_lm_studio()
            return False

    async def _cleanup_lm_studio(self):
        """Clean up LM Studio SDK resources."""
        if self._lm_studio_model_handle is not None:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._lm_studio_model_handle.unload),
                    timeout=10.0
                )
                logger.info("LM Studio embedding model unloaded")
            except asyncio.TimeoutError:
                logger.warning("Timeout unloading LM Studio model")
            except Exception as e:
                logger.warning(f"Error unloading LM Studio model: {e}")

        self._lm_studio_model_handle = None
        self._lm_studio_identifier = None

        if self._lm_studio_client:
            try:
                self._lm_studio_client.close()
            except Exception:
                pass
            self._lm_studio_client = None

    async def auto_unload(self):
        """Unload embedding model if we auto-loaded it."""
        if not self._auto_loaded:
            logger.debug("No auto-loaded model to unload")
            return

        logger.info(
            f"Unloading embedding model: {self.active_model} "
            f"from {self.active_provider.value if self.active_provider else 'none'}"
        )

        # For LM Studio, actually unload via SDK
        if self.active_provider == EmbeddingProvider.LM_STUDIO:
            await self._cleanup_lm_studio()

        self.active_provider = None
        self.active_model = None
        self.active_base_url = None
        self._auto_loaded = False
        self._embedding_dim = None

    # ==================== Embedding ====================

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using active provider.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError if no provider available
        """
        if not texts:
            return []

        # Auto-load if not already loaded
        if self.active_provider is None:
            success = await self.auto_load()
            if not success:
                raise RuntimeError(
                    "No embedding provider available. "
                    "Ensure LM Studio or Ollama is running with an embedding model."
                )

        return await self._embed_batch(
            texts, self.active_provider, self.active_base_url, self.active_model
        )

    async def _load_local_embedding_model(self, model_id: str) -> bool:
        """Load local sentence-transformers model, downloading if needed."""
        if not HAS_SENTENCE_TRANSFORMERS:
            return False

        import os
        loop = asyncio.get_event_loop()

        try:
            # Set HuggingFace cache to our models directory
            models_dir = self._get_models_dir()
            os.environ["HF_HOME"] = models_dir
            os.environ["TRANSFORMERS_CACHE"] = models_dir

            logger.info(f"Loading local embedding model: {model_id} (cache: {models_dir})")

            # Load model (will download on first use)
            self._local_model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(model_id, cache_folder=models_dir)
            )

            logger.info(f"Local embedding model loaded: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            self._local_model = None
            return False

    async def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Embed using local sentence-transformers model."""
        if self._local_model is None:
            raise RuntimeError("Local embedding model not loaded")

        loop = asyncio.get_event_loop()

        # Run embedding in thread pool to not block
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._local_model.encode(texts, convert_to_numpy=True).tolist()
        )

        return embeddings

    async def _embed_batch(
        self,
        texts: List[str],
        provider: EmbeddingProvider,
        base_url: str,
        model_id: str
    ) -> List[List[float]]:
        """Embed a batch of texts."""
        if provider == EmbeddingProvider.LOCAL:
            return await self._embed_local(texts)
        elif provider == EmbeddingProvider.LM_STUDIO:
            return await self._embed_lm_studio(texts, base_url, model_id)
        elif provider == EmbeddingProvider.OLLAMA:
            return await self._embed_ollama(texts, base_url, model_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _embed_single(
        self,
        text: str,
        provider: EmbeddingProvider,
        base_url: str,
        model_id: str
    ) -> Optional[List[float]]:
        """Embed a single text, returns None on failure."""
        try:
            results = await self._embed_batch([text], provider, base_url, model_id)
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Single embed failed: {e}")
            return None

    async def _embed_lm_studio(
        self,
        texts: List[str],
        base_url: str,
        model_id: str
    ) -> List[List[float]]:
        """Embed using LM Studio's OpenAI-compatible endpoint."""
        session = await self._get_session()

        # LM Studio supports batch embedding
        async with session.post(
            f"{base_url}/embeddings",
            json={"input": texts, "model": model_id},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"LM Studio embedding error: {error_text}")

            data = await response.json()
            embeddings_data = data.get("data", [])

            # Sort by index and extract embeddings
            embeddings_data.sort(key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in embeddings_data]

    async def _embed_ollama(
        self,
        texts: List[str],
        base_url: str,
        model_id: str
    ) -> List[List[float]]:
        """Embed using Ollama's embeddings endpoint."""
        session = await self._get_session()

        # Ollama doesn't support batch, so we embed one at a time
        embeddings = []
        for text in texts:
            async with session.post(
                f"{base_url}/api/embeddings",
                json={"model": model_id, "prompt": text},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama embedding error: {error_text}")

                data = await response.json()
                embeddings.append(data["embedding"])

        return embeddings

    # ==================== Status ====================

    def get_status(self) -> Dict[str, Any]:
        """Get current embedding manager status."""
        return {
            "active_provider": self.active_provider.value if self.active_provider else None,
            "active_model": self.active_model,
            "auto_loaded": self._auto_loaded,
            "embedding_dim": self._embedding_dim,
            "ready": self.active_provider is not None,
        }

    async def close(self):
        """Clean up resources."""
        await self.auto_unload()
        if self._session and not self._session.closed:
            await self._session.close()
