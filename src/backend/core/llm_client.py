import asyncio
import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import ollama

logger = logging.getLogger(__name__)


class LLMCache:
    """Simple cache for LLM responses to avoid duplicate API calls"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize LLM cache with max size and TTL (in seconds)"""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time-to-live in seconds
        self.last_cleanup = datetime.now()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if (datetime.now() - entry["timestamp"]).total_seconds() < self.ttl:
                return entry["value"]
            else:
                # Remove expired entry
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp"""
        # Cleanup if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup()

        self.cache[key] = {"value": value, "timestamp": datetime.now()}

    def _cleanup(self) -> None:
        """Remove expired entries and trim cache if needed"""
        now = datetime.now()
        # Remove expired entries
        expired_keys = [
            k
            for k, v in self.cache.items()
            if (now - v["timestamp"]).total_seconds() >= self.ttl
        ]
        for k in expired_keys:
            del self.cache[k]

        # If still too large, remove oldest entries
        if len(self.cache) >= self.max_size:
            sorted_keys = sorted(
                self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
            )
            for k in sorted_keys[
                : len(self.cache) - self.max_size + 10
            ]:  # Remove batch
                del self.cache[k]


class EnhancedLLMClient:
    """Enhanced LLM client with caching and improved error handling"""

    def __init__(self):
        """Initialize enhanced LLM client"""
        self.cache = LLMCache(max_size=1000, ttl=3600)  # 1 hour cache
        self.embedding_cache = LLMCache(max_size=5000, ttl=86400)  # 24 hour cache
        self.models_info = {}
        self.last_error = None
        self.error_count = 0
        self.last_request_time = datetime.now()

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Send chat messages to the LLM

        Args:
            model: Model name
            messages: List of message dicts with role and content
            stream: Whether to stream the response
            options: Additional options to pass to the model

        Returns:
            Response dict or async generator for streaming
        """
        start_time = time.time()
        options = options or {}

        # For non-streaming, check cache
        if not stream:
            cache_key = f"{model}_{str(messages)}_{str(options)}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {model}")
                return cached

        try:
            self.last_request_time = datetime.now()

            # Format messages for Ollama
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Add system message as a special parameter
                    options["system"] = msg["content"]
                else:
                    formatted_messages.append(msg)

            if stream:
                # Return streaming generator
                return self._stream_response(model, formatted_messages, options)
            else:
                # For non-streaming, get complete response
                response = await asyncio.to_thread(
                    ollama.chat,
                    model=model,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in formatted_messages
                    ],
                    options=options,
                )

                # Format response to standard structure
                result = {
                    "message": {
                        "role": "assistant",
                        "content": response["message"]["content"],
                    },
                    "response": response["message"]["content"],
                }

                # Cache the result
                self.cache.set(cache_key, result)

                logger.debug(
                    f"LLM request to {model} completed in {time.time() - start_time:.2f}s"
                )
                return result

        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Error in LLM request to {model}: {str(e)}")
            raise

    async def _stream_response(
        self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from LLM"""
        try:
            # Call Ollama's streaming API
            response_stream = await asyncio.to_thread(
                ollama.chat,
                model=model,
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
                options=options,
                stream=True,
            )

            # Process and yield each chunk
            for chunk in response_stream:
                yield {
                    "message": {
                        "role": "assistant",
                        "content": chunk["message"]["content"],
                    },
                    "response": chunk["message"]["content"],
                }

        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Error in streaming LLM request to {model}: {str(e)}")
            yield {"message": {"role": "assistant", "content": f"Error: {str(e)}"}}

    async def embed(
        self, model: str, text: str, options: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Get embeddings for text

        Args:
            model: Model name
            text: Text to embed
            options: Additional options

        Returns:
            List of embedding floats
        """
        options = options or {}
        cache_key = f"embed_{model}_{text}"

        # Check cache
        cached = self.embedding_cache.get(cache_key)
        if cached:
            return cached

        try:
            # Use Ollama's embeddings API
            response = await asyncio.to_thread(
                ollama.embeddings, model=model, prompt=text, options=options
            )

            embeddings = response["embedding"]

            # Cache result
            self.embedding_cache.set(cache_key, embeddings)

            return embeddings

        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Error getting embeddings from {model}: {str(e)}")
            raise

    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            response = await asyncio.to_thread(ollama.list)
            return response["models"]
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error listing models: {str(e)}")
            return []


# Create a global instance
llm_client = EnhancedLLMClient()
LLMClient = EnhancedLLMClient  # Alias for backwards compatibility
