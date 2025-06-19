import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..core.llm_client import LLMCache, llm_client

logger = logging.getLogger(__name__)


class ModelComplexity(str, Enum):
    """Complexity levels for model selection"""

    SIMPLE = "simple"  # For simple queries, greetings
    STANDARD = "standard"  # For most conversational queries
    COMPLEX = "complex"  # For reasoning, coding, creative tasks
    CRITICAL = "critical"  # For safety-critical operations


class ModelConfig(BaseModel):
    """Configuration for a specific LLM model"""

    name: str
    complexity_levels: List[ModelComplexity]
    max_tokens: int
    cost_per_token: float = 0.0  # Cost per 1000 tokens
    priority: int = 1  # Lower number means higher priority
    concurrency_limit: int = 10  # Max concurrent requests
    is_enabled: bool = True
    supports_streaming: bool = True
    supports_embedding: bool = False
    description: Optional[str] = None


class ModelUsageStats(BaseModel):
    """Usage statistics for a model"""

    total_requests: int = 0
    total_tokens: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    last_error: Optional[str] = None
    last_used: datetime = datetime.now()


class ModelSelectionMetrics(BaseModel):
    """Metrics used for model selection decisions"""

    query_length: int
    complexity_score: float
    keyword_score: float
    priority_features: List[str]
    selected_model: str
    complexity_level: ModelComplexity
    selection_reason: str


class HybridLLMClient:
    """
    Enhanced LLM client with hybrid model selection, resource management,
    and automatic fallback.
    """

    def __init__(self):
        """Initialize hybrid LLM client"""
        self.base_client = llm_client
        self.model_configs: Dict[str, ModelConfig] = self._init_model_configs()
        self.model_stats: Dict[str, ModelUsageStats] = {
            name: ModelUsageStats() for name in self.model_configs.keys()
        }

        # Create semaphores for each model
        self.semaphores: Dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(config.concurrency_limit)
            for name, config in self.model_configs.items()
        }

        # Create caches for each model
        self.caches: Dict[str, LLMCache] = {
            name: LLMCache(max_size=1000, ttl=3600)  # 1 hour TTL
            for name in self.model_configs.keys()
        }

        # For metrics collection
        self.selection_metrics: List[ModelSelectionMetrics] = []
        self.last_cleanup = datetime.now()

    def _init_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations"""
        return {
            "gemma:2b": ModelConfig(
                name="gemma:2b",
                complexity_levels=[ModelComplexity.SIMPLE],
                max_tokens=4096,
                cost_per_token=0.01,
                priority=1,
                concurrency_limit=20,
            ),
            "gemma3:latest": ModelConfig(
                name="gemma3:latest",
                complexity_levels=[
                    ModelComplexity.STANDARD,
                    ModelComplexity.COMPLEX,
                    ModelComplexity.CRITICAL,
                ],
                max_tokens=32768,
                cost_per_token=0.10,
                priority=2,
                concurrency_limit=5,
            ),
            "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0": ModelConfig(
                name="SpeakLeash/bielik-11b-v2.3-instruct:Q8_0",
                complexity_levels=[ModelComplexity.COMPLEX, ModelComplexity.CRITICAL],
                max_tokens=32768,
                cost_per_token=0.15,
                priority=3,
                concurrency_limit=3,
                description="Polish language specialized model",
            ),
        }

    async def _get_complexity_level(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None
    ) -> Tuple[ModelComplexity, float, List[str]]:
        """
        Determine query complexity level based on content analysis
        Returns: (complexity_level, complexity_score, priority_features)
        """
        # Extract the user query
        query = ""
        for msg in messages:
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        if not query:
            return ModelComplexity.SIMPLE, 0.0, []

        # Initialize score and features
        complexity_score = 0.0
        priority_features = []

        # 1. Length-based complexity
        query_length = len(query)
        if query_length < 50:
            complexity_score += 0.2
        elif query_length < 200:
            complexity_score += 0.5
            priority_features.append("medium_length")
        else:
            complexity_score += 0.8
            priority_features.append("long_query")

        # 2. Keyword-based complexity
        complex_keywords = [
            "wyjaśnij",
            "porównaj",
            "przeanalizuj",
            "uzasadnij",
            "zaprojektuj",
            "utwórz",
            "zaimplementuj",
            "kod",
            "program",
            "algorytm",
            "dlaczego",
            "jakie są przyczyny",
            "co by się stało gdyby",
        ]

        critical_keywords = [
            "pilne",
            "krytyczne",
            "ważne",
            "pomoc medyczna",
            "zagrożenie",
            "alert",
            "przeprowadź analizę",
            "złożony problem",
            "optymalizacja",
            "zrównoleglenie",
        ]

        keyword_score = 0.0

        for keyword in complex_keywords:
            if keyword.lower() in query.lower():
                keyword_score += 0.2
                priority_features.append(f"complex_term:{keyword}")

        for keyword in critical_keywords:
            if keyword.lower() in query.lower():
                keyword_score += 0.4
                priority_features.append(f"critical_term:{keyword}")

        # Cap keyword score
        keyword_score = min(1.0, keyword_score)
        complexity_score += keyword_score

        # 3. Pattern-based complexity
        if re.search(r"[\[\]\(\)\{\}]", query):  # Code-like characters
            complexity_score += 0.3
            priority_features.append("code_syntax")

        if query.count("\n") > 3:  # Multi-line text often indicates complex request
            complexity_score += 0.3
            priority_features.append("multi_line")

        if re.search(
            r"\d+[\.,]\d+", query
        ):  # Numbers with decimals often in technical content
            complexity_score += 0.2
            priority_features.append("technical_numbers")

        # 4. System prompt complexity (if provided)
        if system_prompt:
            if len(system_prompt) > 200:
                complexity_score += 0.3
                priority_features.append("complex_system_prompt")

        # 5. History complexity (not implemented yet - would need full conversation history)

        # Normalize final score (0.0 - 1.0)
        normalized_score = min(1.0, complexity_score / 3.0)

        # Map score to complexity level
        if normalized_score < 0.3:
            complexity_level = ModelComplexity.SIMPLE
        elif normalized_score < 0.6:
            complexity_level = ModelComplexity.STANDARD
        elif normalized_score < 0.85:
            complexity_level = ModelComplexity.COMPLEX
        else:
            complexity_level = ModelComplexity.CRITICAL

        return complexity_level, normalized_score, priority_features

    def _select_model(
        self,
        complexity_level: ModelComplexity,
        complexity_score: float,
        priority_features: List[str],
    ) -> Tuple[str, str]:
        """
        Select appropriate model based on complexity level and available resources
        Returns: (model_name, selection_reason)
        """
        # Get all models supporting this complexity level
        eligible_models = [
            (name, config)
            for name, config in self.model_configs.items()
            if config.is_enabled and complexity_level in config.complexity_levels
        ]

        if not eligible_models:
            # Fallback to standard model if no models support the complexity level
            fallback_msg = (
                f"No models support {complexity_level}, falling back to gemma3:12b"
            )
            logger.warning(fallback_msg)
            return "gemma3:12b", fallback_msg

        # Sort by priority (lower number = higher priority)
        eligible_models.sort(key=lambda x: x[1].priority)

        # Check semaphore availability
        for name, config in eligible_models:
            if self.semaphores[name]._value > 0:  # Semaphore has available slots
                reason = f"Selected {name} for {complexity_level} query (score: {complexity_score:.2f})"
                return name, reason

        # If all appropriate models are at capacity, use the highest priority one anyway
        name, config = eligible_models[0]
        reason = f"Selected {name} for {complexity_level} query despite capacity limit"
        logger.warning(
            f"All models for {complexity_level} at capacity, using {name} anyway"
        )
        return name, reason

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        force_complexity: Optional[ModelComplexity] = None,
    ) -> Any:
        """
        Enhanced chat method with automatic model selection
        """
        start_time = time.time()

        try:
            # Auto-determine complexity if model not explicitly provided
            if not model:
                if force_complexity:
                    complexity_level = force_complexity
                    complexity_score = (
                        1.0 if complexity_level == ModelComplexity.CRITICAL else 0.5
                    )
                    priority_features = [f"forced_{complexity_level.value}"]
                else:
                    (
                        complexity_level,
                        complexity_score,
                        priority_features,
                    ) = await self._get_complexity_level(messages, system_prompt)

                # Select appropriate model
                model, selection_reason = self._select_model(
                    complexity_level, complexity_score, priority_features
                )

                # Record selection metrics
                query = next(
                    (
                        msg.get("content", "")
                        for msg in messages
                        if msg.get("role") == "user"
                    ),
                    "",
                )
                self.selection_metrics.append(
                    ModelSelectionMetrics(
                        query_length=len(query),
                        complexity_score=complexity_score,
                        keyword_score=complexity_score,  # Simplified
                        priority_features=priority_features,
                        selected_model=model,
                        complexity_level=complexity_level,
                        selection_reason=selection_reason,
                    )
                )

                logger.info(f"Auto-selected model: {model} ({selection_reason})")
            else:
                # Ensure the model exists in our configs
                if model not in self.model_configs:
                    logger.warning(
                        f"Unknown model: {model}, falling back to gemma3:12b"
                    )
                    model = "gemma3:12b"

            # Check model stats for disabled models
            if not self.model_configs[model].is_enabled:
                fallback_model = "gemma3:12b"
                logger.warning(
                    f"Model {model} is disabled, falling back to {fallback_model}"
                )
                model = fallback_model

            # Update usage stats
            model_stats = self.model_stats[model]
            model_stats.total_requests += 1
            model_stats.last_used = datetime.now()

            # If system prompt provided, integrate it into messages
            if system_prompt and not any(
                msg.get("role") == "system" for msg in messages
            ):
                messages = [{"role": "system", "content": system_prompt}] + messages

            # Process with resource limiting
            async with self.semaphores[model]:
                if stream:
                    # For streaming, we need to wrap the generator
                    return self._wrap_streaming_response(model, messages, options)
                else:
                    # For normal requests
                    response = await self.base_client.chat(
                        model=model, messages=messages, stream=False, options=options
                    )

                    # Update success stats
                    if response and not isinstance(response, Exception):
                        model_stats.successful_requests += 1
                        # Estimate tokens (rough approximation)
                        if "message" in response and "content" in response["message"]:
                            tokens = len(response["message"]["content"]) // 4
                            model_stats.total_tokens += tokens
                    else:
                        model_stats.failed_requests += 1
                        if isinstance(response, Exception):
                            model_stats.last_error = str(response)

                    # Update latency
                    latency = time.time() - start_time
                    model_stats.average_latency = (
                        (
                            model_stats.average_latency
                            * (model_stats.successful_requests - 1)
                            + latency
                        )
                        / model_stats.successful_requests
                        if model_stats.successful_requests > 0
                        else latency
                    )

                    return response

        except Exception as error_msg:
            logger.error(f"Error in hybrid chat: {str(error_msg)}")

            if model and model in self.model_stats:
                self.model_stats[model].failed_requests += 1
                self.model_stats[model].last_error = str(error_msg)

            # Return error response in expected format
            return {
                "message": {"content": f"Error processing request: {str(error_msg)}"},
                "response": f"Error processing request: {str(error_msg)}",
            }

    async def _wrap_streaming_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Wrap streaming response with stats tracking"""
        start_time = time.time()
        chunks_count = 0
        total_length = 0

        try:
            # Get streaming response from base client
            stream = await self.base_client.chat(
                model=model, messages=messages, stream=True, options=options
            )

            # Process each chunk
            async for chunk in stream:
                chunks_count += 1
                if (
                    isinstance(chunk, dict)
                    and "message" in chunk
                    and "content" in chunk["message"]
                ):
                    total_length += len(chunk["message"]["content"])
                yield chunk

            # Update success stats
            self.model_stats[model].successful_requests += 1
            # Estimate tokens (rough approximation)
            tokens = total_length // 4
            self.model_stats[model].total_tokens += tokens

            # Update latency
            latency = time.time() - start_time
            model_stats = self.model_stats[model]
            model_stats.average_latency = (
                (
                    model_stats.average_latency * (model_stats.successful_requests - 1)
                    + latency
                )
                / model_stats.successful_requests
                if model_stats.successful_requests > 0
                else latency
            )

        except Exception as error_msg:
            logger.error(f"Error in streaming response: {str(error_msg)}")
            self.model_stats[model].failed_requests += 1
            self.model_stats[model].last_error = str(error_msg)

            # Yield error message
            yield {
                "message": {"content": f"Error during streaming: {str(error_msg)}"},
                "response": f"Error during streaming: {str(error_msg)}",
            }

    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embeddings with automatic model selection"""
        if not model:
            # Use smallest model that supports embeddings
            embedding_models = [
                name
                for name, config in self.model_configs.items()
                if config.is_enabled and config.supports_embedding
            ]

            if embedding_models:
                model = embedding_models[0]
            else:
                # Default to standard model
                model = "gemma3:12b"

        # Ensure the model exists
        if model not in self.model_configs:
            logger.warning(
                f"Unknown embedding model: {model}, falling back to gemma3:12b"
            )
            model = "gemma3:12b"

        # Update usage stats
        self.model_stats[model].total_requests += 1
        self.model_stats[model].last_used = datetime.now()

        try:
            # Use semaphore for resource control
            async with self.semaphores[model]:
                embeddings = await self.base_client.embed(model=model, text=text)

                # Update success stats
                self.model_stats[model].successful_requests += 1

                return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            self.model_stats[model].failed_requests += 1
            self.model_stats[model].last_error = str(e)
            return []

    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}

        for name, config in self.model_configs.items():
            stats = self.model_stats[name]
            semaphore = self.semaphores[name]

            status[name] = {
                "enabled": config.is_enabled,
                "available_slots": semaphore._value,
                "max_concurrency": config.concurrency_limit,
                "total_requests": stats.total_requests,
                "success_rate": (
                    stats.successful_requests / stats.total_requests
                    if stats.total_requests > 0
                    else 1.0
                ),
                "average_latency": stats.average_latency,
                "last_used": stats.last_used.isoformat(),
                "complexity_levels": [
                    level.value for level in config.complexity_levels
                ],
            }

        return status

    def maintenance(self) -> Dict[str, Any]:
        """Perform maintenance tasks and return stats"""
        # Cleanup old metrics
        now = datetime.now()
        if now - self.last_cleanup > timedelta(hours=1):
            # Keep only the last 100 metrics
            if len(self.selection_metrics) > 100:
                self.selection_metrics = self.selection_metrics[-100:]
            self.last_cleanup = now

        # Collect stats
        return {
            "models": self.get_models_status(),
            "total_requests": sum(
                stats.total_requests for stats in self.model_stats.values()
            ),
            "total_tokens": sum(
                stats.total_tokens for stats in self.model_stats.values()
            ),
            "metrics_collected": len(self.selection_metrics),
        }

    async def with_retry_fallback(
        self,
        messages: List[Dict[str, str]],
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        max_retries: int = 2,
        stream: bool = False,
    ) -> Any:
        """Try with primary model, falling back to simpler model if necessary"""
        error_msg = None

        # Determine complexity if no primary model specified
        if not primary_model:
            complexity_level, _, _ = await self._get_complexity_level(messages)
            primary_model, _ = self._select_model(complexity_level, 0.5, [])

        # Try primary model
        for attempt in range(max_retries):
            try:
                return await self.chat(
                    messages=messages, model=primary_model, stream=stream
                )
            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Error with primary model {primary_model}, attempt {attempt+1}: {str(e)}"
                )
                await asyncio.sleep(0.5 * (2**attempt))

        # If fallback model not specified, choose one simpler than primary
        if not fallback_model:
            primary_priority = self.model_configs[primary_model].priority
            fallback_candidates = [
                name
                for name, config in self.model_configs.items()
                if config.is_enabled and config.priority < primary_priority
            ]

            if fallback_candidates:
                fallback_model = fallback_candidates[0]
            else:
                fallback_model = "gemma2:2b"  # Simplest model as last resort

        # Try fallback model
        logger.warning(
            f"Falling back to {fallback_model} after {max_retries} failed attempts with {primary_model}"
        )

        try:
            return await self.chat(
                messages=messages, model=fallback_model, stream=stream
            )
        except Exception as e:
            logger.error(f"Error with fallback model {fallback_model}: {str(e)}")
            # If all else fails, return error
            return {
                "message": {
                    "content": f"Error processing request with all models: {str(e)}"
                },
                "response": f"Error processing request with all models: {str(e)}",
            }


# Initialize hybrid client
hybrid_llm_client = HybridLLMClient()
