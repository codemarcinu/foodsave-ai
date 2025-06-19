import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "search_cache"
)
DEFAULT_TTL = 600  # 10 minutes in seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Start with 1 second delay, doubles each retry
REQUEST_TIMEOUT = 10.0  # 10 seconds


# Sources configuration
class SourceConfig(BaseModel):
    """Configuration for a search source"""

    name: str
    enabled: bool = True
    api_key_env_var: str
    base_url: str
    api_key: Optional[str] = None
    quota_per_day: Optional[int] = None
    quota_per_minute: Optional[int] = None
    priority: int = 1  # Lower number = higher priority
    whitelist: List[str] = []
    blacklist: List[str] = []


class SearchResult(BaseModel):
    """Search result model"""

    title: str
    url: str
    snippet: str
    source: str
    date: Optional[str] = None
    source_confidence: float = 1.0


class SearchResponse(BaseModel):
    """API response for search"""

    query: str
    results: List[SearchResult]
    timestamp: str
    source: str
    cached: bool = False


class WebSearchClient:
    """Client for making API calls to search engines with caching and source verification"""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        ttl: int = DEFAULT_TTL,
        sources_config: Optional[List[Dict[str, Any]]] = None,
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.sources: List[SourceConfig] = []
        self.source_usage: Dict[str, Dict[str, Any]] = {}
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize sources
        self._init_sources(sources_config)

    def _init_sources(
        self, sources_config: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Initialize search sources from config"""
        default_sources = [
            {
                "name": "newsapi",
                "enabled": True,
                "api_key_env_var": "NEWS_API_KEY",
                "base_url": "https://newsapi.org/v2",
                "quota_per_day": 100,
                "priority": 1,
                "whitelist": ["bbc.co.uk", "cnn.com", "reuters.com", "nytimes.com"],
            },
            {
                "name": "bing",
                "enabled": True,
                "api_key_env_var": "BING_SEARCH_API_KEY",
                "base_url": "https://api.bing.microsoft.com/v7.0/search",
                "quota_per_minute": 3,
                "priority": 2,
            },
        ]

        config_to_use = sources_config or default_sources

        for src_config in config_to_use:
            source = SourceConfig(**src_config)

            # Get API key from environment
            source.api_key = os.environ.get(source.api_key_env_var)

            if source.api_key:
                self.sources.append(source)
                # Initialize usage tracking
                self.source_usage[source.name] = {
                    "daily_count": 0,
                    "minute_count": 0,
                    "last_minute": datetime.now().minute,
                    "last_reset_day": datetime.now().day,
                    "errors": 0,
                    "last_error": None,
                }
            else:
                logger.warning(
                    f"No API key found for {source.name} (env var: {source.api_key_env_var})"
                )

        # Sort sources by priority
        self.sources.sort(key=lambda x: x.priority)

        logger.info(f"Initialized {len(self.sources)} search sources")

    def _get_cache_path(self, query: str) -> str:
        """Get cache file path for a query"""
        # Create a hash of the query to use as filename
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.json")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file exists and is within TTL"""
        if not os.path.exists(cache_path):
            return False

        # Check file modification time
        mod_time = os.path.getmtime(cache_path)
        age = time.time() - mod_time

        return age < self.ttl

    async def _load_from_cache(self, query: str) -> Optional[SearchResponse]:
        """Load search results from cache if available"""
        cache_path = self._get_cache_path(query)

        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)

                response = SearchResponse(**data)
                response.cached = True

                logger.info(f"Loaded results for '{query}' from cache")
                return response
            except Exception as e:
                logger.error(f"Error loading from cache: {e}")

        return None

    async def _save_to_cache(self, response: SearchResponse) -> None:
        """Save search results to cache"""
        cache_path = self._get_cache_path(response.query)

        try:
            with open(cache_path, "w") as f:
                json.dump(response.dict(), f)

            logger.debug(f"Saved results for '{response.query}' to cache")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    async def _make_request(
        self, source: SourceConfig, query: str, retries: int = MAX_RETRIES
    ) -> Optional[Dict[str, Any]]:
        """Make a request to a search API with retries"""
        # Track usage
        source_stats = self.source_usage[source.name]

        # Reset daily count if day changed
        current_day = datetime.now().day
        if source_stats["last_reset_day"] != current_day:
            source_stats["daily_count"] = 0
            source_stats["last_reset_day"] = current_day

        # Reset minute count if minute changed
        current_minute = datetime.now().minute
        if source_stats["last_minute"] != current_minute:
            source_stats["minute_count"] = 0
            source_stats["last_minute"] = current_minute

        # Check quota limits
        if source.quota_per_day and source_stats["daily_count"] >= source.quota_per_day:
            logger.warning(f"Daily quota exceeded for {source.name}")
            return None

        if (
            source.quota_per_minute
            and source_stats["minute_count"] >= source.quota_per_minute
        ):
            logger.warning(f"Minute quota exceeded for {source.name}")
            return None

        # Prepare request based on source type
        if source.name == "newsapi":
            url = f"{source.base_url}/everything"
            params = {
                "q": query,
                "apiKey": source.api_key,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 10,
            }
            headers = {}
        elif source.name == "bing":
            url = source.base_url
            params = {"q": query, "count": 10, "offset": 0, "mkt": "en-US"}
            headers = {"Ocp-Apim-Subscription-Key": source.api_key}
        else:
            logger.error(f"Unknown source type: {source.name}")
            return None

        # Make request with retries
        attempt = 0
        while attempt < retries:
            try:
                response = await self.client.get(url, params=params, headers=headers)
                response.raise_for_status()

                # Update usage counters
                source_stats["daily_count"] += 1
                source_stats["minute_count"] += 1

                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error {e.response.status_code} from {source.name}: {str(e)}"
                )

                # Special handling for rate limiting
                if e.response.status_code == 429:
                    wait_time = int(
                        e.response.headers.get(
                            "Retry-After", RETRY_DELAY * (2**attempt)
                        )
                    )
                    logger.warning(
                        f"Rate limited by {source.name}, waiting {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)

                source_stats["errors"] += 1
                source_stats["last_error"] = str(e)

            except Exception as e:
                logger.error(f"Error querying {source.name}: {str(e)}")
                source_stats["errors"] += 1
                source_stats["last_error"] = str(e)

            # Exponential backoff
            wait_time = RETRY_DELAY * (2**attempt)
            logger.info(
                f"Retrying {source.name} in {wait_time}s (attempt {attempt+1}/{retries})"
            )
            await asyncio.sleep(wait_time)
            attempt += 1

        return None

    def _parse_newsapi_response(
        self, data: Dict[str, Any], query: str
    ) -> SearchResponse:
        """Parse NewsAPI response into standard format"""
        results = []

        for article in data.get("articles", []):
            source_name = article.get("source", {}).get("name", "Unknown")

            # Apply source verification
            if not self._verify_source(source_name, "newsapi"):
                continue

            results.append(
                SearchResult(
                    title=article.get("title", ""),
                    url=article.get("url", ""),
                    snippet=article.get("description", ""),
                    source=source_name,
                    date=article.get("publishedAt", ""),
                    source_confidence=0.9,
                )
            )

        return SearchResponse(
            query=query,
            results=results,
            timestamp=datetime.now().isoformat(),
            source="newsapi",
        )

    def _parse_bing_response(self, data: Dict[str, Any], query: str) -> SearchResponse:
        """Parse Bing Search API response into standard format"""
        results = []

        for page in data.get("webPages", {}).get("value", []):
            # Extract domain from URL for source verification
            url = page.get("url", "")
            try:
                from urllib.parse import urlparse

                domain = urlparse(url).netloc
            except:
                domain = "unknown"

            # Apply source verification
            if not self._verify_source(domain, "bing"):
                continue

            results.append(
                SearchResult(
                    title=page.get("name", ""),
                    url=url,
                    snippet=page.get("snippet", ""),
                    source=domain,
                    source_confidence=0.8,
                )
            )

        return SearchResponse(
            query=query,
            results=results,
            timestamp=datetime.now().isoformat(),
            source="bing",
        )

    def _verify_source(self, source: str, provider: str) -> bool:
        """Verify if a source is trustworthy based on whitelist/blacklist"""
        source = source.lower()
        provider_config = next((s for s in self.sources if s.name == provider), None)

        if not provider_config:
            return False

        # Check blacklist
        for blocked in provider_config.blacklist:
            if blocked.lower() in source:
                logger.debug(f"Source {source} blocked by blacklist rule: {blocked}")
                return False

        # If whitelist exists, only allow sources that match
        if provider_config.whitelist:
            for allowed in provider_config.whitelist:
                if allowed.lower() in source:
                    return True
            logger.debug(f"Source {source} not in whitelist")
            return False

        # If no whitelist, any non-blacklisted source is allowed
        return True

    async def search(self, query: str, force_refresh: bool = False) -> SearchResponse:
        """Search for query across configured sources with caching"""
        # Try cache first unless forced refresh
        if not force_refresh:
            cached = await self._load_from_cache(query)
            if cached:
                return cached

        # Try each source in priority order
        for source in self.sources:
            if not source.enabled:
                continue

            logger.info(f"Searching '{query}' using {source.name}")

            try:
                raw_data = await self._make_request(source, query)

                if not raw_data:
                    continue

                # Parse based on source type
                if source.name == "newsapi":
                    response = self._parse_newsapi_response(raw_data, query)
                elif source.name == "bing":
                    response = self._parse_bing_response(raw_data, query)
                else:
                    logger.error(f"Unknown source type for parsing: {source.name}")
                    continue

                # Only return if we got meaningful results
                if response.results:
                    # Cache the results
                    await self._save_to_cache(response)
                    return response
                else:
                    logger.warning(f"No results from {source.name} for '{query}'")

            except Exception as e:
                logger.error(f"Error searching with {source.name}: {e}")

        # If all sources failed, return empty results
        logger.warning(f"All sources failed for query: '{query}'")
        return SearchResponse(
            query=query, results=[], timestamp=datetime.now().isoformat(), source="none"
        )

    async def close(self) -> None:
        """Close HTTP client"""
        await self.client.aclose()


class WebSearch:
    """Main interface for web search functionality"""

    def __init__(self, client: Optional[WebSearchClient] = None):
        self.client = client or WebSearchClient()

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search and return simplified results"""
        response = await self.client.search(query)
        return [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
            for r in response.results[:max_results]
        ]

    async def close(self) -> None:
        """Close underlying client"""
        await self.client.close()


# Singleton instances
web_search_client = WebSearchClient()
web_search = WebSearch(web_search_client)
