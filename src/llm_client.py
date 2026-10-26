"""
LLM API client with caching for reproducible experiments.
Supports OpenAI-compatible APIs and offline replay mode.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class CachedLLMClient:
    """
    LLM client with response caching for reproducibility.
    
    Caches API responses keyed by (model, system_prompt, user_prompt, seed).
    Supports offline mode that only reads from cache.
    """
    
    def __init__(
        self,
        cache_dir: str = "results/cache",
        offline_mode: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode
        self.model = model
        
        if not offline_mode:
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
            if not api_key:
                raise ValueError(
                    "API key required in online mode. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = None
    
    def _cache_key(self, system: str, user: str, seed: int) -> str:
        """Generate cache key from prompts and seed."""
        content = f"{self.model}||{system}||{user}||{seed}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.json"
    
    def _load_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if exists."""
        cache_path = self._cache_path(key)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, key: str, data: Dict[str, Any]):
        """Save response to cache."""
        cache_path = self._cache_path(key)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        seed: int = 42,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get completion from LLM with caching.
        
        Returns dict with:
            - response: the model's text response
            - cached: whether this was from cache
            - latency_ms: response time in milliseconds (0 if cached)
            - usage: token usage info (if available)
        """
        cache_key = self._cache_key(system_prompt, user_prompt, seed)
        
        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            cached_data['cached'] = True
            cached_data['latency_ms'] = 0
            return cached_data
        
        # In offline mode, fail if not cached
        if self.offline_mode:
            raise ValueError(
                f"Response not found in cache and offline_mode=True. "
                f"Cache key: {cache_key}"
            )
        
        # Make API call
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            seed=seed
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract response
        result = {
            "response": response.choices[0].message.content,
            "cached": False,
            "latency_ms": latency_ms,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            "model": response.model,
            "seed": seed
        }
        
        # Save to cache
        self._save_to_cache(cache_key, result)
        
        return result


def create_client(
    offline_mode: bool = False,
    model: str = "gpt-3.5-turbo"
) -> CachedLLMClient:
    """
    Create LLM client with automatic configuration from environment.
    
    In online mode, reads OPENAI_API_KEY and optionally OPENAI_BASE_URL from environment.
    In offline mode, only uses cached responses.
    """
    if offline_mode:
        return CachedLLMClient(offline_mode=True, model=model)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Either set the environment variable or run in offline mode."
        )
    
    base_url = os.getenv("OPENAI_BASE_URL")
    
    return CachedLLMClient(
        api_key=api_key,
        base_url=base_url,
        offline_mode=False,
        model=model
    )
