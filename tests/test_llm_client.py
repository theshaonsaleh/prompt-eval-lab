"""
Tests for the LLM client caching functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.llm_client import CachedLLMClient


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_cache_key_generation(temp_cache_dir):
    """Test that cache keys are generated consistently."""
    client = CachedLLMClient(cache_dir=temp_cache_dir, offline_mode=True)
    
    key1 = client._cache_key("system1", "user1", 42)
    key2 = client._cache_key("system1", "user1", 42)
    key3 = client._cache_key("system1", "user1", 43)
    
    assert key1 == key2  # Same inputs should give same key
    assert key1 != key3  # Different seed should give different key


def test_save_and_load_cache(temp_cache_dir):
    """Test saving and loading from cache."""
    client = CachedLLMClient(cache_dir=temp_cache_dir, offline_mode=True)
    
    key = client._cache_key("system", "user", 42)
    data = {"response": "positive", "cached": False}
    
    client._save_to_cache(key, data)
    loaded = client._load_from_cache(key)
    
    assert loaded is not None
    assert loaded["response"] == "positive"


def test_offline_mode_with_cache(temp_cache_dir):
    """Test that offline mode uses cached responses."""
    client = CachedLLMClient(cache_dir=temp_cache_dir, offline_mode=True)
    
    # Manually add a cached response
    key = client._cache_key("system", "user", 42)
    cached_data = {
        "response": "positive",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    client._save_to_cache(key, cached_data)
    
    # Request should use cache
    result = client.complete("system", "user", seed=42)
    assert result["response"] == "positive"
    assert result["cached"] is True
    assert result["latency_ms"] == 0


def test_offline_mode_without_cache(temp_cache_dir):
    """Test that offline mode fails without cache."""
    client = CachedLLMClient(cache_dir=temp_cache_dir, offline_mode=True)
    
    with pytest.raises(ValueError, match="not found in cache"):
        client.complete("system", "user", seed=42)


def test_cache_directory_creation():
    """Test that cache directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "cache" / "nested"
        client = CachedLLMClient(cache_dir=str(cache_path), offline_mode=True)
        
        assert cache_path.exists()
        assert cache_path.is_dir()
