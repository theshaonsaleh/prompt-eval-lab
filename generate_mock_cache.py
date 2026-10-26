"""
Generate mock cached responses for offline mode demonstration.
This allows the notebook to run without requiring an API key.
"""

import json
import hashlib
import random
from pathlib import Path

from src.dataset import get_dataset, get_few_shot_examples
from src.prompts import get_strategy

# Create cache directory
cache_dir = Path("results/cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Get dataset
texts, labels = get_dataset()
few_shot_examples = get_few_shot_examples(n=3)

# Model to simulate
MODEL = "gpt-3.5-turbo"

def generate_cache_key(system: str, user: str, seed: int) -> str:
    """Generate cache key (same logic as in llm_client.py)."""
    content = f"{MODEL}||{system}||{user}||{seed}"
    return hashlib.sha256(content.encode()).hexdigest()

def generate_mock_response(text: str, true_label: int, strategy: str, seed: int) -> dict:
    """Generate a realistic mock response."""
    # Introduce some intentional errors for realism (10% error rate base)
    random.seed(seed + hash(text))
    
    # Higher accuracy for zero_shot and few_shot, lower for others
    if strategy == "zero_shot":
        accuracy = 0.85
    elif strategy == "few_shot":
        accuracy = 0.88
    elif strategy == "chain_of_thought":
        accuracy = 0.83
    elif strategy == "structured_json":
        accuracy = 0.81
    else:
        accuracy = 0.80
    
    # Decide if correct or not
    is_correct = random.random() < accuracy
    predicted_label = true_label if is_correct else (1 - true_label)
    sentiment = "positive" if predicted_label == 1 else "negative"
    
    # Generate response based on strategy
    if strategy == "zero_shot":
        response = sentiment
    elif strategy == "few_shot":
        response = sentiment
    elif strategy == "chain_of_thought":
        # Add reasoning
        key_words = "love, great, wonderful, fantastic, happy" if predicted_label == 1 else "hate, terrible, awful, sad, disappointing"
        response = f"""Let me analyze this step by step:

1. Key sentiment words: The text contains words like {key_words.split(', ')[0]} which indicate {sentiment} sentiment.
2. Overall tone: The tone is clearly {sentiment}.
3. Final sentiment: {sentiment}

Final answer: {sentiment}"""
    else:  # structured_json
        response = f'{{"sentiment": "{sentiment}"}}'
    
    # Generate realistic token counts
    prompt_tokens = len(system.split()) + len(user.split()) + 50
    if strategy == "chain_of_thought":
        completion_tokens = 60 + random.randint(-10, 10)
    elif strategy == "structured_json":
        completion_tokens = 8 + random.randint(-2, 2)
    else:
        completion_tokens = 3 + random.randint(-1, 1)
    
    # Generate realistic latency (ms)
    base_latency = 200
    if strategy == "chain_of_thought":
        latency = base_latency + 150 + random.randint(-50, 50)
    else:
        latency = base_latency + random.randint(-50, 100)
    
    return {
        "response": response,
        "cached": False,
        "latency_ms": latency,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "model": MODEL,
        "seed": seed
    }

# Generate cache for all strategies and seeds
strategies = ["zero_shot", "few_shot", "chain_of_thought", "structured_json"]
seeds = [42, 43, 44]

print("Generating mock cached responses...")
print()

total_cached = 0

for strategy_name in strategies:
    print(f"Strategy: {strategy_name}")
    strategy_fn = get_strategy(strategy_name)
    
    for seed in seeds:
        for i, (text, label) in enumerate(zip(texts, labels)):
            # Generate prompts
            if strategy_name == "few_shot":
                system, user = strategy_fn(text, few_shot_examples)
            else:
                system, user = strategy_fn(text)
            
            # Generate cache key
            cache_key = generate_cache_key(system, user, seed)
            cache_path = cache_dir / f"{cache_key}.json"
            
            # Generate mock response
            mock_data = generate_mock_response(text, label, strategy_name, seed)
            
            # Save to cache
            with open(cache_path, 'w') as f:
                json.dump(mock_data, f, indent=2)
            
            total_cached += 1
    
    print(f"  Generated {len(texts) * len(seeds)} cached responses")

print()
print(f"Total cached responses: {total_cached}")
print(f"Cache directory: {cache_dir}")
print()
print("✓ Mock cache generated successfully!")
print("  The notebook can now run in offline mode.")
