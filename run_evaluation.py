"""
Run the complete evaluation and generate cached responses.
This script runs the evaluation once to populate the cache,
allowing the notebook to run in offline mode.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import create_client, compare_strategies

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set.")
    print("Please set your API key in .env file or environment variable.")
    print("See .env.example for template.")
    sys.exit(1)

print("Starting evaluation...")
print("This will make API calls and cache responses for reproducibility.")
print()

# Create client
client = create_client(offline_mode=False, model="gpt-3.5-turbo")

# Run evaluation on all strategies
strategies = ["zero_shot", "few_shot", "chain_of_thought", "structured_json"]
results = compare_strategies(
    client=client,
    strategy_names=strategies,
    seeds=[42, 43, 44],
    verbose=True
)

print()
print("="*60)
print("Evaluation complete!")
print("="*60)
print()
print("Results summary:")
for strategy, data in results.items():
    metrics = data['metrics']
    print(f"\n{strategy}:")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Consistency: {metrics['consistency']:.3f}")
    print(f"  Parse rate:  {metrics['parse_rate']:.3f}")
    print(f"  Mean latency: {metrics['mean_latency_ms']:.0f} ms")
    print(f"  Total tokens: {metrics['total_tokens']:,}")

print()
print("Cached responses saved to results/mock_cache/")
print("The notebook can now run in offline mode using these cached responses.")
