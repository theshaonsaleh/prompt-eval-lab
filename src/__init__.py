"""
Prompt Evaluation Lab - A reproducible framework for evaluating LLM prompt strategies.
"""

__version__ = "0.1.0"

from .dataset import get_dataset, get_few_shot_examples, SENTIMENT_DATASET
from .prompts import get_strategy, STRATEGIES
from .llm_client import create_client, CachedLLMClient
from .parser import parse_response
from .metrics import (
    calculate_accuracy,
    calculate_consistency,
    calculate_parse_rate,
    calculate_cost_metrics,
    evaluate_strategy
)
from .evaluator import (
    run_evaluation,
    run_multi_seed_evaluation,
    compare_strategies
)

__all__ = [
    # Dataset
    'get_dataset',
    'get_few_shot_examples',
    'SENTIMENT_DATASET',
    
    # Prompts
    'get_strategy',
    'STRATEGIES',
    
    # LLM Client
    'create_client',
    'CachedLLMClient',
    
    # Parser
    'parse_response',
    
    # Metrics
    'calculate_accuracy',
    'calculate_consistency',
    'calculate_parse_rate',
    'calculate_cost_metrics',
    'evaluate_strategy',
    
    # Evaluator
    'run_evaluation',
    'run_multi_seed_evaluation',
    'compare_strategies',
]
