"""
Main evaluation pipeline for comparing prompt strategies.
"""

from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm

from .dataset import get_dataset, get_few_shot_examples
from .prompts import get_strategy
from .llm_client import CachedLLMClient
from .parser import parse_response
from .metrics import evaluate_strategy, calculate_consistency


def run_evaluation(
    client: CachedLLMClient,
    strategy_name: str,
    texts: List[str],
    labels: List[int],
    seed: int = 42,
    few_shot_examples: List[Tuple[str, int]] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Run evaluation for a single strategy.
    
    Args:
        client: LLM client with caching
        strategy_name: Name of prompting strategy
        texts: Input texts to classify
        labels: True labels
        seed: Random seed for reproducibility
        few_shot_examples: Examples for few-shot prompting (if applicable)
        
    Returns:
        Tuple of (predictions, api_results)
    """
    strategy_fn = get_strategy(strategy_name)
    predictions = []
    api_results = []
    
    for text in tqdm(texts, desc=f"Evaluating {strategy_name}", leave=False):
        # Generate prompts
        if strategy_name == "few_shot" and few_shot_examples:
            system, user = strategy_fn(text, few_shot_examples)
        else:
            system, user = strategy_fn(text)
        
        # Get LLM response
        result = client.complete(system, user, seed=seed)
        
        # Parse prediction
        prediction = parse_response(result['response'], strategy_name)
        
        predictions.append(prediction)
        api_results.append(result)
    
    return predictions, api_results


def run_multi_seed_evaluation(
    client: CachedLLMClient,
    strategy_name: str,
    texts: List[str],
    labels: List[int],
    seeds: List[int] = [42, 43, 44],
    few_shot_examples: List[Tuple[str, int]] = None
) -> Dict[str, Any]:
    """
    Run evaluation across multiple seeds to measure consistency.
    
    Args:
        client: LLM client
        strategy_name: Name of prompting strategy
        texts: Input texts
        labels: True labels
        seeds: List of random seeds to try
        few_shot_examples: Examples for few-shot prompting
        
    Returns:
        Dict with aggregated metrics and per-seed results
    """
    all_predictions = []
    all_results = []
    
    for seed in seeds:
        predictions, results = run_evaluation(
            client, strategy_name, texts, labels, seed, few_shot_examples
        )
        all_predictions.append(predictions)
        all_results.extend(results)
    
    # Use first seed's predictions for main metrics
    main_predictions = all_predictions[0]
    
    # Calculate metrics
    metrics = evaluate_strategy(main_predictions, labels, all_results)
    
    # Add consistency metric
    metrics['consistency'] = calculate_consistency(all_predictions, labels)
    metrics['n_seeds'] = len(seeds)
    
    return {
        'strategy': strategy_name,
        'metrics': metrics,
        'predictions': all_predictions,
        'results': all_results
    }


def compare_strategies(
    client: CachedLLMClient,
    strategy_names: List[str],
    seeds: List[int] = [42, 43, 44],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare multiple prompting strategies.
    
    Args:
        client: LLM client
        strategy_names: List of strategy names to compare
        seeds: Random seeds for consistency evaluation
        verbose: Whether to print progress
        
    Returns:
        Dict with results for each strategy
    """
    texts, labels = get_dataset()
    few_shot_examples = get_few_shot_examples(n=3)
    
    results = {}
    
    for strategy_name in strategy_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating strategy: {strategy_name}")
            print(f"{'='*60}")
        
        strategy_results = run_multi_seed_evaluation(
            client=client,
            strategy_name=strategy_name,
            texts=texts,
            labels=labels,
            seeds=seeds,
            few_shot_examples=few_shot_examples
        )
        
        results[strategy_name] = strategy_results
        
        if verbose:
            metrics = strategy_results['metrics']
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Consistency: {metrics['consistency']:.3f}")
            print(f"Parse rate: {metrics['parse_rate']:.3f}")
            print(f"Mean latency: {metrics['mean_latency_ms']:.0f} ms")
            print(f"Total tokens: {metrics['total_tokens']:,}")
    
    return results
