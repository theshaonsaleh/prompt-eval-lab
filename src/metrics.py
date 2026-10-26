"""
Evaluation metrics for prompt strategies.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_accuracy(predictions: List[str], labels: List[int]) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: List of predicted sentiments ("positive", "negative", or "unknown")
        labels: List of true labels (1 for positive, 0 for negative)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    
    correct = 0
    total = 0
    
    for pred, label in zip(predictions, labels):
        if pred == "unknown":
            continue
        
        pred_label = 1 if pred == "positive" else 0
        if pred_label == label:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def calculate_consistency(predictions_list: List[List[str]], labels: List[int]) -> float:
    """
    Calculate consistency score across multiple runs.
    
    Consistency is the fraction of examples where all runs produce the same prediction.
    
    Args:
        predictions_list: List of prediction lists from multiple runs
        labels: True labels (for filtering valid predictions)
        
    Returns:
        Consistency score between 0 and 1
    """
    if not predictions_list:
        return 0.0
    
    n_examples = len(labels)
    consistent_count = 0
    
    for i in range(n_examples):
        preds_for_example = [preds[i] for preds in predictions_list]
        # Filter out "unknown" predictions
        valid_preds = [p for p in preds_for_example if p != "unknown"]
        
        if len(valid_preds) > 0 and len(set(valid_preds)) == 1:
            consistent_count += 1
    
    return consistent_count / n_examples


def calculate_parse_rate(predictions: List[str]) -> float:
    """
    Calculate the rate of successfully parsed responses.
    
    Args:
        predictions: List of predictions (including "unknown" for parse failures)
        
    Returns:
        Parse success rate between 0 and 1
    """
    parsed = sum(1 for p in predictions if p != "unknown")
    return parsed / len(predictions) if predictions else 0.0


def calculate_cost_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate cost and latency metrics from API responses.
    
    Args:
        results: List of API response dicts with 'usage' and 'latency_ms' keys
        
    Returns:
        Dict with mean/median latency and total token usage
    """
    latencies = [r['latency_ms'] for r in results if not r.get('cached', False)]
    
    total_prompt_tokens = sum(
        r['usage']['prompt_tokens'] 
        for r in results 
        if r.get('usage') and not r.get('cached', False)
    )
    total_completion_tokens = sum(
        r['usage']['completion_tokens'] 
        for r in results 
        if r.get('usage') and not r.get('cached', False)
    )
    
    return {
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens
    }


def evaluate_strategy(
    predictions: List[str],
    labels: List[int],
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a strategy.
    
    Args:
        predictions: Predicted sentiments
        labels: True labels
        results: API response metadata
        
    Returns:
        Dict of evaluation metrics
    """
    metrics = {
        "accuracy": calculate_accuracy(predictions, labels),
        "parse_rate": calculate_parse_rate(predictions),
    }
    
    cost_metrics = calculate_cost_metrics(results)
    metrics.update(cost_metrics)
    
    return metrics
