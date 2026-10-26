"""
Tests for the metrics module.
"""

import pytest
from src.metrics import (
    calculate_accuracy,
    calculate_consistency,
    calculate_parse_rate,
    calculate_cost_metrics,
    evaluate_strategy
)


def test_calculate_accuracy_perfect():
    """Test accuracy calculation with perfect predictions."""
    predictions = ["positive", "negative", "positive"]
    labels = [1, 0, 1]
    assert calculate_accuracy(predictions, labels) == 1.0


def test_calculate_accuracy_half():
    """Test accuracy calculation with 50% accuracy."""
    predictions = ["positive", "negative", "negative", "positive"]
    labels = [1, 0, 1, 0]
    assert calculate_accuracy(predictions, labels) == 0.5


def test_calculate_accuracy_with_unknown():
    """Test accuracy ignores unknown predictions."""
    predictions = ["positive", "unknown", "negative"]
    labels = [1, 0, 0]
    # Only 2 valid predictions: 1 correct (positive=1), 1 correct (negative=0)
    assert calculate_accuracy(predictions, labels) == 1.0


def test_calculate_consistency_perfect():
    """Test consistency with identical predictions across runs."""
    predictions_list = [
        ["positive", "negative", "positive"],
        ["positive", "negative", "positive"],
        ["positive", "negative", "positive"]
    ]
    labels = [1, 0, 1]
    assert calculate_consistency(predictions_list, labels) == 1.0


def test_calculate_consistency_variable():
    """Test consistency with some variation."""
    predictions_list = [
        ["positive", "negative", "positive"],
        ["positive", "positive", "positive"],
        ["positive", "negative", "positive"]
    ]
    labels = [1, 0, 1]
    # First and third are consistent (2/3)
    consistency = calculate_consistency(predictions_list, labels)
    assert abs(consistency - 2/3) < 0.01


def test_calculate_parse_rate_perfect():
    """Test parse rate with all successful parses."""
    predictions = ["positive", "negative", "positive"]
    assert calculate_parse_rate(predictions) == 1.0


def test_calculate_parse_rate_with_failures():
    """Test parse rate with some failures."""
    predictions = ["positive", "unknown", "negative", "unknown"]
    assert calculate_parse_rate(predictions) == 0.5


def test_calculate_cost_metrics():
    """Test cost metrics calculation."""
    results = [
        {
            "cached": False,
            "latency_ms": 100,
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        },
        {
            "cached": False,
            "latency_ms": 200,
            "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65}
        },
        {
            "cached": True,
            "latency_ms": 0,
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }
    ]
    
    metrics = calculate_cost_metrics(results)
    
    assert metrics["mean_latency_ms"] == 150.0  # (100 + 200) / 2, ignoring cached
    assert metrics["median_latency_ms"] == 150.0
    assert metrics["total_prompt_tokens"] == 100  # 2 uncached requests
    assert metrics["total_completion_tokens"] == 25
    assert metrics["total_tokens"] == 125


def test_evaluate_strategy():
    """Test full strategy evaluation."""
    predictions = ["positive", "negative", "positive"]
    labels = [1, 0, 1]
    results = [
        {
            "cached": False,
            "latency_ms": 100,
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }
    ] * 3
    
    metrics = evaluate_strategy(predictions, labels, results)
    
    assert "accuracy" in metrics
    assert "parse_rate" in metrics
    assert "mean_latency_ms" in metrics
    assert "total_tokens" in metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["parse_rate"] == 1.0
