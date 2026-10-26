"""
Tests for the dataset module.
"""

import pytest
from src.dataset import get_dataset, get_few_shot_examples, SENTIMENT_DATASET


def test_dataset_size():
    """Dataset should have 100 examples."""
    assert len(SENTIMENT_DATASET) == 100


def test_dataset_balance():
    """Dataset should be balanced (50 positive, 50 negative)."""
    labels = [label for _, label in SENTIMENT_DATASET]
    assert sum(labels) == 50  # 50 positive examples
    assert len(labels) - sum(labels) == 50  # 50 negative examples


def test_get_dataset():
    """get_dataset should return texts and labels."""
    texts, labels = get_dataset()
    assert len(texts) == 100
    assert len(labels) == 100
    assert all(isinstance(t, str) for t in texts)
    assert all(isinstance(l, int) for l in labels)
    assert all(l in [0, 1] for l in labels)


def test_get_few_shot_examples():
    """get_few_shot_examples should return balanced examples."""
    examples = get_few_shot_examples(n=4)
    assert len(examples) == 4
    
    labels = [label for _, label in examples]
    # Should have roughly equal positive and negative
    assert 1 <= sum(labels) <= 3


def test_no_empty_texts():
    """All texts should be non-empty."""
    texts, _ = get_dataset()
    assert all(len(t.strip()) > 0 for t in texts)
