"""
Tests for the prompts module.
"""

import pytest
from src.prompts import (
    zero_shot_prompt,
    few_shot_prompt,
    chain_of_thought_prompt,
    structured_json_prompt,
    get_strategy,
    STRATEGIES
)


def test_all_strategies_registered():
    """All strategy functions should be in registry."""
    expected = ["zero_shot", "few_shot", "chain_of_thought", "structured_json"]
    assert set(STRATEGIES.keys()) == set(expected)


def test_get_strategy():
    """get_strategy should return correct function."""
    assert get_strategy("zero_shot") == zero_shot_prompt
    assert get_strategy("few_shot") == few_shot_prompt


def test_get_strategy_invalid():
    """get_strategy should raise error for invalid name."""
    with pytest.raises(ValueError):
        get_strategy("invalid_strategy")


def test_zero_shot_prompt():
    """zero_shot_prompt should return system and user prompts."""
    system, user = zero_shot_prompt("I love this")
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert "I love this" in user
    assert len(system) > 0


def test_few_shot_prompt():
    """few_shot_prompt should include examples."""
    examples = [("Great movie", 1), ("Terrible book", 0)]
    system, user = few_shot_prompt("I love this", examples)
    assert "Great movie" in user
    assert "Terrible book" in user
    assert "I love this" in user


def test_chain_of_thought_prompt():
    """chain_of_thought_prompt should ask for reasoning."""
    system, user = chain_of_thought_prompt("I love this")
    assert "step" in user.lower()
    assert "reasoning" in user.lower() or "think" in user.lower()


def test_structured_json_prompt():
    """structured_json_prompt should request JSON format."""
    system, user = structured_json_prompt("I love this")
    assert "json" in user.lower() or "JSON" in user
    assert "{" in user and "}" in user
