"""
Tests for the parser module.
"""

import pytest
from src.parser import (
    parse_response,
    _parse_simple_response,
    _parse_cot_response,
    _parse_json_response
)


def test_parse_simple_exact():
    """Parser should handle exact matches."""
    assert _parse_simple_response("positive") == "positive"
    assert _parse_simple_response("negative") == "negative"


def test_parse_simple_case_insensitive():
    """Parser should be case insensitive."""
    assert _parse_simple_response("Positive") == "positive"
    assert _parse_simple_response("NEGATIVE") == "negative"


def test_parse_simple_first_word():
    """Parser should extract first word."""
    assert _parse_simple_response("positive sentiment") == "positive"
    assert _parse_simple_response("negative overall") == "negative"


def test_parse_simple_substring():
    """Parser should handle substring matches."""
    assert _parse_simple_response("the sentiment is positive") == "positive"
    assert _parse_simple_response("this is negative") == "negative"


def test_parse_simple_unknown():
    """Parser should return unknown for ambiguous responses."""
    assert _parse_simple_response("maybe") == "unknown"
    assert _parse_simple_response("both positive and negative") == "unknown"


def test_parse_cot_final_answer():
    """Chain-of-thought parser should extract final answer."""
    response = """
    1. The words "love" and "great" are positive
    2. The tone is enthusiastic
    3. Final answer: positive
    """
    assert _parse_cot_response(response) == "positive"


def test_parse_json_valid():
    """JSON parser should extract from valid JSON."""
    assert _parse_json_response('{"sentiment": "positive"}') == "positive"
    assert _parse_json_response('{"sentiment": "negative"}') == "negative"


def test_parse_json_with_wrapper():
    """JSON parser should handle JSON wrapped in text."""
    response = 'Here is the result: {"sentiment": "positive"}'
    assert _parse_json_response(response) == "positive"


def test_parse_json_invalid_fallback():
    """JSON parser should fall back to simple parsing."""
    assert _parse_json_response("positive") == "positive"
    assert _parse_json_response("negative") == "negative"


def test_parse_response_routing():
    """parse_response should route to correct parser."""
    assert parse_response("positive", "zero_shot") == "positive"
    assert parse_response('{"sentiment": "positive"}', "structured_json") == "positive"
    assert parse_response("Final answer: negative", "chain_of_thought") == "negative"
