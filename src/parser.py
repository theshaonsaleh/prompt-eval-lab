"""
Response parsing utilities to extract predictions from LLM outputs.
"""

import re
import json


def parse_response(response_text: str, strategy: str) -> str:
    """
    Parse LLM response to extract sentiment prediction.
    
    Args:
        response_text: Raw text response from LLM
        strategy: Name of the prompting strategy used
        
    Returns:
        "positive", "negative", or "unknown" if parsing failed
    """
    if strategy == "structured_json":
        return _parse_json_response(response_text)
    elif strategy == "chain_of_thought":
        return _parse_cot_response(response_text)
    else:
        return _parse_simple_response(response_text)


def _parse_simple_response(response_text: str) -> str:
    """Parse simple positive/negative response."""
    response_lower = response_text.lower().strip()
    
    # Try exact match first
    if response_lower == "positive":
        return "positive"
    if response_lower == "negative":
        return "negative"
    
    # Try first word
    first_word = response_lower.split()[0] if response_lower else ""
    if first_word in ["positive", "negative"]:
        return first_word
    
    # Try substring search
    if "positive" in response_lower and "negative" not in response_lower:
        return "positive"
    if "negative" in response_lower and "positive" not in response_lower:
        return "negative"
    
    return "unknown"


def _parse_cot_response(response_text: str) -> str:
    """Parse chain-of-thought response looking for final answer."""
    response_lower = response_text.lower()
    
    # Look for "final answer:" pattern
    match = re.search(r'final answer:\s*(positive|negative)', response_lower)
    if match:
        return match.group(1)
    
    # Fall back to simple parsing
    return _parse_simple_response(response_text)


def _parse_json_response(response_text: str) -> str:
    """Parse JSON structured response."""
    try:
        # Try to extract JSON from response
        # Sometimes model includes extra text before/after JSON
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            sentiment = data.get("sentiment", "").lower()
            if sentiment in ["positive", "negative"]:
                return sentiment
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fall back to simple parsing if JSON fails
    return _parse_simple_response(response_text)
