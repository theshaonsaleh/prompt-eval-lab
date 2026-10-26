"""
Prompt strategies for sentiment classification evaluation.
Each strategy returns a system prompt and a user prompt template.
"""

def zero_shot_prompt(text):
    """Zero-shot prompting: direct classification without examples."""
    system = "You are a sentiment classifier. Classify the sentiment of text as positive or negative."
    user = f"""Classify the sentiment of the following text as either "positive" or "negative".

Text: {text}

Answer with only one word: positive or negative."""
    return system, user


def few_shot_prompt(text, examples):
    """Few-shot prompting: provide examples before classification."""
    system = "You are a sentiment classifier. Classify the sentiment of text as positive or negative."
    
    example_text = ""
    for ex_text, ex_label in examples:
        sentiment = "positive" if ex_label == 1 else "negative"
        example_text += f'\nText: {ex_text}\nSentiment: {sentiment}\n'
    
    user = f"""Classify the sentiment of text as either "positive" or "negative".

Here are some examples:{example_text}

Now classify this text:
Text: {text}

Answer with only one word: positive or negative."""
    return system, user


def chain_of_thought_prompt(text):
    """Chain-of-thought: ask model to explain reasoning before classifying."""
    system = "You are a sentiment classifier. Think step by step and classify the sentiment of text."
    user = f"""Classify the sentiment of the following text as either "positive" or "negative".

Text: {text}

Think step by step:
1. What are the key sentiment words?
2. What is the overall tone?
3. What is the final sentiment?

After your reasoning, provide your final answer on a new line as: "Final answer: positive" or "Final answer: negative"."""
    return system, user


def structured_json_prompt(text):
    """Structured output: request JSON format for classification."""
    system = "You are a sentiment classifier. Respond with valid JSON only."
    user = f"""Classify the sentiment of the following text.

Text: {text}

Respond with a JSON object in this exact format:
{{"sentiment": "positive"}} or {{"sentiment": "negative"}}

JSON response:"""
    return system, user


# Registry of all strategies
STRATEGIES = {
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "chain_of_thought": chain_of_thought_prompt,
    "structured_json": structured_json_prompt,
}

def get_strategy(name):
    """Get a prompting strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]
