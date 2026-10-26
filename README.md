# Prompt Evaluation Lab

A reproducible research project for evaluating LLM prompting strategies on sentiment classification.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]() [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Question

**How do different prompting strategies affect the accuracy, consistency, and cost of LLM-based sentiment classification?**

This project compares four prompting approaches:
- **Zero-shot**: Direct classification without examples
- **Few-shot**: Classification with 3 labeled examples  
- **Chain-of-thought**: Explicit reasoning before answering
- **Structured JSON**: Requesting formatted output

## Method

### Dataset
- 100 hand-crafted sentences (50 positive, 50 negative)
- Public domain, created for educational purposes
- Balanced classes, simple binary sentiment

### Evaluation Protocol
1. Each strategy evaluated on all 100 examples
2. Three random seeds (42, 43, 44) for consistency measurement
3. Fixed temperature (0.0) for maximum determinism
4. All responses cached by `(model, prompt, seed)` for reproducibility

### Metrics
- **Accuracy**: Fraction of correct predictions
- **Consistency**: Fraction of examples with identical predictions across seeds
- **Parse rate**: Fraction of successfully parsed responses
- **Latency**: Mean response time (milliseconds)
- **Token usage**: Total prompt + completion tokens

## Results

Based on evaluation with simulated `gpt-3.5-turbo` responses (1,200 cached examples in `results/cache/`):

| Strategy | Accuracy | Consistency | Parse Rate |
|----------|----------|-------------|------------|
| Zero Shot | 0.870 | 0.570 | 1.000 |
| Few Shot | 0.880 | 0.680 | 1.000 |
| Chain Of Thought | 0.810 | 0.520 | 1.000 |
| Structured Json | 0.800 | 0.500 | 1.000 |

**Key findings:**
1. Few-shot prompting achieved highest accuracy (88%)
2. All strategies successfully parsed responses (100% parse rate)
3. Consistency scores show meaningful variation across strategies (52-68%)
4. Zero-shot provides good accuracy-simplicity trade-off (87% accuracy)

**Note**: These results use mock cached responses for demonstration. Real API results may vary.

## Threats to Validity

### Internal
- **Small dataset**: 100 examples may not capture full complexity
- **Hand-crafted data**: Real text contains more ambiguity and edge cases
- **Parsing assumptions**: May miss valid but differently-formatted responses

### External  
- **Simple task**: Binary sentiment is relatively easy; results may not generalize to complex reasoning
- **Single domain**: Sentiment classification only; other NLP tasks may show different patterns
- **Single model**: Tested only with gpt-3.5-turbo; other models may behave differently

### Construct
- **Limited seeds**: Three seeds may not fully characterize consistency
- **Temperature fixed at 0**: Higher temperatures would show different consistency patterns
- **Offline simulation**: Mock data approximates but doesn't replace real API responses

## Next Steps

1. **Scale up**: Test on larger datasets (SST-2, IMDB reviews, 1000+ examples)
2. **More strategies**: Self-consistency, role prompting, retrieval-augmented generation
3. **Model comparison**: GPT-4, Claude, Llama, open-source models
4. **Error analysis**: Characterize failure modes and edge cases
5. **Parameter sweep**: Temperature, top-p, max tokens effects
6. **Real-world deployment**: Latency under load, cost optimization, error handling

## Project Structure

```
prompt-eval-lab/
├── src/                      # Source code
│   ├── dataset.py           # Hand-crafted sentiment dataset
│   ├── prompts.py           # Prompting strategies
│   ├── llm_client.py        # API client with caching
│   ├── parser.py            # Response parsing
│   ├── metrics.py           # Evaluation metrics
│   └── evaluator.py         # Main evaluation pipeline
├── tests/                   # Test suite (36 tests)
├── results/cache/           # Cached API responses (committed for reproducibility)
├── evaluation.ipynb         # Main analysis notebook
├── requirements.txt         # Pinned dependencies
├── .env.example            # API key template
└── README.md               # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-eval-lab.git
cd prompt-eval-lab

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Offline Mode (No API Key Required)

The notebook runs using pre-cached responses:

```bash
jupyter notebook evaluation.ipynb
```

All cells will execute using cached data in `results/cache/`.

### Option 2: Live API Calls

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. In the notebook, set `OFFLINE_MODE = False`

4. Run the evaluation:
   ```bash
   python3 run_evaluation.py
   ```

### Running Tests

```bash
pytest tests/ -v
```

All 36 tests should pass.

## Reproducibility

This project prioritizes reproducibility:

✅ **Fixed seeds** for all random operations  
✅ **Response caching** keyed by (model, prompt, seed)  
✅ **Committed cache** allows offline reproduction  
✅ **Pinned dependencies** in requirements.txt  
✅ **Comprehensive tests** (36 tests, 100% passing)  
✅ **No external data dependencies** (dataset is self-contained)

To reproduce these exact results:
1. Use Python 3.8+
2. Install dependencies from requirements.txt  
3. Run notebook in offline mode with committed cache
4. Or re-run with same model + seeds (may vary slightly due to API changes)

## References

- Brown, T. et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.
- Kojima, T. et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS*.
- Wang, X. et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023*.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This is a learning project. Contributions welcome:
- Additional prompting strategies
- More evaluation metrics  
- Extended datasets
- Documentation improvements

## Disclaimer

This is an **educational project** for learning about LLM evaluation. It is:
- ✅ Open source and non-commercial
- ✅ Using synthetic/public-domain data only
- ✅ Designed for reproducible research
- ❌ Not affiliated with any company or institution
- ❌ Not intended for production use
- ❌ No real personal data or proprietary information

For production LLM applications, consider additional factors: safety, bias, robustness, monitoring, and compliance.
