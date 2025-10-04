# LLM Counting Analysis

Causal mediation analysis and mechanistic interpretability experiments on LLMs performing simple counting tasks.

## Overview

This project investigates how language models internally represent and track running counts. We use three complementary approaches:

1. **Benchmark**: Evaluate model accuracy on counting tasks
2. **Causal Mediation**: Identify which layers encode count information via activation patching
3. **Linear Probes**: Train classifiers to decode count from layer activations
4. **Intervention**: Use probes to steer model outputs by modifying activations

## Repository Structure

```
.
├── generate_dataset.py          # Generate counting task dataset
├── benchmark_bedrock.py          # Benchmark models on AWS Bedrock
├── causal_mediation_v2.py        # Full analysis pipeline (mediation + probes + intervention)
├── counting_dataset.jsonl        # Generated dataset (5000 examples)
├── results/                      # Experiment outputs
│   ├── benchmark_results_bedrock.json
│   ├── benchmark_results_bedrock.png
│   └── mediation_results_v2.json (pending)
└── README.md
```

## Experiments

### 1. Dataset Generation

Create a balanced dataset of counting tasks:
- 10 categories (fruit, animal, vehicle, color, tool, furniture, clothing, food, sport, instrument)
- Lists of 5-10 words
- 0-5 matching items per list
- Clear prompt format with expected answer in parentheses: `(N)`

```bash
python generate_dataset.py
```

### 2. Model Benchmark

Benchmark models on AWS Bedrock (1000 examples):

**Results:**
- **Llama 3.1 70B**: 78.2% accuracy (best)
- **Claude Haiku 3.5**: 57.8%
- **Mistral 7B**: 49.6%
- **Llama 3.1 8B**: 34.9%

Parse error rates were near 0% with lenient parsing (accepts `(N)` format or first number in output).

```bash
python benchmark_bedrock.py --max_examples 1000
```

### 3. Causal Mediation Analysis

Use activation patching to identify which layers encode running count:
- Create minimal pairs: identical lists except one word (count differs by 1)
- Patch activations from count=N+1 into count=N prompt
- Measure which layers cause output to shift toward N+1

```bash
python causal_mediation_v2.py --mediation_examples 200 --skip_probe --skip_intervention
```

### 4. Linear Probes

Train linear classifiers to decode count (0-5) from layer activations:
- Collect activations from 5000 examples
- Train logistic regression on each layer
- Identify which layers have linearly separable count representations

```bash
python causal_mediation_v2.py --probe_examples 5000 --skip_mediation --skip_intervention
```

### 5. Intervention Experiment

Use trained probes to steer model outputs:
- Find best probe layer
- Compute steering vectors for each count (0-5)
- Intervene by adding steering difference to activations
- Measure if output changes as expected

```bash
python causal_mediation_v2.py --intervention_examples 100
```

## Requirements

```
torch
transformers
accelerate
boto3
numpy
matplotlib
seaborn
scikit-learn
tqdm
```

Install with:
```bash
pip install torch transformers accelerate boto3 numpy matplotlib seaborn scikit-learn tqdm
```

## AWS Bedrock Setup

For benchmarking, you need AWS credentials with access to Bedrock models:

1. Configure AWS CLI: `aws configure`
2. Request model access in AWS Console (Bedrock → Model access)
3. Use inference profiles with `us.` prefix for cross-region access

## Key Findings

### Benchmark
- Larger models (70B) significantly outperform smaller ones (8B) on this simple task
- Even small models can learn the output format with clear prompts
- Lenient parsing dramatically improves measured accuracy

### Analysis (Pending)
- Mediation analysis running on EC2
- Expected to identify middle-to-late layers as encoding count
- Linear probes will validate if count is linearly represented

## Citation

If you use this code, please cite:

```
@misc{llm-counting-analysis,
  author = {Daniel},
  title = {LLM Counting Analysis},
  year = {2025},
  url = {https://github.com/yourusername/llm-counting-analysis}
}
```
