# LLM Counting Analysis

Mechanistic interpretability experiments investigating how language models internally represent and track counts during simple counting tasks.

## Experiments Performed

### 1. Model Benchmarking on Counting Task

**Objective**: Evaluate how accurately different foundation models can count matching items in short lists.

**Dataset**: 1000 examples from `counting_dataset.jsonl`
- Lists of 5-10 words randomly selected from 10 categories (fruit, animal, vehicle, color, tool, furniture, clothing, food, sport, instrument)
- 0-5 matching items per list (balanced distribution)
- Clear prompt format requesting answer in parentheses: `(N)`

**Models tested**: We initially benchmarked using AWS Bedrock for API access. To perform mechanistic interpretability experiments requiring model weights, we needed to run models locally. Memory constraints on EC2 led us to downsize to Qwen 2.5 3B, which could run comfortably on Google Colab's T4 GPU.

- **AWS Bedrock**: Llama 3.1 70B Instruct, Llama 3.1 8B Instruct, Claude 3.5 Haiku, Mistral 7B Instruct
- **Google Colab**: Qwen 2.5 3B Instruct

**Results**:

![Benchmark Results](results/benchmark_comparison.png)

| Model | Accuracy | Parse Errors | Numerical Errors |
|-------|----------|--------------|------------------|
| Llama 3.1 70B | **78.2%** | 0.0% | 21.8% |
| Claude 3.5 Haiku | 57.8% | 0.0% | 42.2% |
| Mistral 7B | 49.6% | 0.3% | 50.1% |
| Qwen 2.5 3B | 35.2% | 0.0% | 64.8% |
| Llama 3.1 8B | 34.9% | 0.0% | 65.1% |

**Key findings**:
- All models successfully learned the output format (`(N)`) with minimal parse errors
- Accuracy scales strongly with model size (70B vs 8B shows 2.2x improvement)
- Qwen 2.5 3B performs comparably to Llama 3.1 8B despite being smaller
- Even on this simple task, smaller models struggle significantly
- Lenient parsing (accepting any number in parentheses or first number) was critical for fair evaluation

---

### 2. Causal Mediation Analysis via Activation Patching

**Objective**: Identify which transformer layers encode the running count by measuring causal effects of activation interventions.

**Method**:
- Generated 200 minimal pairs: identical word lists except the first word changes category membership, causing count to differ by exactly 1
- For each pair, extracted activations from the "count=N+1" prompt at each layer
- Patched these activations into the "count=N" prompt at the same position
- Measured how often the patched prompt's output changed from N to N+1

**Example minimal pair**:
```
Low count (4 tools):
Type: tool
List: tuba tennis ruler clamp motorcycle indigo tape document hammer

High count (5 tools):
Type: tool
List: saw tennis ruler clamp motorcycle indigo tape document hammer

First word changed: 'tuba' → 'saw'
Count changes from 4 to 5
```

**Model**: Qwen 2.5 1.5B Instruct (open-source model via HuggingFace)

**Results**:
- **Layers 21-25 showed strongest causal effects** (~0.44-0.45 mean effect)
  - Layer 21: 0.445 ± 0.124
  - Layer 22: 0.450 ± 0.127
  - Layer 23: 0.450 ± 0.120
  - Layer 24: 0.445 ± 0.131
  - Layer 25: 0.440 ± 0.128
- Early layers (0-10) showed minimal effects (<0.25)
- Effects peaked in mid-to-late layers, suggesting count computation happens progressively

**Key findings**:
- Count information is causally represented in specific mid-to-late layers
- Multiple adjacent layers show similar effects, suggesting distributed representation
- The sharp transition from low to high effects indicates a computational phase change

**Outputs**: `results/mediation_results_v2.json`, `results/mediation_results_v2.png`

---

### 3. Linear Probe Training (Pending)

**Objective**: Determine if count is linearly represented in layer activations by training classifiers to decode count (0-5) from each layer.

**Planned method**:
- Collect activations from all 28 layers on 5000 examples
- Train logistic regression classifiers for each layer
- Measure accuracy and compare to mediation results

**Expected outcome**: Layers with high causal effects (21-25) should also have high probe accuracy, validating that count is both causally important and linearly accessible.

---

### 4. Activation Intervention (Pending)

**Objective**: Use trained probes to steer model outputs by modifying activations with "steering vectors."

**Planned method**:
- Identify best probe layer from experiment #3
- Compute steering vectors: difference in mean activations between count=N and count=M examples
- Add steering vector to activations during forward pass
- Measure if output changes from N to M as expected

**Expected outcome**: If count is cleanly represented, adding the (M-N) steering vector should reliably change output from N to M.

---

## Repository Structure

```
├── data/
│   └── counting_dataset.jsonl     # 5000 examples for all experiments
│
├── results/                        # All final outputs
│   ├── benchmark_results_bedrock.json    # Bedrock benchmarking results
│   ├── qwen_benchmark_results.json       # Qwen 3B benchmarking results
│   ├── benchmark_comparison.png          # Combined benchmark visualization
│   ├── mediation_results_v2.json         # Causal mediation results
│   └── mediation_results_v2.png          # Mediation visualization
│
├── notebooks/
│   └── qwen_benchmark_colab.ipynb # Colab notebook (runs benchmarking + mediation)
│
└── scripts/
    ├── generate_dataset.py        # [AUXILIARY] Creates counting_dataset.jsonl
    ├── benchmark_bedrock.py       # [MAIN] Bedrock benchmarking
    ├── plot_benchmark.py          # [AUXILIARY] Creates benchmark_comparison.png
    ├── mediation_utils.py         # [AUXILIARY] Helper functions for mediation
    └── causal_mediation_v2.py     # [NOT USED - see notebook instead]
```

### Scripts Used for Final Results

**Experiment 1: Model Benchmarking**
- `scripts/benchmark_bedrock.py` → `results/benchmark_results_bedrock.json`
- `notebooks/qwen_benchmark_colab.ipynb` (benchmark section) → `results/qwen_benchmark_results.json`
- `scripts/plot_benchmark.py` → `results/benchmark_comparison.png`

**Experiment 2: Causal Mediation Analysis**
- `notebooks/qwen_benchmark_colab.ipynb` (mediation section) → `results/mediation_results_v2.json`

**Auxiliary Scripts (data prep, utilities)**
- `scripts/generate_dataset.py` - Created the dataset
- `scripts/mediation_utils.py` - Helper functions imported by notebook
- `scripts/causal_mediation_v2.py` - Not used (mediation ran in Colab notebook instead)

---

## How to Run (Optional)

<details>
<summary>Click to expand instructions</summary>

### Requirements
```bash
pip install torch transformers accelerate boto3 numpy matplotlib seaborn scikit-learn tqdm
```

### AWS Bedrock Setup
1. Configure AWS CLI: `aws configure`
2. Request model access in AWS Console (Bedrock → Model access → Enable for us-east-1)

### Generate Dataset
```bash
python scripts/generate_dataset.py
```

### Benchmark Models on AWS Bedrock
```bash
python scripts/benchmark_bedrock.py --max_examples 1000
```

### Benchmark Qwen on Google Colab (Recommended)
1. Open `notebooks/qwen_benchmark_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU → T4 GPU
3. Run all cells
4. Download results file when complete

### Run Mediation Analysis
```bash
python scripts/causal_mediation_v2.py --mediation_examples 200 --skip_probe --skip_intervention
```

### Train Linear Probes
```bash
python scripts/causal_mediation_v2.py --probe_examples 5000 --skip_mediation --skip_intervention
```

### Run Intervention Experiment
```bash
python scripts/causal_mediation_v2.py --intervention_examples 100
```

### Visualize Results
```bash
python scripts/plot_benchmark.py
```

</details>

---

*Written with the help of Claude Sonnet 4.5*

## Citation

```bibtex
@misc{llm-counting-analysis,
  author = {Daniel Larsen},
  title = {LLM Counting Analysis: Mechanistic Interpretability of Simple Counting Tasks},
  year = {2025},
  url = {https://github.com/Larsen-Daniel/llm-counting-analysis}
}
```
