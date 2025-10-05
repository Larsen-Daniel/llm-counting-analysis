# LLM Counting Analysis

Mechanistic interpretability experiments investigating how language models internally represent and track counts during simple counting tasks.

## Experiments Performed

### 1. Model Benchmarking on Counting Task

**Objective**: Evaluate how accurately different foundation models can count matching items in short lists.

**Dataset**: 1000 examples from `counting_dataset.jsonl`
- Lists of 5-10 words randomly selected from 10 categories (fruit, animal, vehicle, color, tool, furniture, clothing, food, sport, instrument)
- 0-5 matching items per list (balanced distribution)
- Clear prompt format requesting answer in parentheses: `(N)`

**Models tested**:
- AWS Bedrock: Llama 3.1 70B Instruct, Llama 3.1 8B Instruct, Claude 3.5 Haiku, Mistral 7B Instruct
- HuggingFace (via Google Colab): Qwen 2.5 3B Instruct

**Results**:
| Model | Accuracy | Parse Errors | Numerical Errors |
|-------|----------|--------------|------------------|
| Llama 3.1 70B | **78.2%** | 0.0% | 21.8% |
| Claude 3.5 Haiku | 57.8% | 0.0% | 42.2% |
| Mistral 7B | 49.6% | 0.3% | 50.1% |
| Llama 3.1 8B | 34.9% | 0.0% | 65.1% |

**Key findings**:
- All models successfully learned the output format (`(N)`) with minimal parse errors
- Accuracy scales strongly with model size (70B vs 8B shows 2.2x improvement)
- Even on this simple task, smaller models struggle significantly
- Lenient parsing (accepting any number in parentheses or first number) was critical for fair evaluation

**Outputs**: `results/benchmark_results_bedrock.json`, `results/benchmark_results_bedrock.png`

---

### 2. Causal Mediation Analysis via Activation Patching

**Objective**: Identify which transformer layers encode the running count by measuring causal effects of activation interventions.

**Method**:
- Generated 200 minimal pairs: identical word lists except one word changes category membership, causing count to differ by exactly 1
- For each pair, extracted activations from the "count=N+1" prompt at each layer
- Patched these activations into the "count=N" prompt at the same position
- Measured how often the patched prompt's output changed from N to N+1

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

## Repository Contents

**Core scripts**:
- `generate_dataset.py` - Generate balanced counting task dataset
- `benchmark_bedrock.py` - Benchmark models on AWS Bedrock
- `qwen_benchmark_colab.ipynb` - **Google Colab notebook to benchmark Qwen 2.5 3B** (recommended for easy GPU access)
- `causal_mediation_v2.py` - Run mediation analysis, probe training, and intervention experiments
- `plot_mediation.py` - Generate visualization of mediation results

**Data**:
- `counting_dataset.jsonl` - 5000 examples (5-10 words, 0-5 matches, balanced)

**Results** (in `results/` folder):
- `benchmark_results_bedrock.json` - Full benchmark data for 4 models
- `benchmark_results_bedrock.png` - Bar charts comparing accuracy and parse errors
- `mediation_results_v2.json` - Layer-wise causal effects from activation patching
- `mediation_results_v2.png` - Visualization showing layers 21-25 with highest effects

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
python generate_dataset.py
```

### Benchmark Models on AWS Bedrock
```bash
python benchmark_bedrock.py --max_examples 1000
```

### Benchmark Qwen on Google Colab (Recommended)
1. Open `qwen_benchmark_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU → T4 GPU
3. Run all cells
4. Download results file when complete

### Run Mediation Analysis
```bash
python causal_mediation_v2.py --mediation_examples 200 --skip_probe --skip_intervention
```

### Train Linear Probes
```bash
python causal_mediation_v2.py --probe_examples 5000 --skip_mediation --skip_intervention
```

### Run Intervention Experiment
```bash
python causal_mediation_v2.py --intervention_examples 100
```

### Visualize Results
```bash
python plot_mediation.py
```

</details>

---

## Citation

```bibtex
@misc{llm-counting-analysis,
  author = {Daniel Larsen},
  title = {LLM Counting Analysis: Mechanistic Interpretability of Simple Counting Tasks},
  year = {2025},
  url = {https://github.com/Larsen-Daniel/llm-counting-analysis}
}
```
