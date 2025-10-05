import json
import matplotlib.pyplot as plt
import numpy as np

# Load Bedrock results
with open('results/benchmark_results_bedrock.json', 'r') as f:
    bedrock_data = json.load(f)

# Load Qwen results
with open('results/qwen_benchmark_results.json', 'r') as f:
    qwen_data = json.load(f)

# Prepare data
models = []
accuracies = []
parse_errors = []

# Add Bedrock models
for result in bedrock_data:
    model_name = result['model_id'].replace('us.', '').replace('anthropic.', '').replace('-v1:0', '').replace('-v0:2', '')
    models.append(model_name)
    accuracies.append(result['accuracy'] * 100)
    parse_errors.append(result['parse_error_rate'] * 100)

# Add Qwen
models.append('Qwen 2.5 3B')
accuracies.append(qwen_data['accuracy'] * 100)
parse_errors.append(qwen_data['parse_error_rate'] * 100)

# Sort by accuracy
sorted_indices = np.argsort(accuracies)[::-1]
models = [models[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]
parse_errors = [parse_errors[i] for i in sorted_indices]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Accuracy
colors = ['#2E86AB' if 'Qwen' not in m else '#A23B72' for m in models]
bars1 = ax1.barh(models, accuracies, color=colors)
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy on Counting Task', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 100)

# Add value labels
for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
    ax1.text(acc + 2, i, f'{acc:.1f}%', va='center', fontsize=10)

# Add grid
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Plot 2: Parse Errors
bars2 = ax2.barh(models, parse_errors, color=colors)
ax2.set_xlabel('Parse Error Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Models Failing to Output (N) Format', fontsize=14, fontweight='bold')
ax2.set_xlim(0, max(parse_errors) * 1.5 if max(parse_errors) > 0 else 5)

# Add value labels
for i, (bar, err) in enumerate(zip(bars2, parse_errors)):
    ax2.text(err + 0.05, i, f'{err:.1f}%', va='center', fontsize=10)

# Add grid
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='AWS Bedrock'),
    Patch(facecolor='#A23B72', label='HuggingFace (Colab)')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to results/benchmark_comparison.png")
