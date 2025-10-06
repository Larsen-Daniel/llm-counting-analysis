#!/usr/bin/env python3
"""Plot mediation results for Qwen 3B."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
with open('results/mediation_results_qwen3b-final.json', 'r') as f:
    mediation = json.loads(f.read())

with open('results/probe_results_v2.json', 'r') as f:
    probe_data = json.loads(f.read())

# Extract mediation data
layers = []
exact_match = []
changed_rate = []
mean_effect = []

for layer_str, data in mediation['layer_effects'].items():
    layer = int(layer_str)
    if layer <= 27:  # Only plot up to layer 27 to match probe data
        layers.append(layer)
        exact_match.append(data['exact_match_rate'] * 100)
        changed_rate.append(data['changed_rate'] * 100)
        mean_effect.append(data['mean_effect'] * 100)

# Extract probe data (only up to layer 27)
probe_layers = []
probe_acc = []
for layer_str, data in probe_data['probe']['probe_accuracy'].items():
    layer = int(layer_str)
    probe_layers.append(layer)
    probe_acc.append(data['test_accuracy'] * 100)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot mediation metrics
ax.plot(layers, exact_match, 'o-', label='Activation Patching: Exact Match Rate',
        linewidth=2, markersize=6, color='#2E86AB')
ax.plot(layers, changed_rate, 's--', label='Activation Patching: Changed Rate',
        linewidth=1.5, markersize=4, alpha=0.6, color='#A23B72')

# Plot probe accuracy
ax.plot(probe_layers, probe_acc, '^-', label='Linear Probe: Test Accuracy',
        linewidth=2, markersize=6, color='#F18F01')

# Styling
ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Qwen 2.5 3B: Activation Patching vs Linear Probing\n100 Perfect Pairs (Mediation), 5000 Examples (Probing)',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 100])
ax.set_xlim([-1, 28])

# Add annotations for key layers
peak_mediation_idx = np.argmax(exact_match)
peak_mediation_layer = layers[peak_mediation_idx]
peak_mediation_val = exact_match[peak_mediation_idx]
ax.annotate(f'Peak Mediation\nLayer {peak_mediation_layer}: {peak_mediation_val:.1f}%',
            xy=(peak_mediation_layer, peak_mediation_val),
            xytext=(peak_mediation_layer - 5, peak_mediation_val + 10),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86AB', alpha=0.2),
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

peak_probe_idx = np.argmax(probe_acc)
peak_probe_layer = probe_layers[peak_probe_idx]
peak_probe_val = probe_acc[peak_probe_idx]
ax.annotate(f'Peak Probe\nLayer {peak_probe_layer}: {peak_probe_val:.1f}%',
            xy=(peak_probe_layer, peak_probe_val),
            xytext=(peak_probe_layer + 3, peak_probe_val - 15),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F18F01', alpha=0.2),
            arrowprops=dict(arrowstyle='->', color='#F18F01', lw=1.5))

plt.tight_layout()
plt.savefig('results/mediation_vs_probing_qwen3b.png', dpi=300, bbox_inches='tight')
print("Saved to results/mediation_vs_probing_qwen3b.png")

# Print statistics
print("\n" + "="*80)
print("MEDIATION VS PROBING COMPARISON")
print("="*80)
print(f"\nTop 5 Layers by Exact Match Rate (Mediation):")
sorted_mediation = sorted(zip(layers, exact_match), key=lambda x: x[1], reverse=True)
for layer, rate in sorted_mediation[:5]:
    print(f"  Layer {layer:2d}: {rate:5.1f}%")

print(f"\nTop 5 Layers by Probe Accuracy:")
sorted_probe = sorted(zip(probe_layers, probe_acc), key=lambda x: x[1], reverse=True)
for layer, acc in sorted_probe[:5]:
    print(f"  Layer {layer:2d}: {acc:5.1f}%")

print(f"\nLayers 21-27 Comparison:")
print(f"  Mediation exact match: {np.mean([exact_match[i] for i, l in enumerate(layers) if 21 <= l <= 27]):.1f}% avg")
print(f"  Probe accuracy:        {np.mean([probe_acc[i] for i, l in enumerate(probe_layers) if 21 <= l <= 27]):.1f}% avg")
