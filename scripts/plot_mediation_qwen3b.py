#!/usr/bin/env python3
"""Plot mediation results for Qwen 3B."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/mediation_results_qwen3b-final.json', 'r') as f:
    mediation = json.loads(f.read())

# Extract mediation data (Qwen 3B has 36 layers: 0-35)
layers = []
exact_match = []
changed_rate = []
mean_effect = []

for layer_str, data in mediation['layer_effects'].items():
    layer = int(layer_str)
    layers.append(layer)
    exact_match.append(data['exact_match_rate'] * 100)
    changed_rate.append(data['changed_rate'] * 100)
    mean_effect.append(data['mean_effect'] * 100)

# Sort by layer
sorted_data = sorted(zip(layers, exact_match, changed_rate, mean_effect))
layers, exact_match, changed_rate, mean_effect = zip(*sorted_data)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot mediation metrics
ax.plot(layers, exact_match, 'o-', label='Exact Match Rate',
        linewidth=2, markersize=6, color='#2E86AB')
ax.plot(layers, changed_rate, 's--', label='Changed Rate',
        linewidth=1.5, markersize=4, alpha=0.6, color='#A23B72')
ax.plot(layers, mean_effect, '^:', label='Mean Effect',
        linewidth=1.5, markersize=4, alpha=0.6, color='#6A994E')

# Styling
ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Qwen 2.5 3B (36 layers): Activation Patching Results\n100 Perfect Pairs',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 100])

# Add annotation for peak layer
peak_idx = np.argmax(exact_match)
peak_layer = layers[peak_idx]
peak_val = exact_match[peak_idx]
ax.annotate(f'Peak\nLayer {peak_layer}: {peak_val:.1f}%',
            xy=(peak_layer, peak_val),
            xytext=(peak_layer - 5, peak_val + 10),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86AB', alpha=0.2),
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

plt.tight_layout()
plt.savefig('results/mediation_qwen3b.png', dpi=300, bbox_inches='tight')
print("Saved to results/mediation_qwen3b.png")

# Print statistics
print("\n" + "="*80)
print("ACTIVATION PATCHING RESULTS (Qwen 2.5 3B)")
print("="*80)
print(f"\nTop 5 Layers by Exact Match Rate:")
sorted_mediation = sorted(zip(layers, exact_match, mean_effect, changed_rate),
                          key=lambda x: x[1], reverse=True)
for layer, em, me, cr in sorted_mediation[:5]:
    print(f"  Layer {layer:2d}: {em:5.1f}% exact match, {me:5.1f}% mean effect, {cr:5.1f}% changed rate")
