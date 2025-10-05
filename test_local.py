#!/usr/bin/env python3
"""Test mediation_utils locally to debug the issue."""

import sys
sys.path.append('scripts')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mediation_utils import generate_minimal_pairs, patch_and_generate, extract_answer
import json

# Load a small test dataset
print("Loading dataset...")
dataset = []
with open('data/counting_dataset.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        dataset.append(json.loads(line))

print(f"Loaded {len(dataset)} examples")

# Generate a few minimal pairs
print("\nGenerating 3 minimal pairs...")
pairs = generate_minimal_pairs(dataset, n_pairs=3)

print("\n" + "="*80)
print("EXAMINING MINIMAL PAIRS")
print("="*80)

for i, (pair_low, pair_high) in enumerate(pairs):
    print(f"\n--- Pair {i+1} ---")
    print(f"Category: {pair_low['category']}")
    print(f"Low count: {pair_low['answer']}")
    print(f"  List: {' '.join(pair_low['word_list'])}")
    print(f"High count: {pair_high['answer']}")
    print(f"  List: {' '.join(pair_high['word_list'])}")

    # Check if lists differ by only first word
    low_list = pair_low['word_list']
    high_list = pair_high['word_list']
    if len(low_list) != len(high_list):
        print(f"  WARNING: Lists have different lengths! {len(low_list)} vs {len(high_list)}")
    else:
        diffs = [j for j in range(len(low_list)) if low_list[j] != high_list[j]]
        if len(diffs) == 1 and diffs[0] == 0:
            print(f"  ✓ Lists differ only at position 0: '{low_list[0]}' vs '{high_list[0]}'")
        else:
            print(f"  WARNING: Lists differ at positions: {diffs}")

print("\n" + "="*80)
print("TESTING WITH A TINY MODEL")
print("="*80)

# Load smallest possible model for testing
print("\nLoading Qwen/Qwen2.5-0.5B-Instruct...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()

print(f"Model has {len(model.model.layers)} layers")

# Test on one pair, one layer
pair_low, pair_high = pairs[0]
test_layer = len(model.model.layers) // 2  # Middle layer

print(f"\nTesting layer {test_layer} on pair 1...")
print(f"  Low count: {pair_low['answer']}, High count: {pair_high['answer']}")

# Extract activation from high-count prompt
inputs_high = tokenizer(pair_high['prompt'], return_tensors="pt").to('cpu')
print(f"  High prompt tokenized to {inputs_high.input_ids.shape[1]} tokens")

with torch.no_grad():
    outputs_high = model(
        input_ids=inputs_high.input_ids,
        output_hidden_states=True
    )

activation_high_full = outputs_high.hidden_states[test_layer].cpu()
print(f"  Extracted activation shape: {activation_high_full.shape}")

# Get position for low prompt
inputs_low = tokenizer(pair_low['prompt'], return_tensors="pt")
print(f"  Low prompt tokenized to {inputs_low.input_ids.shape[1]} tokens")

seq_len = inputs_low.input_ids.shape[1]
last_pos = seq_len - 1
max_pos = activation_high_full.shape[1] - 1
patch_pos = min(last_pos, max_pos)

print(f"  Patching at position {patch_pos}")

# Extract activation at position
activation_at_pos = activation_high_full[:, patch_pos, :].clone()
print(f"  Position activation shape: {activation_at_pos.shape}")

# Generate without patching
print("\n  Generating WITHOUT patching...")
with torch.no_grad():
    outputs_original = model.generate(
        inputs_low.input_ids.to('cpu'),
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
original_text = tokenizer.decode(outputs_original[0][inputs_low.input_ids.shape[1]:], skip_special_tokens=True)
original_num = extract_answer(original_text)
print(f"    Output: '{original_text}'")
print(f"    Extracted: {original_num}")

# Generate WITH patching
print("\n  Generating WITH patching...")
patched_text = patch_and_generate(
    model, tokenizer, pair_low['prompt'],
    activation_at_pos, test_layer, patch_pos, 'cpu'
)
patched_num = extract_answer(patched_text)
print(f"    Output: '{patched_text}'")
print(f"    Extracted: {patched_num}")

# Check result
print(f"\n  RESULT:")
print(f"    Expected change: {pair_low['answer']} → {pair_high['answer']}")
print(f"    Actual output: {original_num} → {patched_num}")
if patched_num == pair_high['answer']:
    print(f"    ✓ SUCCESS: Patched output matches high count!")
elif original_num != patched_num:
    print(f"    ~ PARTIAL: Output changed but not to target")
else:
    print(f"    ✗ FAIL: Output didn't change")
