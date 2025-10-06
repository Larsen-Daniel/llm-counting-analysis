#!/usr/bin/env python3
"""Compare dataset examples vs synthetic pairs performance."""

import sys
sys.path.append('scripts')

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mediation_utils import generate_simple_pairs, extract_answer
import random

# Load dataset
print("Loading dataset...")
dataset = []
with open('data/counting_dataset.jsonl', 'r') as f:
    for line in f:
        dataset.append(json.loads(line))
print(f"Loaded {len(dataset)} examples\n")

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True
)

if device == "cpu":
    model = model.to(device)

model.eval()
print("Model loaded!\n")

# Sample 50 examples from the dataset
random.seed(42)
dataset_sample = random.sample(dataset, 50)

# Generate 50 of our synthetic pairs
print("Generating synthetic pairs...")
test_pairs = generate_simple_pairs(dataset, n_pairs=100)[:50]
print()

# Test model on dataset examples
print("="*80)
print("TESTING ON DATASET EXAMPLES")
print("="*80)

dataset_correct = 0
dataset_results = []

for i, example in enumerate(dataset_sample):
    inputs = tokenizer(example['prompt'], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    predicted = extract_answer(generated)
    correct = (predicted == example['answer'])
    dataset_correct += correct
    dataset_results.append({
        'predicted': predicted,
        'true': example['answer'],
        'correct': correct,
        'list': ' '.join(example['word_list'])
    })

    if i < 3:  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"  Category: {example['category']}")
        print(f"  List: {' '.join(example['word_list'])}")
        print(f"  Output: '{generated}'")
        print(f"  Predicted: {predicted}, True: {example['answer']}, Correct: {correct}")

dataset_accuracy = dataset_correct / len(dataset_sample)

# Test model on our synthetic pairs (test the low-count version)
print("\n" + "="*80)
print("TESTING ON SYNTHETIC PAIRS (LOW-COUNT VERSION)")
print("="*80)

synthetic_correct = 0
synthetic_results = []

for i, (pair_low, pair_high) in enumerate(test_pairs):
    inputs = tokenizer(pair_low['prompt'], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    predicted = extract_answer(generated)
    correct = (predicted == pair_low['answer'])
    synthetic_correct += correct
    synthetic_results.append({
        'predicted': predicted,
        'true': pair_low['answer'],
        'correct': correct,
        'list': ' '.join(pair_low['word_list'])
    })

    if i < 3:  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"  Category: {pair_low['category']}")
        print(f"  Low list:  {' '.join(pair_low['word_list'])}")
        print(f"  High list: {' '.join(pair_high['word_list'])}")
        print(f"  Output: '{generated}'")
        print(f"  Predicted: {predicted}, True: {pair_low['answer']}, Correct: {correct}")

synthetic_accuracy = synthetic_correct / len(test_pairs)

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Dataset examples accuracy:  {dataset_accuracy:.1%} ({dataset_correct}/{len(dataset_sample)})")
print(f"Synthetic pairs accuracy:   {synthetic_accuracy:.1%} ({synthetic_correct}/{len(test_pairs)})")
print(f"Difference: {(dataset_accuracy - synthetic_accuracy)*100:.1f} percentage points")

if dataset_accuracy > synthetic_accuracy + 0.05:
    print("\n⚠️  HYPOTHESIS CONFIRMED: Model performs significantly better on dataset examples!")
elif synthetic_accuracy > dataset_accuracy + 0.05:
    print("\n✓ HYPOTHESIS REJECTED: Synthetic pairs are not harder")
else:
    print("\n~ Results are similar (within 5 percentage points)")

# Analyze prompt differences
print("\n" + "="*80)
print("PROMPT COMPARISON")
print("="*80)
print("\n--- Dataset example prompt ---")
print(dataset_sample[0]['prompt'])
print("\n--- Synthetic pair prompt ---")
print(test_pairs[0][0]['prompt'])
print("\n--- High-count pair prompt (from dataset) ---")
print(test_pairs[0][1]['prompt'])

# Check if prompts are identical in format
dataset_prompt_lines = dataset_sample[0]['prompt'].split('\n')
synthetic_prompt_lines = test_pairs[0][0]['prompt'].split('\n')

print("\n" + "="*80)
print("LINE-BY-LINE PROMPT COMPARISON")
print("="*80)
for i, (d_line, s_line) in enumerate(zip(dataset_prompt_lines, synthetic_prompt_lines)):
    if d_line == s_line:
        print(f"Line {i}: ✓ MATCH")
    else:
        print(f"Line {i}: ✗ DIFFERENT")
        print(f"  Dataset:   '{d_line}'")
        print(f"  Synthetic: '{s_line}'")
