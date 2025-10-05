import torch
import numpy as np
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import re
from scipy import stats

def extract_answer(text: str) -> int | None:
    """Extract numerical answer from model output."""
    matches = re.findall(r'\((\d+)\)', text)
    if matches:
        return int(matches[-1])

    first_line = text.strip().split('\n')[0]
    number_match = re.search(r'\d+', first_line)
    if number_match:
        return int(number_match.group(0))

    return None

def generate_minimal_pairs(dataset: List[Dict], n_pairs: int = 200, list_length: int = 5) -> List[Tuple[Dict, Dict]]:
    """Generate minimal pairs by swapping the first word in the list.

    Args:
        dataset: Full dataset of examples
        n_pairs: Number of pairs to generate
        list_length: Length of word lists (default 5 for simpler examples)
    """

    # Category words to use for swapping
    CATEGORIES = {
        "fruit": ["apple", "banana", "cherry", "grape", "orange"],
        "animal": ["dog", "cat", "bird", "fish", "horse"],
        "vehicle": ["car", "bus", "truck", "bike", "train"],
        "color": ["red", "blue", "green", "yellow", "purple"],
        "tool": ["hammer", "wrench", "saw", "drill", "pliers"],
        "furniture": ["chair", "table", "sofa", "bed", "desk"],
        "clothing": ["shirt", "pants", "dress", "shoes", "hat"],
        "food": ["pizza", "burger", "pasta", "rice", "bread"],
        "sport": ["soccer", "tennis", "baseball", "hockey", "golf"],
        "instrument": ["guitar", "piano", "drums", "violin", "flute"]
    }

    NOISE_WORDS = ["bowl", "window", "door", "cloud", "mountain"]

    pairs = []

    # Filter to only examples with specified list length
    filtered_dataset = [ex for ex in dataset if len(ex['word_list']) == list_length]

    print(f"Filtered to {len(filtered_dataset)} examples with list_length={list_length}")
    if len(filtered_dataset) == 0:
        print(f"WARNING: No examples found with list_length={list_length}")
        print(f"Available lengths: {set(len(ex['word_list']) for ex in dataset[:100])}")
        return []

    for ex in filtered_dataset:
        if len(pairs) >= n_pairs:
            break

        category = ex['category']
        word_list = ex['word_list']
        first_word = word_list[0]

        # Check if first word matches the category
        if first_word in CATEGORIES.get(category, []):
            # First word matches - swap it with a noise word to decrease count
            new_word = NOISE_WORDS[len(pairs) % len(NOISE_WORDS)]
            new_list = [new_word] + word_list[1:]
            new_answer = ex['answer'] - 1

            pair_low = {
                'prompt': f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(new_list)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """,
                'word_list': new_list,
                'category': category,
                'answer': new_answer
            }
            pairs.append((pair_low, ex))

        else:
            # First word doesn't match - swap it with a category word to increase count
            new_word = CATEGORIES[category][len(pairs) % len(CATEGORIES[category])]
            new_list = [new_word] + word_list[1:]
            new_answer = ex['answer'] + 1

            pair_high = {
                'prompt': f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(new_list)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """,
                'word_list': new_list,
                'category': category,
                'answer': new_answer
            }
            pairs.append((ex, pair_high))

    print(f"Generated {len(pairs)} minimal pairs")
    if len(pairs) > 0:
        print(f"\nExample pair:")
        print(f"  Low:  {pairs[0][0]['word_list']} → answer={pairs[0][0]['answer']}")
        print(f"  High: {pairs[0][1]['word_list']} → answer={pairs[0][1]['answer']}")

    return pairs

def extract_activations(model, tokenizer, prompt: str, device: str) -> List[torch.Tensor]:
    """Extract activations from all layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            output_hidden_states=True
        )

    return [hidden.cpu() for hidden in outputs.hidden_states]

def patch_and_generate(model, tokenizer, prompt: str, patch_activation: torch.Tensor,
                       layer_idx: int, patch_pos: int, device: str) -> str:
    """Generate with patched activation at a specific layer and position.

    Args:
        patch_activation: Single position activation tensor [batch, hidden_dim]
    """
    # Defensive check
    if not isinstance(patch_activation, torch.Tensor):
        raise TypeError(f"patch_activation must be a torch.Tensor, got {type(patch_activation)}. "
                       f"Did you forget to reload mediation_utils after git pull?")
    if len(patch_activation.shape) != 2:
        raise ValueError(f"patch_activation must have shape [batch, hidden_dim], got {patch_activation.shape}")

    def patching_hook(module, input, output):
        # Handle both tuple output and direct tensor output
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Only patch if this is the initial forward pass (full sequence length)
            if hidden_states.shape[1] > patch_pos:
                hidden_states = hidden_states.clone()
                hidden_states[:, patch_pos, :] = patch_activation.to(device)
                return (hidden_states,) + output[1:]
            return output
        else:
            # Only patch if this is the initial forward pass (full sequence length)
            if output.shape[1] > patch_pos:
                output = output.clone()
                output[:, patch_pos, :] = patch_activation.to(device)
            return output

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(patching_hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return generated_text

    finally:
        handle.remove()

def run_mediation_analysis(model, tokenizer, pairs: List[Tuple[Dict, Dict]], device: str) -> Dict:
    """Run causal mediation analysis across all layers.

    Organized by example (not by layer) for efficiency.
    Baseline is the unpatched model output.
    """

    n_layers = len(model.model.layers)

    # Store results: baseline_outputs[i] = original output for pair i
    # layer_outputs[layer_idx][i] = patched output for pair i at layer_idx
    baseline_outputs = []
    layer_outputs = {i: [] for i in range(n_layers)}
    target_answers = []

    # First pass: benchmark baseline performance
    print("\n" + "="*80)
    print("BENCHMARKING BASELINE (unpatched model)")
    print("="*80)

    for idx, (pair_low, pair_high) in enumerate(tqdm(pairs, desc="Baseline evaluation")):
        target_answers.append(pair_high['answer'])

        # Get baseline (unpatched) output
        inputs_low = tokenizer(pair_low['prompt'], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_baseline = model.generate(
                inputs_low.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        baseline_text = tokenizer.decode(
            outputs_baseline[0][inputs_low.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        baseline_answer = extract_answer(baseline_text)
        baseline_outputs.append(baseline_answer)

        # Show first example in detail
        if idx == 0:
            print("\n" + "="*80)
            print("FIRST EXAMPLE PROMPT:")
            print("="*80)
            print(pair_low['prompt'])
            print("\n" + "="*80)
            print("MODEL RAW OUTPUT:")
            print("="*80)
            print(f"'{baseline_text}'")
            print("\n" + "="*80)
            print("PARSED RESULT:")
            print("="*80)
            print(f"Extracted answer: {baseline_answer}")
            print(f"Target answer: {pair_high['answer']}")
            print(f"Correct: {baseline_answer == pair_high['answer']}")
            print("="*80 + "\n")

    # Report baseline immediately
    baseline_correct = sum(1 for i in range(len(pairs))
                          if baseline_outputs[i] is not None
                          and baseline_outputs[i] == target_answers[i])
    baseline_accuracy = baseline_correct / len(pairs)

    # Show sample outputs for debugging
    print(f"\nSample baseline outputs (first 3):")
    for i in range(min(3, len(pairs))):
        print(f"  Example {i+1}: predicted={baseline_outputs[i]}, target={target_answers[i]}, " +
              f"correct={baseline_outputs[i] == target_answers[i]}")

    print(f"\nBaseline accuracy: {baseline_accuracy:.1%} ({baseline_correct}/{len(pairs)})")
    print("="*80 + "\n")

    # Second pass: run mediation analysis
    print("Running activation patching across all layers...")
    for idx, (pair_low, pair_high) in enumerate(tqdm(pairs, desc="Patching activations")):

        # 2. Extract activations from high-count prompt (all layers at once)
        inputs_high = tokenizer(pair_high['prompt'], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_high = model(
                input_ids=inputs_high.input_ids,
                output_hidden_states=True
            )

        # Determine patch position
        seq_len = inputs_low.input_ids.shape[1]
        last_pos = seq_len - 1
        max_pos = outputs_high.hidden_states[0].shape[1] - 1
        patch_pos = min(last_pos, max_pos)

        # 3. Patch each layer and generate
        for layer_idx in range(n_layers):
            # Extract activation for this layer at patch position
            activation_high_full = outputs_high.hidden_states[layer_idx].cpu()
            activation_at_pos = activation_high_full[:, patch_pos, :].clone()

            # Patch and generate
            generated = patch_and_generate(
                model, tokenizer, pair_low['prompt'],
                activation_at_pos, layer_idx, patch_pos, device
            )

            predicted = extract_answer(generated)
            layer_outputs[layer_idx].append(predicted)

        # Clear memory
        del outputs_high
        torch.cuda.empty_cache()

    # Compute metrics
    results = {
        'baseline_accuracy': 0,
        'layer_effects': {},
        'model_name': model.config._name_or_path,
        'n_pairs': len(pairs)
    }

    # Baseline accuracy
    baseline_correct = sum(1 for i in range(len(pairs))
                          if baseline_outputs[i] is not None
                          and baseline_outputs[i] == target_answers[i])
    results['baseline_accuracy'] = baseline_correct / len(pairs)

    # Per-layer metrics
    for layer_idx in range(n_layers):
        changed = []
        directional = []
        exact = []
        magnitude = []

        for i in range(len(pairs)):
            baseline = baseline_outputs[i]
            patched = layer_outputs[layer_idx][i]
            target = target_answers[i]

            if baseline is not None and patched is not None:
                changed.append(1 if patched != baseline else 0)
                directional.append(1 if patched == target else 0)
                exact.append(1 if patched == target else 0)
                magnitude.append(abs(patched - baseline))
            else:
                changed.append(0)
                directional.append(0)
                exact.append(0)
                magnitude.append(0)

        # Calculate mean_effect: average absolute change (matches original implementation)
        mean_effect = float(np.mean([abs(layer_outputs[layer_idx][i] - baseline_outputs[i])
                                     for i in range(len(pairs))
                                     if layer_outputs[layer_idx][i] is not None
                                     and baseline_outputs[i] is not None]))

        results['layer_effects'][str(layer_idx)] = {
            'changed_rate': float(np.mean(changed)),
            'exact_match_rate': float(np.mean(exact)),
            'mean_effect': mean_effect,  # Average shift in output (matches README metric)
            'std_effect': float(np.std(magnitude)),
            'n_exact_matches': int(sum(exact))
        }

    return results
