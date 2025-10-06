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

def generate_minimal_pairs(dataset: List[Dict], n_pairs: int = 200, list_length: int = 4) -> List[Tuple[Dict, Dict]]:
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

    print(f"Generating pairs with list_length={list_length}")

    for ex in dataset:
        if len(pairs) >= n_pairs:
            break

        category = ex['category']

        # Take first list_length words from the example
        word_list = ex['word_list'][:list_length]

        # Recalculate answer for truncated list
        correct_count = sum(1 for word in word_list if word in CATEGORIES.get(category, []))

        first_word = word_list[0]

        # Check if first word matches the category
        if first_word in CATEGORIES.get(category, []):
            # First word matches - swap it with a noise word to decrease count
            new_word = NOISE_WORDS[len(pairs) % len(NOISE_WORDS)]
            new_list = [new_word] + word_list[1:]
            new_answer = correct_count - 1

            pair_low = {
                'prompt': f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(new_list)}

IMPORTANT: Respond with ONLY a number in parentheses. Nothing else.

Example:
Type: fruit
List: apple door banana cloud
Answer: (2)

Do NOT write "2 (2)" or "The answer is (2)" or any other text.
ONLY write: (N)

Answer: """,
                'word_list': new_list,
                'category': category,
                'answer': new_answer
            }
            # Create pair_high with original list
            pair_high_dict = {
                'prompt': ex['prompt'],
                'word_list': word_list,
                'category': category,
                'answer': correct_count
            }
            pairs.append((pair_low, pair_high_dict))

        else:
            # First word doesn't match - swap it with a category word to increase count
            new_word = CATEGORIES[category][len(pairs) % len(CATEGORIES[category])]
            new_list = [new_word] + word_list[1:]
            new_answer = correct_count + 1

            pair_high = {
                'prompt': f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(new_list)}

IMPORTANT: Respond with ONLY a number in parentheses. Nothing else.

Example:
Type: fruit
List: apple door banana cloud
Answer: (2)

Do NOT write "2 (2)" or "The answer is (2)" or any other text.
ONLY write: (N)

Answer: """,
                'word_list': new_list,
                'category': category,
                'answer': new_answer
            }
            # Create pair_low with original list
            pair_low_dict = {
                'prompt': ex['prompt'],
                'word_list': word_list,
                'category': category,
                'answer': correct_count
            }
            pairs.append((pair_low_dict, pair_high))

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

    # First pass: benchmark baseline performance and filter to perfect examples
    print("\n" + "="*80)
    print("BENCHMARKING BASELINE (filtering to perfect examples)")
    print("="*80)

    filtered_pairs = []
    tested_count = 0

    for idx, (pair_low, pair_high) in enumerate(pairs):
        if len(filtered_pairs) >= 20:
            print(f"\n✓ Found 20 perfect examples after testing {tested_count} pairs")
            break

        tested_count += 1

        # Skip if answer is 0 (too easy)
        if pair_low['answer'] == 0:
            print(f"⊘  Test {tested_count}: Skipped (answer=0)")
            continue

        # Get baseline (unpatched) output - greedy decoding
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

        # Skip if greedy doesn't match target (no point sampling)
        if baseline_answer != pair_low['answer']:
            word_list_str = ' '.join(pair_low['word_list'])
            print(f"✗  Test {tested_count} ({pair_low['category']}): [{word_list_str}] → greedy={baseline_answer}, target={pair_low['answer']} - SKIP")
            continue

        # Sample 4 times with temperature=0.7 to measure reliability
        sampled_answers = []
        for _ in range(4):
            with torch.no_grad():
                outputs_sampled = model.generate(
                    inputs_low.input_ids,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            sampled_text = tokenizer.decode(
                outputs_sampled[0][inputs_low.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            sampled_answer = extract_answer(sampled_text)
            sampled_answers.append(sampled_answer)

        # Only keep if model gets it right 100% of the time (5/5 including greedy)
        correct_samples = sum(1 for ans in sampled_answers if ans == pair_low['answer'])
        is_perfect = (correct_samples == 4 and baseline_answer == pair_low['answer'])

        # Show result
        status = "✓✓" if is_perfect else "✗ "
        word_list_str = ' '.join(pair_low['word_list'])
        print(f"{status} Test {tested_count} ({pair_low['category']}): [{word_list_str}] → greedy={baseline_answer}, target={pair_low['answer']}, " +
              f"sampled={sampled_answers}, p={correct_samples}/4")

        if is_perfect:
            filtered_pairs.append((pair_low, pair_high))
            baseline_outputs.append(baseline_answer)
            target_answers.append(pair_low['answer'])

    if len(filtered_pairs) < 20:
        print(f"\nWARNING: Only found {len(filtered_pairs)} perfect examples out of {tested_count} tested")

    # Use filtered pairs for rest of analysis
    pairs = filtered_pairs

    # Report baseline immediately
    if len(pairs) == 0:
        print("\nERROR: No pairs found after filtering. Cannot proceed.")
        return {'error': 'No perfect examples found'}

    baseline_correct = sum(1 for i in range(len(pairs))
                          if baseline_outputs[i] is not None
                          and baseline_outputs[i] == target_answers[i])
    baseline_accuracy = baseline_correct / len(pairs)

    print(f"\nBaseline accuracy: {baseline_accuracy:.1%} ({baseline_correct}/{len(pairs)})")
    print("="*80 + "\n")

    # Second pass: run mediation analysis
    print("\n" + "="*80)
    print("ACTIVATION PATCHING ACROSS ALL LAYERS")
    print("="*80)

    for idx, (pair_low, pair_high) in enumerate(pairs):
        low_words = ' '.join(pair_low['word_list'])
        high_words = ' '.join(pair_high['word_list'])
        print(f"\nExample {idx+1}/{len(pairs)}:")
        print(f"  Low  (answer={pair_low['answer']}): {low_words}")
        print(f"  High (answer={pair_high['answer']}): {high_words}")
        print(f"  Target after patching: {pair_low['answer']} → {pair_high['answer']}")

        # 2. Extract activations from high-count prompt (all layers at once)
        inputs_low = tokenizer(pair_low['prompt'], return_tensors="pt").to(device)
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

            # Show effect of this layer's patch
            baseline_val = baseline_outputs[idx]
            target_val = baseline_val + 1  # pair_high answer
            changed = "→" if predicted != baseline_val else "="
            hit_target = "✓" if predicted == target_val else " "
            print(f"  Layer {layer_idx:2d}: {baseline_val} {changed} {predicted} {hit_target}")

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
            baseline_target = target_answers[i]  # pair_low answer
            patch_target = baseline_target + 1  # pair_high answer (always +1)

            if baseline is not None and patched is not None:
                changed.append(1 if patched != baseline else 0)
                directional.append(1 if patched == patch_target else 0)
                exact.append(1 if patched == patch_target else 0)
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
