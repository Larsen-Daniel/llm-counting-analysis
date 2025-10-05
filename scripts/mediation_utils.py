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

def generate_minimal_pairs(dataset: List[Dict], n_pairs: int = 200) -> List[Tuple[Dict, Dict]]:
    """Generate minimal pairs by swapping the first word in the list."""

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

    for ex in dataset:
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

    Tracks three metrics per layer:
    1. changed: Whether the patch changed the answer at all
    2. directional: Whether it changed toward the target count
    3. exact: Whether it matched the target count exactly

    Also includes null hypothesis baseline (random sampling instead of patching).
    """

    n_layers = len(model.model.layers)
    layer_metrics = {
        i: {'changed': [], 'directional': [], 'exact': [], 'magnitude': []}
        for i in range(n_layers)
    }

    # Null hypothesis: generate with different random seed (temperature sampling)
    null_metrics = {'changed': [], 'directional': [], 'exact': [], 'magnitude': []}

    # Process one layer at a time to save memory
    for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
        for pair_idx, (pair_low, pair_high) in enumerate(pairs):
            # Get original (unpatched) answer first
            inputs_low = tokenizer(pair_low['prompt'], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_original = model.generate(
                    inputs_low.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            original_text = tokenizer.decode(
                outputs_original[0][inputs_low.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            original_answer = extract_answer(original_text)

            # Extract only the activation for this specific layer
            inputs_high = tokenizer(pair_high['prompt'], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs_high = model(
                    input_ids=inputs_high.input_ids,
                    output_hidden_states=True
                )

            # Get only this layer's activation and move to CPU immediately
            activation_high_full = outputs_high.hidden_states[layer_idx].cpu()

            # Clear GPU memory
            del outputs_high
            torch.cuda.empty_cache()

            seq_len = inputs_low.input_ids.shape[1]
            last_pos = seq_len - 1

            # Ensure patch position is within bounds
            max_pos = activation_high_full.shape[1] - 1
            patch_pos = min(last_pos, max_pos)

            # Extract activation at the patch position [batch, hidden_dim]
            activation_at_pos = activation_high_full[:, patch_pos, :].clone()

            # Patch just this layer
            generated = patch_and_generate(
                model, tokenizer, pair_low['prompt'],
                activation_at_pos, layer_idx, patch_pos, device
            )

            predicted = extract_answer(generated)

            # Null hypothesis baseline: generate with temperature sampling (only once per pair, on layer 0)
            if layer_idx == 0:
                with torch.no_grad():
                    outputs_null = model.generate(
                        inputs_low.input_ids,
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                null_text = tokenizer.decode(
                    outputs_null[0][inputs_low.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                null_answer = extract_answer(null_text)

                if null_answer is not None and original_answer is not None:
                    null_metrics['changed'].append(1 if null_answer != original_answer else 0)
                    null_metrics['directional'].append(1 if null_answer == pair_high['answer'] else 0)
                    null_metrics['exact'].append(1 if null_answer == pair_high['answer'] else 0)
                    null_metrics['magnitude'].append(abs(null_answer - original_answer))
                else:
                    null_metrics['changed'].append(0)
                    null_metrics['directional'].append(0)
                    null_metrics['exact'].append(0)
                    null_metrics['magnitude'].append(0)

            # Track three metrics for this layer
            if predicted is not None and original_answer is not None:
                # Metric 1: Did the answer change at all?
                did_change = (predicted != original_answer)
                layer_metrics[layer_idx]['changed'].append(1 if did_change else 0)

                # Metric 2: Did it change toward the target (exact match)?
                directional_correct = (predicted == pair_high['answer'])
                layer_metrics[layer_idx]['directional'].append(1 if directional_correct else 0)

                # Metric 3: Exact match (same as directional for this task)
                layer_metrics[layer_idx]['exact'].append(1 if directional_correct else 0)

                # Metric 4: Magnitude of change
                magnitude = abs(predicted - original_answer)
                layer_metrics[layer_idx]['magnitude'].append(magnitude)
            else:
                # If we couldn't parse answers, record as failure
                layer_metrics[layer_idx]['changed'].append(0)
                layer_metrics[layer_idx]['directional'].append(0)
                layer_metrics[layer_idx]['exact'].append(0)
                layer_metrics[layer_idx]['magnitude'].append(0)

            # Clear activation
            del activation_high_full
            del activation_at_pos

    results = {
        'layer_effects': {},
        'null_hypothesis': {},
        'model_name': model.config._name_or_path,
        'n_pairs': len(pairs)
    }

    # Add null hypothesis baseline
    results['null_hypothesis'] = {
        'changed_rate': float(np.mean(null_metrics['changed'])),
        'directional_accuracy': float(np.mean(null_metrics['directional'])),
        'exact_match_rate': float(np.mean(null_metrics['exact'])),
        'mean_magnitude': float(np.mean(null_metrics['magnitude'])),
        'std_magnitude': float(np.std(null_metrics['magnitude'])),
        'n_exact_matches': int(sum(null_metrics['exact']))
    }

    # Add per-layer results with statistical comparison to null
    for layer_idx in range(n_layers):
        metrics = layer_metrics[layer_idx]

        # Calculate p-value using binomial test for directional accuracy
        null_mean = results['null_hypothesis']['directional_accuracy']
        layer_mean = float(np.mean(metrics['directional']))

        # Simple binomial test: is this layer better than null baseline?
        n_successes = sum(metrics['directional'])
        n_total = len(metrics['directional'])

        # Two-tailed binomial test against null hypothesis proportion
        if null_mean > 0:
            p_value = stats.binom_test(n_successes, n_total, null_mean, alternative='two-sided')
        else:
            p_value = 1.0

        results['layer_effects'][str(layer_idx)] = {
            'changed_rate': float(np.mean(metrics['changed'])),
            'directional_accuracy': float(np.mean(metrics['directional'])),
            'exact_match_rate': float(np.mean(metrics['exact'])),
            'mean_magnitude': float(np.mean(metrics['magnitude'])),
            'std_magnitude': float(np.std(metrics['magnitude'])),
            'n_exact_matches': int(sum(metrics['exact'])),
            'p_value_vs_null': float(p_value),
            'better_than_null': layer_mean > null_mean
        }

    return results
