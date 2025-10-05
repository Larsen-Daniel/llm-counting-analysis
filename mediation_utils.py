import torch
import numpy as np
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import re

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
                'prompt': f"Count how many {category} are in this list: {', '.join(new_list)}. Answer with just the number in parentheses like (N).",
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
                'prompt': f"Count how many {category} are in this list: {', '.join(new_list)}. Answer with just the number in parentheses like (N).",
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

def patch_and_generate(model, tokenizer, prompt: str, patch_activations: torch.Tensor,
                       layer_idx: int, patch_pos: int, device: str) -> str:
    """Generate with patched activations."""

    def patching_hook(module, input, output):
        # Handle both tuple output and direct tensor output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Only patch if the position exists in the current activation
        if len(hidden_states.shape) == 3 and patch_pos < hidden_states.shape[1]:
            hidden_states[:, patch_pos, :] = patch_activations[layer_idx][:, patch_pos, :].to(device)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

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
    """Run causal mediation analysis across all layers."""

    n_layers = len(model.model.layers)
    layer_effects = {i: [] for i in range(n_layers)}

    # Process one layer at a time to save memory
    for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
        for pair_low, pair_high in pairs:
            # Extract only the activation for this specific layer
            inputs_high = tokenizer(pair_high['prompt'], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs_high = model(
                    input_ids=inputs_high.input_ids,
                    output_hidden_states=True
                )

            # Get only this layer's activation and move to CPU immediately
            activation_high = outputs_high.hidden_states[layer_idx].cpu()

            # Clear GPU memory
            del outputs_high
            torch.cuda.empty_cache()

            inputs_low = tokenizer(pair_low['prompt'], return_tensors="pt")
            seq_len = inputs_low.input_ids.shape[1]
            last_pos = seq_len - 1

            # Ensure patch position is within bounds
            max_pos = activation_high.shape[1] - 1
            patch_pos = min(last_pos, max_pos)

            # Patch just this layer
            generated = patch_and_generate(
                model, tokenizer, pair_low['prompt'],
                [activation_high], 0, patch_pos, device  # Pass single activation, use index 0
            )

            predicted = extract_answer(generated)

            if predicted == pair_high['answer']:
                layer_effects[layer_idx].append(1)
            else:
                layer_effects[layer_idx].append(0)

            # Clear activation
            del activation_high

    results = {
        'layer_effects': {},
        'model_name': model.config._name_or_path,
        'n_pairs': len(pairs)
    }

    for layer_idx in range(n_layers):
        effects = layer_effects[layer_idx]
        results['layer_effects'][str(layer_idx)] = {
            'mean_effect': float(np.mean(effects)),
            'std_effect': float(np.std(effects)),
            'n_successes': int(sum(effects))
        }

    return results
