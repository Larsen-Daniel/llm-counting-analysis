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
    """Generate minimal pairs where count differs by exactly 1."""
    pairs = []

    # Group by category and answer
    by_category = {}
    for ex in dataset:
        key = (ex['category'], ex['answer'])
        if key not in by_category:
            by_category[key] = []
        by_category[key].append(ex)

    # Find pairs with adjacent counts
    for (cat, ans) in by_category:
        if (cat, ans + 1) in by_category:
            examples_low = by_category[(cat, ans)]
            examples_high = by_category[(cat, ans + 1)]

            for low in examples_low[:10]:  # Limit per category
                for high in examples_high[:10]:
                    if len(pairs) >= n_pairs:
                        return pairs
                    pairs.append((low, high))

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
        output[0][:, patch_pos, :] = patch_activations[layer_idx][:, patch_pos, :].to(device)
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
    """Run causal mediation analysis across all layers."""

    n_layers = len(model.model.layers)
    layer_effects = {i: [] for i in range(n_layers)}

    for pair_low, pair_high in tqdm(pairs, desc="Mediation analysis"):
        activations_high = extract_activations(model, tokenizer, pair_high['prompt'], device)

        inputs_low = tokenizer(pair_low['prompt'], return_tensors="pt")
        last_pos = inputs_low.input_ids.shape[1] - 1

        for layer_idx in range(n_layers):
            generated = patch_and_generate(
                model, tokenizer, pair_low['prompt'],
                activations_high, layer_idx, last_pos, device
            )

            predicted = extract_answer(generated)

            if predicted == pair_high['answer']:
                layer_effects[layer_idx].append(1)
            else:
                layer_effects[layer_idx].append(0)

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
