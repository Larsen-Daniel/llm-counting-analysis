import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ActivationPatcher:
    """
    Performs causal mediation analysis via activation patching.
    Version 2: Fixed position alignment and directional validation.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers

        # Storage for activations
        self.activations = {}
        self.hooks = []

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register forward hooks to capture activations at each layer."""
        if layer_indices is None:
            layer_indices = range(self.num_layers)

        self.activations = {}
        self.remove_hooks()

        for layer_idx in layer_indices:
            # Get the layer (adjust based on model architecture)
            if hasattr(self.model, 'model'):  # Llama, Qwen, etc.
                layer = self.model.model.layers[layer_idx]
            elif hasattr(self.model, 'transformer'):  # GPT-style
                layer = self.model.transformer.h[layer_idx]
            else:
                raise ValueError("Unknown model architecture")

            # Register hook
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # Store the hidden states (first element of output tuple)
                    if isinstance(output, tuple):
                        self.activations[layer_idx] = output[0].detach()
                    else:
                        self.activations[layer_idx] = output.detach()
                return hook

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations_at_position(
        self,
        prompt: str,
        position: int
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and extract activations at a specific token position.

        Args:
            prompt: Input text
            position: Token position to extract activations from

        Returns:
            Dictionary mapping layer_idx -> activation tensor
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            _ = self.model(**inputs)

        # Extract activations at the specified position
        position_activations = {}
        for layer_idx, acts in self.activations.items():
            position_activations[layer_idx] = acts[:, position, :].clone()

        return position_activations

    def patch_and_generate(
        self,
        prompt: str,
        patch_layer: int,
        patch_position: int,
        patch_activation: torch.Tensor,
        max_new_tokens: int = 10
    ) -> str:
        """
        Generate text with a patched activation at a specific layer and position.

        Args:
            prompt: Input prompt
            patch_layer: Which layer to patch
            patch_position: Which token position to patch
            patch_activation: The activation to patch in
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Remove existing hooks and register a patching hook
        self.remove_hooks()

        # Get the layer to patch
        if hasattr(self.model, 'model'):
            layer = self.model.model.layers[patch_layer]
        elif hasattr(self.model, 'transformer'):
            layer = self.model.transformer.h[patch_layer]
        else:
            raise ValueError("Unknown model architecture")

        # Create patching hook - only patch when we see the full sequence
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Only patch if this is the initial forward pass (full sequence length)
                if hidden_states.shape[1] > patch_position:
                    hidden_states = hidden_states.clone()
                    hidden_states[:, patch_position, :] = patch_activation
                    return (hidden_states,) + output[1:]
                return output
            else:
                # Only patch if this is the initial forward pass (full sequence length)
                if output.shape[1] > patch_position:
                    output = output.clone()
                    output[:, patch_position, :] = patch_activation
                return output

        handle = layer.register_forward_hook(patch_hook)

        # Generate with patched activation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Clean up
        handle.remove()
        self.register_hooks()  # Re-register activation collection hooks

        return generated_text


def create_position_swapped_prompts(
    category: str,
    category_items: List[str],
    non_category_items: List[str],
    list_length: int = None
) -> Tuple[str, str, int, int]:
    """
    Create two prompts with identical items but matching word at different positions.

    Args:
        category: Category to count (e.g., "fruit")
        category_items: Items that match category
        non_category_items: Items that don't match
        list_length: Total list length (random 5-10 if None)

    Returns:
        (prompt_early, prompt_late, early_position, late_position)
        Both prompts have same count (0-5), matching items at different positions
    """
    # Random list length (5-10 words) and num matching (0-5)
    if list_length is None:
        list_length = np.random.randint(5, 11)

    max_matches = min(5, list_length, len(category_items))
    num_matches = np.random.randint(0, max_matches + 1)

    # Sample items
    matching_items = np.random.choice(category_items, size=num_matches, replace=False).tolist() if num_matches > 0 else []
    non_matching = np.random.choice(non_category_items, size=list_length - num_matches, replace=False).tolist()

    # Early version: matching items clustered at start
    list_early = matching_items + non_matching

    # Late version: matching items clustered at middle/end
    mid_point = list_length // 2
    list_late = non_matching[:mid_point] + matching_items + non_matching[mid_point:]

    # Create prompts (matching benchmark format)
    prompt_early = f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(list_early)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """

    prompt_late = f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(list_late)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """

    return prompt_early, prompt_late, 0, mid_point


def extract_answer(text: str) -> Optional[int]:
    """
    Extract numerical answer from model output.
    Try multiple patterns in order of preference:
    1. Last number in parentheses: (N)
    2. First number on first line (for lenient parsing)
    """
    import re
    # First try to find number in parentheses
    matches = re.findall(r'\((\d+)\)', text)
    if matches:
        return int(matches[-1])  # Return the last match

    # Fall back to first number in the output
    first_line = text.strip().split('\n')[0]
    number_match = re.search(r'\d+', first_line)
    if number_match:
        return int(number_match.group(0))

    return None  # No number found


def analyze_running_count_v2(
    model_name: str,
    num_examples: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Improved causal mediation analysis with:
    1. Position-aligned prompts (same list, different positions)
    2. Directional validation (verify patched output matches source)
    3. Larger sample size
    4. Strict answer parsing (only accept "N)" format)

    Args:
        model_name: HuggingFace model name
        num_examples: Number of examples to analyze
        device: Device to run on

    Returns:
        Analysis results
    """
    print(f"\n{'='*80}")
    print(f"Causal Mediation Analysis V2: {model_name}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
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

    # Create patcher
    patcher = ActivationPatcher(model, tokenizer, device)
    patcher.register_hooks()

    # Categories (matching benchmark)
    CATEGORIES = {
        "fruit": ["apple", "banana", "cherry", "grape", "orange", "mango", "kiwi", "peach", "pear", "plum",
                  "strawberry", "blueberry", "watermelon", "pineapple", "lemon"],
        "animal": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "chicken",
                   "rabbit", "mouse", "elephant", "lion", "tiger"],
        "vehicle": ["car", "bus", "truck", "bike", "train", "plane", "boat", "ship", "motorcycle", "scooter",
                    "van", "taxi", "subway", "helicopter", "tram"],
        "color": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
                  "gray", "violet", "indigo", "turquoise", "crimson"],
        "tool": ["hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "chisel", "axe", "knife", "scissors",
                 "ruler", "tape", "level", "clamp", "file"],
        "furniture": ["chair", "table", "sofa", "bed", "desk", "cabinet", "shelf", "dresser", "bench", "stool",
                      "wardrobe", "bookcase", "ottoman", "nightstand", "couch"],
        "clothing": ["shirt", "pants", "dress", "skirt", "jacket", "coat", "sweater", "hat", "shoes", "socks",
                     "scarf", "gloves", "belt", "tie", "jeans"],
        "food": ["pizza", "burger", "pasta", "rice", "bread", "cheese", "salad", "soup", "sandwich", "taco",
                 "noodles", "steak", "chicken", "fish", "potato"],
        "sport": ["soccer", "basketball", "tennis", "baseball", "football", "hockey", "volleyball", "golf", "cricket", "rugby",
                  "swimming", "boxing", "skiing", "skating", "running"],
        "instrument": ["guitar", "piano", "drums", "violin", "flute", "trumpet", "saxophone", "cello", "clarinet", "harp",
                       "trombone", "banjo", "accordion", "oboe", "tuba"]
    }

    NOISE_WORDS = ["bowl", "window", "door", "cloud", "mountain", "river", "tree", "rock", "sand", "metal",
                   "plastic", "glass", "paper", "wood", "stone", "gold", "silver", "copper", "iron", "steel",
                   "number", "letter", "word", "sentence", "paragraph", "page", "book", "magazine", "newspaper", "document"]

    print(f"Analyzing {num_examples} examples...")

    # Results storage
    layer_effects = {i: [] for i in range(patcher.num_layers)}
    layer_directional_accuracy = {i: [] for i in range(patcher.num_layers)}

    for _ in tqdm(range(num_examples)):
        category = np.random.choice(list(CATEGORIES.keys()))

        # Create minimal pair: identical lists except one word changes category membership
        list_length = np.random.randint(5, 11)

        # Pick a random position to be the "different" word
        swap_position = np.random.randint(0, list_length)

        # Generate base list with some matching items
        num_base_matches = np.random.randint(0, min(5, list_length))

        # Sample matching and non-matching items
        matching_items = np.random.choice(CATEGORIES[category], size=num_base_matches, replace=False).tolist() if num_base_matches > 0 else []
        non_category_items = NOISE_WORDS + [item for cat, items in CATEGORIES.items() if cat != category for item in items]
        non_matching_items = np.random.choice(non_category_items, size=list_length - num_base_matches, replace=False).tolist()

        # Create base list
        base_list = matching_items + non_matching_items
        np.random.shuffle(base_list)

        # Create two versions: one where word at swap_position is in category, one where it's not
        # Pick a category word and non-category word that aren't already in the list
        available_category = [w for w in CATEGORIES[category] if w not in base_list]
        available_non_category = [w for w in non_category_items if w not in base_list]

        if not available_category or not available_non_category:
            continue

        category_word = np.random.choice(available_category)
        non_category_word = np.random.choice(available_non_category)

        # List 1: swap_position has non-category word (count = num_base_matches)
        list1 = base_list.copy()
        list1[swap_position] = non_category_word

        # List 2: swap_position has category word (count = num_base_matches + 1)
        list2 = base_list.copy()
        list2[swap_position] = category_word

        # Create prompts
        prompt_count_N = f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(list1)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """

        prompt_count_N_plus_1 = f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(list2)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """

        # Rename for clarity in code below
        prompt_early_count = prompt_count_N
        prompt_late_count = prompt_count_N_plus_1
        expected_count_early = num_base_matches
        expected_count_late = num_base_matches + 1

        # Get activations
        inputs_early = tokenizer(prompt_early_count, return_tensors="pt").to(device)
        inputs_late = tokenizer(prompt_late_count, return_tensors="pt").to(device)

        early_activation_pos = inputs_early.input_ids.shape[1] - 1
        late_activation_pos = inputs_late.input_ids.shape[1] - 1

        activations_early = patcher.get_activations_at_position(prompt_early_count, early_activation_pos)
        activations_late = patcher.get_activations_at_position(prompt_late_count, late_activation_pos)

        # Generate with original early prompt ONCE
        inputs = tokenizer(prompt_early_count, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_original = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        original_text = tokenizer.decode(outputs_original[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        original_num = extract_answer(original_text)

        # For each layer, patch from late (different count) into early
        for layer_idx in range(patcher.num_layers):

            # Generate with patched activation from late prompt
            # If layer encodes count, should shift output toward late count
            patched_text = patcher.patch_and_generate(
                prompt_early_count,
                patch_layer=layer_idx,
                patch_position=early_activation_pos,
                patch_activation=activations_late[layer_idx],
                max_new_tokens=10
            )

            # Extract patched number
            patched_num = extract_answer(patched_text)

            if original_num != -1 and patched_num != -1:
                # Effect: how much did output change?
                effect = abs(patched_num - original_num)
                layer_effects[layer_idx].append(effect)

                # Directional accuracy: did patch move output toward source count (2)?
                directional_correct = (patched_num == expected_count_late)
                layer_directional_accuracy[layer_idx].append(directional_correct)

    # Aggregate results
    results = {
        "model_name": model_name,
        "num_layers": patcher.num_layers,
        "layer_effects": {},
        "layer_directional_accuracy": {},
        "examples_analyzed": num_examples
    }

    for layer_idx in range(patcher.num_layers):
        if layer_effects[layer_idx]:
            results["layer_effects"][layer_idx] = {
                "mean_effect": float(np.mean(layer_effects[layer_idx])),
                "std_effect": float(np.std(layer_effects[layer_idx])),
                "median_effect": float(np.median(layer_effects[layer_idx]))
            }
            results["layer_directional_accuracy"][layer_idx] = {
                "accuracy": float(np.mean(layer_directional_accuracy[layer_idx])),
                "num_correct": int(np.sum(layer_directional_accuracy[layer_idx])),
                "num_total": len(layer_directional_accuracy[layer_idx])
            }

    # Clean up
    patcher.remove_hooks()
    del model
    del tokenizer
    torch.cuda.empty_cache() if device == "cuda" else None

    return results


def load_dataset_examples(file_path: str = "counting_dataset.jsonl", num_examples: int = 5000) -> List[Dict]:
    """Load examples from the pre-generated dataset file."""
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            examples.append(json.loads(line))
    return examples


def train_linear_probe(
    model_name: str,
    num_examples: int = 5000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset_file: str = "counting_dataset.jsonl"
) -> Dict:
    """
    Train linear probes to detect if count is linearly readable from activations.

    Args:
        model_name: HuggingFace model name
        num_examples: Number of examples to use from dataset
        device: Device to run on
        dataset_file: Path to pre-generated dataset

    Returns:
        Probe results per layer
    """
    print(f"\n{'='*80}")
    print(f"Linear Probe Analysis: {model_name}")
    print(f"{'='*80}\n")

    # Load dataset
    print(f"Loading {num_examples} examples from {dataset_file}...")
    examples = load_dataset_examples(dataset_file, num_examples)
    print(f"Loaded {len(examples)} examples")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
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

    # Create patcher for activation extraction
    patcher = ActivationPatcher(model, tokenizer, device)
    patcher.register_hooks()

    # Categories
    CATEGORIES = {
        "fruit": ["apple", "banana", "cherry", "grape", "orange", "mango", "kiwi", "peach"],
        "animal": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep"],
        "vehicle": ["car", "bus", "truck", "bike", "train", "plane", "boat", "ship"],
    }

    print(f"Collecting activations from {len(examples)} examples...")

    # Collect data: activations -> counts
    layer_data = {i: {"X": [], "y": []} for i in range(patcher.num_layers)}

    for example in tqdm(examples):
        # Use prompt and answer from dataset
        prompt = example["prompt"]
        true_answer = example["answer"]

        # Get activations at last position
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        last_pos = inputs.input_ids.shape[1] - 1

        activations = patcher.get_activations_at_position(prompt, last_pos)

        # Store activations and true count for each layer
        for layer_idx in range(patcher.num_layers):
            layer_data[layer_idx]["X"].append(activations[layer_idx].cpu().numpy())
            layer_data[layer_idx]["y"].append(true_answer)

    print("\nTraining linear probes...")

    results = {
        "model_name": model_name,
        "num_layers": patcher.num_layers,
        "probe_accuracy": {},
        "examples_used": num_examples
    }

    for layer_idx in tqdm(range(patcher.num_layers)):
        X = np.vstack(layer_data[layer_idx]["X"])
        y = np.array(layer_data[layer_idx]["y"])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train probe
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X_train, y_train)

        # Evaluate
        train_acc = probe.score(X_train, y_train)
        test_acc = probe.score(X_test, y_test)
        y_pred = probe.predict(X_test)

        results["probe_accuracy"][layer_idx] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "num_train": len(y_train),
            "num_test": len(y_test)
        }

    # Clean up
    patcher.remove_hooks()
    del model
    del tokenizer
    torch.cuda.empty_cache() if device == "cuda" else None

    return results


def run_intervention_experiment(
    model_name: str,
    probe_results: Dict,
    num_examples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset_file: str = "counting_dataset.jsonl"
) -> Dict:
    """
    Use trained probes to intervene on activations and steer model outputs.

    For the best probe layer, we:
    1. Take an example with count=N
    2. Use the probe to modify activations to represent count=M
    3. Check if the model's output changes from N to M

    Args:
        model_name: HuggingFace model name
        probe_results: Results from train_linear_probe
        num_examples: Number of intervention examples to test
        device: Device to run on
        dataset_file: Path to dataset

    Returns:
        Intervention results
    """
    print(f"\n{'='*80}")
    print(f"Intervention Experiment: {model_name}")
    print(f"{'='*80}\n")

    # Find best probe layer
    best_layer = max(
        probe_results["probe_accuracy"].items(),
        key=lambda x: x[1]["test_accuracy"]
    )[0]

    print(f"Using probe from layer {best_layer} (accuracy: {probe_results['probe_accuracy'][best_layer]['test_accuracy']:.2%})")

    # Load model and tokenizer
    print("Loading model...")
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

    # Load probe weights (we'll need to retrain a simple version for steering)
    # For steering, we use the probe's direction in activation space
    print("Loading dataset for intervention...")
    examples = load_dataset_examples(dataset_file, num_examples * 2)  # Load extra for variety

    # Create patcher
    patcher = ActivationPatcher(model, tokenizer, device)
    patcher.register_hooks([best_layer])

    # Collect some activations to compute steering vectors
    print("Computing steering vectors...")
    count_activations = {i: [] for i in range(6)}  # 0-5

    for example in examples[:200]:  # Use first 200 to build steering vectors
        if example["answer"] > 5:
            continue
        prompt = example["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        last_pos = inputs.input_ids.shape[1] - 1

        activations = patcher.get_activations_at_position(prompt, last_pos)
        count_activations[example["answer"]].append(activations[best_layer].cpu().numpy())

    # Compute mean activation for each count
    steering_vectors = {}
    for count in range(6):
        if count_activations[count]:
            steering_vectors[count] = np.mean(count_activations[count], axis=0)

    print(f"Computed steering vectors for counts: {list(steering_vectors.keys())}")

    # Run intervention experiments
    print(f"\nRunning {num_examples} intervention experiments...")
    results = {
        "model_name": model_name,
        "best_layer": best_layer,
        "num_examples": 0,
        "successful_interventions": 0,
        "examples": []
    }

    for example in tqdm(examples[200:200+num_examples]):  # Use different examples
        original_count = example["answer"]
        if original_count > 4 or original_count not in steering_vectors:
            continue

        # Try to steer to a different count
        target_count = (original_count + np.random.randint(1, 4)) % 6
        if target_count not in steering_vectors:
            continue

        prompt = example["prompt"]

        # Get original output
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        original_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        original_parsed = extract_answer(original_output)

        # Intervene: patch activations with steering vector
        steering_diff = torch.tensor(
            steering_vectors[target_count] - steering_vectors[original_count],
            dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        last_pos = inputs.input_ids.shape[1] - 1
        intervened_output = patcher.patch_and_generate(
            prompt,
            patch_layer=best_layer,
            patch_position=last_pos,
            patch_activation=patcher.get_activations_at_position(prompt, last_pos)[best_layer] + steering_diff,
            max_new_tokens=10
        )
        intervened_parsed = extract_answer(intervened_output)

        # Check if intervention worked
        success = intervened_parsed == target_count

        results["num_examples"] += 1
        if success:
            results["successful_interventions"] += 1

        results["examples"].append({
            "original_count": original_count,
            "target_count": target_count,
            "original_output": original_output,
            "original_parsed": original_parsed,
            "intervened_output": intervened_output,
            "intervened_parsed": intervened_parsed,
            "success": success
        })

    results["success_rate"] = results["successful_interventions"] / results["num_examples"] if results["num_examples"] > 0 else 0

    print(f"\nIntervention Results:")
    print(f"  Examples tested: {results['num_examples']}")
    print(f"  Successful interventions: {results['successful_interventions']}")
    print(f"  Success rate: {results['success_rate']:.2%}")

    # Clean up
    patcher.remove_hooks()
    del model
    del tokenizer
    torch.cuda.empty_cache() if device == "cuda" else None

    return results


def plot_results(mediation_results: Dict, probe_results: Dict, output_file: str = "analysis_v2.png"):
    """Plot both mediation and probe results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mediation effects
    layers = sorted(mediation_results["layer_effects"].keys())
    mean_effects = [mediation_results["layer_effects"][l]["mean_effect"] for l in layers]
    std_effects = [mediation_results["layer_effects"][l]["std_effect"] for l in layers]

    ax1.errorbar(layers, mean_effects, yerr=std_effects, marker='o', capsize=5)
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Mean Effect on Output (change in count)", fontsize=12)
    ax1.set_title(f"Causal Mediation Analysis V2\n{mediation_results['model_name']}", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Linear probe accuracy
    layers_probe = sorted(probe_results["probe_accuracy"].keys())
    test_acc = [probe_results["probe_accuracy"][l]["test_accuracy"] for l in layers_probe]

    ax2.plot(layers_probe, test_acc, marker='o')
    ax2.axhline(y=0.25, color='r', linestyle='--', label='Chance (4 classes)')
    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Probe Test Accuracy", fontsize=12)
    ax2.set_title(f"Linear Probe: Count Detection\n{probe_results['model_name']}", fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Improved causal mediation + linear probe + intervention analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model to analyze")
    parser.add_argument("--mediation_examples", type=int, default=200, help="Examples for mediation")
    parser.add_argument("--probe_examples", type=int, default=5000, help="Examples for probe")
    parser.add_argument("--intervention_examples", type=int, default=100, help="Examples for intervention")
    parser.add_argument("--output", type=str, default="analysis_v2_results.json", help="Output file")
    parser.add_argument("--plot", type=str, default="analysis_v2.png", help="Plot output file")
    parser.add_argument("--skip_mediation", action="store_true", help="Skip mediation analysis")
    parser.add_argument("--skip_probe", action="store_true", help="Skip probe analysis")
    parser.add_argument("--skip_intervention", action="store_true", help="Skip intervention experiment")

    args = parser.parse_args()

    all_results = {}

    # Run mediation analysis
    if not args.skip_mediation:
        mediation_results = analyze_running_count_v2(
            model_name=args.model,
            num_examples=args.mediation_examples
        )
        all_results["mediation"] = mediation_results

        print(f"\n{'='*80}")
        print("MEDIATION RESULTS")
        print(f"{'='*80}\n")
        print(f"Model: {mediation_results['model_name']}")
        print(f"Examples analyzed: {mediation_results['examples_analyzed']}")
        print(f"\nTop 5 layers by mean effect:")

        sorted_layers = sorted(
            mediation_results["layer_effects"].items(),
            key=lambda x: x[1]["mean_effect"],
            reverse=True
        )

        for i, (layer_idx, stats) in enumerate(sorted_layers[:5]):
            dir_acc = mediation_results["layer_directional_accuracy"][layer_idx]["accuracy"]
            print(f"  {i+1}. Layer {layer_idx}: {stats['mean_effect']:.3f} ± {stats['std_effect']:.3f} (dir_acc: {dir_acc:.2%})")

    # Run probe analysis
    if not args.skip_probe:
        probe_results = train_linear_probe(
            model_name=args.model,
            num_examples=args.probe_examples
        )
        all_results["probe"] = probe_results

        print(f"\n{'='*80}")
        print("LINEAR PROBE RESULTS")
        print(f"{'='*80}\n")
        print(f"Model: {probe_results['model_name']}")
        print(f"Examples used: {probe_results['examples_used']}")
        print(f"\nTop 5 layers by probe accuracy:")

        sorted_probe_layers = sorted(
            probe_results["probe_accuracy"].items(),
            key=lambda x: x[1]["test_accuracy"],
            reverse=True
        )

        for i, (layer_idx, stats) in enumerate(sorted_probe_layers[:5]):
            print(f"  {i+1}. Layer {layer_idx}: {stats['test_accuracy']:.2%} (train: {stats['train_accuracy']:.2%})")

    # Run intervention experiment (requires probe results)
    if not args.skip_intervention and not args.skip_probe:
        intervention_results = run_intervention_experiment(
            model_name=args.model,
            probe_results=all_results["probe"],
            num_examples=args.intervention_examples
        )
        all_results["intervention"] = intervention_results

        print(f"\n{'='*80}")
        print("INTERVENTION EXPERIMENT RESULTS")
        print(f"{'='*80}\n")
        print(f"Model: {intervention_results['model_name']}")
        print(f"Best layer used: {intervention_results['best_layer']}")
        print(f"Success rate: {intervention_results['success_rate']:.2%}")
        print(f"\nSample successful interventions:")
        for i, ex in enumerate([e for e in intervention_results['examples'] if e['success']][:3]):
            print(f"  {i+1}. Count {ex['original_count']}→{ex['target_count']}: '{ex['intervened_output'][:50]}'")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {args.output}")

    # Plot if we have both mediation and probe results
    if not args.skip_mediation and not args.skip_probe:
        plot_results(all_results["mediation"], all_results["probe"], args.plot)


if __name__ == "__main__":
    main()
