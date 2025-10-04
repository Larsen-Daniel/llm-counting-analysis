import random
import json
from typing import List, Tuple

# Define categories and their items
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

# Additional noise words that don't belong to any category
NOISE_WORDS = ["bowl", "window", "door", "cloud", "mountain", "river", "tree", "rock", "sand", "metal",
               "plastic", "glass", "paper", "wood", "stone", "gold", "silver", "copper", "iron", "steel",
               "number", "letter", "word", "sentence", "paragraph", "page", "book", "magazine", "newspaper", "document"]

def generate_example(list_length: int = None, category: str = None) -> Tuple[str, str, List[str], int]:
    """
    Generate a single example.

    Returns:
        Tuple of (prompt, category, word_list, answer)
    """
    if list_length is None:
        list_length = random.randint(5, 10)  # Max 10 words

    if category is None:
        category = random.choice(list(CATEGORIES.keys()))

    category_items = CATEGORIES[category]

    # Determine how many matching items to include (0 to 5, avg ~2.5)
    max_matches = min(5, list_length, len(category_items))
    num_matches = random.randint(0, max_matches)

    # Sample matching items
    matching_items = random.sample(category_items, num_matches) if num_matches > 0 else []

    # Get non-matching items from other categories and noise words
    non_matching_pool = NOISE_WORDS.copy()
    for cat, items in CATEGORIES.items():
        if cat != category:
            non_matching_pool.extend(items)

    # Sample non-matching items
    num_non_matches = list_length - num_matches
    non_matching_items = random.sample(non_matching_pool, num_non_matches)

    # Combine and shuffle
    word_list = matching_items + non_matching_items
    random.shuffle(word_list)

    # Create prompt with extremely clear instructions
    prompt = f"""Count how many words in the list below match the given type.

Type: {category}
List: {' '.join(word_list)}

YOU MUST respond with ONLY a number in parentheses, like this: (5)
Do NOT include any other text, explanations, or words.
Just output the number in parentheses and nothing else.

Answer: """

    # Expected completion is "(N)"
    expected_completion = f"({num_matches})"

    return prompt, category, word_list, num_matches, expected_completion

def generate_dataset(num_examples: int = 5000, output_file: str = "counting_dataset.jsonl") -> None:
    """
    Generate a dataset of counting examples.

    Args:
        num_examples: Number of examples to generate
        output_file: Path to output JSONL file
    """
    examples = []

    for i in range(num_examples):
        # All lists are 5-10 words
        list_length = random.randint(5, 10)
        prompt, category, word_list, answer, expected_completion = generate_example(list_length=list_length)

        examples.append({
            "prompt": prompt,
            "category": category,
            "word_list": word_list,
            "answer": answer,
            "list_length": list_length,
            "expected_completion": expected_completion
        })

    # Save to JSONL file
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Generated {num_examples} examples and saved to {output_file}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(examples)}")
    print(f"Categories: {list(CATEGORIES.keys())}")
    print(f"List length range: {min(ex['list_length'] for ex in examples)} - {max(ex['list_length'] for ex in examples)}")
    print(f"Answer range: {min(ex['answer'] for ex in examples)} - {max(ex['answer'] for ex in examples)}")

    # Sample examples
    print("\n--- Sample Examples ---")
    for i in range(3):
        ex = random.choice(examples)
        print(f"\nExample {i+1}:")
        print(ex['prompt'] + ex['expected_completion'])

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    generate_dataset(num_examples=5000, output_file="counting_dataset.jsonl")
