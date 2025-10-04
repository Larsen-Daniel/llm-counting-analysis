import json
import boto3
from typing import List, Dict
import argparse
from tqdm import tqdm
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataset(file_path: str) -> List[Dict]:
    """Load the dataset from JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def extract_answer(text: str) -> int | None:
    """
    Extract numerical answer from model output.
    Try multiple patterns in order of preference:
    1. Last number in parentheses: (N)
    2. First number on first line (for Mistral/others)
    """
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

def benchmark_bedrock_model(
    model_id: str,
    dataset: List[Dict],
    max_examples: int = None,
    region: str = "us-east-1",
    max_retries: int = 3
) -> Dict:
    """
    Benchmark a Bedrock model on the counting task.

    Args:
        model_id: Bedrock model ID (e.g., meta.llama3-1-8b-instruct-v1:0)
        dataset: List of examples
        max_examples: Maximum examples to evaluate
        region: AWS region
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_id}")
    print(f"Region: {region}")
    print(f"{'='*80}\n")

    # Initialize Bedrock client
    bedrock = boto3.client('bedrock-runtime', region_name=region)

    # Prepare dataset
    if max_examples:
        dataset = dataset[:max_examples]

    results = {
        "model_id": model_id,
        "total_examples": len(dataset),
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
        "numerical_errors": 0,
        "api_errors": 0,
        "predictions": []
    }

    print(f"Evaluating on {len(dataset)} examples...")

    for example in tqdm(dataset):
        prompt = example["prompt"]
        true_answer = example["answer"]

        # Retry logic
        generated_text = ""
        for attempt in range(max_retries):
            try:
                # Prepare request body based on model family
                if "llama" in model_id.lower():
                    body = json.dumps({
                        "prompt": prompt,
                        "max_gen_len": 10,
                        "temperature": 0.0,
                        "top_p": 1.0
                    })
                elif "claude" in model_id.lower() or "anthropic" in model_id.lower():
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 10,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}]
                    })
                elif "mistral" in model_id.lower():
                    body = json.dumps({
                        "prompt": prompt,
                        "max_tokens": 10,
                        "temperature": 0.0,
                        "top_p": 1.0
                    })
                else:
                    # Generic format
                    body = json.dumps({
                        "prompt": prompt,
                        "max_tokens": 10,
                        "temperature": 0.0
                    })

                # Call Bedrock
                response = bedrock.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )

                # Parse response
                response_body = json.loads(response['body'].read())

                # Extract generated text based on model family
                if "llama" in model_id.lower():
                    generated_text = response_body.get('generation', '')
                elif "claude" in model_id.lower() or "anthropic" in model_id.lower():
                    content = response_body.get('content', [{}])
                    generated_text = content[0].get('text', '') if content else ''
                elif "mistral" in model_id.lower():
                    outputs = response_body.get('outputs', [{}])
                    generated_text = outputs[0].get('text', '') if outputs else ''
                else:
                    # Try common response formats
                    generated_text = (
                        response_body.get('generation', '') or
                        response_body.get('completion', '') or
                        response_body.get('text', '') or
                        str(response_body)
                    )

                break

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    print(f"\nAPI error after {max_retries} attempts: {e}")
                    generated_text = ""
                    results["api_errors"] += 1

        # Extract answer
        predicted_answer = extract_answer(generated_text)

        # Check correctness
        if predicted_answer is None:
            results["parse_errors"] += 1
            is_correct = False
        else:
            is_correct = predicted_answer == true_answer
            if is_correct:
                results["correct"] += 1
            else:
                results["numerical_errors"] += 1
                results["incorrect"] += 1

        # Store prediction
        results["predictions"].append({
            "prompt": prompt,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "correct": is_correct
        })

    # Calculate metrics
    results["accuracy"] = results["correct"] / results["total_examples"]
    results["parse_error_rate"] = results["parse_errors"] / results["total_examples"]
    results["numerical_error_rate"] = results["numerical_errors"] / results["total_examples"]
    results["api_error_rate"] = results["api_errors"] / results["total_examples"]

    print(f"\nResults for {model_id}:")
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total_examples']})")
    print(f"  Parse Errors: {results['parse_error_rate']:.2%}")
    print(f"  Numerical Errors: {results['numerical_error_rate']:.2%}")
    print(f"  API Errors: {results['api_error_rate']:.2%}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark AWS Bedrock models on counting task")
    parser.add_argument("--dataset", type=str, default="counting_dataset.jsonl", help="Path to dataset")
    parser.add_argument("--output", type=str, default="benchmark_results_bedrock.json", help="Output file")
    parser.add_argument("--max_examples", type=int, default=1000, help="Max examples per model")
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs to benchmark")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")

    args = parser.parse_args()

    # Default models - our chosen 4 for benchmarking
    if args.models is None:
        models = [
            "us.meta.llama3-1-8b-instruct-v1:0",         # 8B
            "us.meta.llama3-1-70b-instruct-v1:0",        # 70B
            "us.anthropic.claude-3-5-haiku-20241022-v1:0", # Haiku 3.5
            "mistral.mistral-7b-instruct-v0:2",          # Mistral 7B
        ]
    else:
        models = args.models

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")

    # Load existing results if available
    all_results = []
    completed_models = set()
    if os.path.exists(args.output):
        print(f"\nLoading existing results from {args.output}...")
        with open(args.output, 'r') as f:
            all_results = json.load(f)
        completed_models = {r['model_id'] for r in all_results}
        print(f"Found results for {len(completed_models)} models: {list(completed_models)}")

    # Benchmark each model
    for model_id in models:
        if model_id in completed_models:
            print(f"\n{'='*80}")
            print(f"Skipping {model_id} - already completed")
            print(f"{'='*80}\n")
            continue
        try:
            results = benchmark_bedrock_model(
                model_id,
                dataset,
                max_examples=args.max_examples,
                region=args.region
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create visualization
    plot_file = args.output.replace('.json', '.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    model_names = [r["model_id"].replace("meta.llama", "Llama").replace("mistral.mistral", "Mistral").replace("anthropic.claude-3-haiku", "Haiku")[:30] for r in all_results]
    accuracies = [r["accuracy"] * 100 for r in all_results]
    parse_errors = [r["parse_error_rate"] * 100 for r in all_results]

    # Plot 1: Accuracy
    bars1 = ax1.barh(model_names, accuracies, color='steelblue')
    ax1.set_xlabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Model Accuracy on Counting Task", fontsize=14)
    ax1.set_xlim(0, 100)
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        ax1.text(acc + 1, i, f"{acc:.1f}%", va='center', fontsize=10)

    # Plot 2: Parse error rate
    bars2 = ax2.barh(model_names, parse_errors, color='coral')
    ax2.set_xlabel("Parse Error Rate (%)", fontsize=12)
    ax2.set_title("Models Failing to Output (N) Format", fontsize=14)
    ax2.set_xlim(0, 100)
    for i, (bar, err) in enumerate(zip(bars2, parse_errors)):
        ax2.text(err + 1, i, f"{err:.1f}%", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_file}")

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")
    print(f"{'Model':<45} {'Accuracy':<12} {'Parse Err':<12} {'Num Err':<12} {'API Err'}")
    print(f"{'-'*100}")
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model_id']:<45} {result['accuracy']:<12.2%} {result['parse_error_rate']:<12.2%} {result['numerical_error_rate']:<12.2%} {result['api_error_rate']:.2%}")

    print(f"\nResults saved to {args.output}")
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    main()
