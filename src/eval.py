import subprocess
import json
import re
import time
import os
import sys
import glob
import argparse
from datetime import datetime, timezone

# ============================================================
# NEW FEATURES: Re-run, Self-review, Check-twice
# ============================================================

# Configuration constants
RUNS_PER_TEST = 3                    # Number of execution attempts per test
CHECK_TWICE = False                  # Set True to enable verification prompt
REVIEW_PROMPT_TEMPLATE = """You previously answered:

{original_output}

Now:
- Check if the answer is correct
- Check if instructions were followed exactly
- Fix any mistakes

Return the final corrected answer only."""

CHECK_TWICE_PROMPT_SUFFIX = """
Before answering:
- Solve the problem
- Check your answer
- Check again for mistakes

Return only the final answer."""

# ============================================================
# EXISTING: Core Ollama execution (unchanged)
# ============================================================

BASE_PREFIX = """You are a deterministic function.
Do not explain your reasoning.
Return only the requested format.
"""

def load_config():
    with open("config/config.json") as f:
        return json.load(f)

def discover_capabilities():
    """Return list of capability directory names under config/tests/"""
    return sorted([d for d in os.listdir("config/tests") if os.path.isdir(os.path.join("config/tests", d))])

# ============================================================
# NEW: Load .txt test prompts (structured_output, reasoning only)
# ============================================================

def load_tests_from_txt(capability):
    """
    Load tests from .txt files in config/tests/<capability>/
    Each file contains a single prompt.
    Returns list of dicts: {"name": <stem>, "prompt": <contents>}
    """
    pattern = f"config/tests/{capability}/*.txt"
    files = sorted(glob.glob(pattern))
    tests = []
    for f in files:
        with open(f) as fh:
            prompt = fh.read().strip()
        name = os.path.splitext(os.path.basename(f))[0]
        tests.append({"name": name, "prompt": prompt})
    return tests

# ============================================================
# NEW: Single prompt execution
# ============================================================

def run_single_prompt(model_config, prompt):
    """
    Execute one prompt via Ollama.
    Returns (output_text, latency_seconds).
    """
    model = model_config["name"]
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout after 300s", 300.0
    latency = time.time() - start
    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}", latency
    return result.stdout.strip(), latency

# ============================================================
# NEW: Re-run logic — multiple independent executions
# ============================================================

def run_with_reruns(model_config, prompt, num_runs=RUNS_PER_TEST):
    """
    Execute the same prompt N times, returning all outputs.
    Returns list of (output, latency) tuples.
    """
    runs = []
    for run_num in range(num_runs):
        output, latency = run_single_prompt(model_config, prompt)
        print(f"      [run {run_num + 1}/{num_runs}] latency={latency:.2f}s")
        runs.append({"output": output, "latency": latency})
    return runs

# ============================================================
# NEW: Self-review pass
# ============================================================

def run_self_review(model_config, original_output):
    """
    Send the review prompt to the model for a second-pass correction.
    Returns (reviewed_output, latency).
    """
    review_prompt = REVIEW_PROMPT_TEMPLATE.format(original_output=original_output)
    reviewed, latency = run_single_prompt(model_config, review_prompt)
    return reviewed, latency

# ============================================================
# HELPER: Apply check-twice modifier to prompt
# ============================================================

def apply_check_twice(prompt, enabled=CHECK_TWICE):
    """
    If CHECK_TWICE is True, append verification instructions to the prompt.
    """
    if enabled:
        return prompt + CHECK_TWICE_PROMPT_SUFFIX
    return prompt

# ============================================================
# MAIN: Limited-scope evaluation (structured_output + reasoning)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation harness (re-run + self-review + check-twice modes)"
    )
    parser.add_argument(
        "--capability", "-c",
        choices=["structured_output", "reasoning"],
        help="Which capability to evaluate (only these two supported)"
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        help="Run only matching model(s)"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=RUNS_PER_TEST,
        help=f"Runs per test (default: {RUNS_PER_TEST})"
    )
    parser.add_argument(
        "--check-twice",
        action="store_true",
        default=CHECK_TWICE,
        help="Enable check-twice verification prompt"
    )
    args = parser.parse_args()

    # Override globals from CLI
    num_runs = args.runs
    check_twice = args.check_twice

    config = load_config()
    models = config["models"]

    # Filter models if requested
    if args.model:
        matched = []
        for model_filter in args.model:
            for m in models:
                if model_filter.lower() in m["name"].lower() and m not in matched:
                    matched.append(m)
        models = matched
        if not models:
            print(f"Error: No models match filters: {args.model}")
            sys.exit(1)

    capability = args.capability
    if not capability:
        print("Error: --capability is required (structured_output or reasoning)")
        sys.exit(1)

    # Load tests from .txt files
    tests = load_tests_from_txt(capability)
    if not tests:
        print(f"Error: No tests found in config/tests/{capability}/")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n=== Evaluation: {capability} ===")
    print(f"Models: {[m['name'] for m in models]}")
    print(f"Tests: {len(tests)}, Runs per test: {num_runs}, Check-twice: {check_twice}")
    print()

    # ============================================================
    # Per-model result files
    # ============================================================

    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"\n--- {model_name} ---")

        model_results = {
            "model": model_name,
            "capability": capability,
            "run_metadata": {
                "timestamp": run_timestamp,
                "run_id": run_id,
                "runs_per_test": num_runs,
                "check_twice": check_twice
            },
            "results": []
        }

        model_total_time = 0.0

        for test in tests:
            test_name = test["name"]
            prompt = test["prompt"]

            # Apply check-twice transformation if enabled
            final_prompt = apply_check_twice(prompt, check_twice)

            print(f"  [{test_name}] ", end="", flush=True)

            test_result_entry = {
                "capability": capability,
                "test": test_name,
                "runs": []
            }

            for run_idx in range(num_runs):
                # --- Stage 1: Initial execution ---
                initial_output, initial_latency = run_single_prompt(model_cfg, final_prompt)
                model_total_time += initial_latency

                # --- Stage 2: Self-review pass (second call) ---
                reviewed_output, review_latency = run_self_review(model_cfg, initial_output)
                model_total_time += review_latency

                run_data = {
                    "initial": initial_output,
                    "reviewed": reviewed_output,
                    "initial_latency": round(initial_latency, 3),
                    "review_latency": round(review_latency, 3),
                    "total_latency": round(initial_latency + review_latency, 3)
                }
                test_result_entry["runs"].append(run_data)

            model_results["results"].append(test_result_entry)
            print("done")

        # ============================================================
        # Write per-model JSON results file
        # ============================================================

        # Per-model timing stats
        model_results["total_time_seconds"] = round(model_total_time, 2)

        out_filename = f"{model_name.replace(':', '_')}_{capability}_{run_id}.json"
        out_path = os.path.join("results", out_filename)
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)

        # Write human-readable summary
        summary_filename = f"{model_name.replace(':', '_')}_{capability}_{run_id}.txt"
        summary_path = os.path.join("results", summary_filename)
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model:        {model_name}\n")
            f.write(f"Capability:   {capability}\n")
            f.write(f"Timestamp:    {run_timestamp}\n")
            f.write(f"Run ID:       {run_id}\n")
            f.write(f"Tests:        {len(tests)}\n")
            f.write(f"Runs per test: {num_runs}\n")
            f.write(f"Check-twice:  {check_twice}\n")
            f.write(f"Total time:   {model_total_time:.2f} s\n")
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("PER-TEST RESULTS\n")
            f.write("=" * 70 + "\n\n")

            for test_result in model_results["results"]:
                test_name = test_result["test"]
                f.write(f"Test: {test_name}\n")
                f.write("-" * 50 + "\n")
                for i, run in enumerate(test_result["runs"], 1):
                    f.write(f"  Run {i}:\n")
                    f.write(f"    Initial (latency {run['initial_latency']}s):  {run['initial'][:120]}\n")
                    f.write(f"    Reviewed (latency {run['review_latency']}s): {run['reviewed'][:120]}\n")
                    f.write(f"    Total latency: {run['total_latency']}s\n\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF SUMMARY\n")

        print(f"  → saved: {out_path}")
        print(f"  → summary: {summary_path}")

    print(f"\nAll done. Results in results/ directory.")

if __name__ == "__main__":
    main()
