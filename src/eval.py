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
# CONFIGURATION
# ============================================================

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
# EXPECTED ANSWERS (hardcoded per test file)
# ============================================================

EXPECTED_ANSWERS = {
    "reasoning/test_1.txt": "3",
    "reasoning/test_2.txt": "0.05",
}

MESSY_KEYWORDS = ["um", "uh", "maybe", "i think", "probably", "perhaps", "like", "stuff"]

# ============================================================
# SCORING: Structured Output (0-4)
# ============================================================

def score_structured_output(output: str) -> int:
    """
    4 = valid JSON, correct keys, no extra text
    3 = valid JSON, minor issues (extra fields OR minor formatting)
    2 = JSON present but malformed OR extra text outside JSON
    1 = attempted but incorrect structure
    0 = no JSON at all
    """
    output = output.strip()

    # Try to extract first JSON object
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if not match:
        return 0

    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return 0  # Malformed JSON

    # Check for extra text before/after JSON
    before = output[:match.start()].strip()
    after = output[match.end():].strip()
    has_extras = bool(before or after)

    # Grade based on structure
    if not has_extras and isinstance(data, dict):
        return 4
    elif has_extras and isinstance(data, dict):
        return 2
    elif isinstance(data, dict):
        return 3
    else:
        return 1

# ============================================================
# SCORING: Reasoning (0-4)
# ============================================================

def score_reasoning(output: str, test_name: str, capability: str) -> int:
    """
    4 = correct answer (clean)
    3 = correct answer but messy explanation
    2 = partially correct logic
    1 = wrong but attempted
    0 = nonsense/empty/error
    """
    output_lower = output.lower().strip()
    if len(output) > 2000 or not output.strip():
        return 0

    key = f"{capability}/{test_name}.txt"
    expected = EXPECTED_ANSWERS.get(key)

    if expected is None:
        # No hardcoded answer — fall back to heuristic
        if any(w in output_lower for w in ["step", "because", "reason", "therefore"]):
            return 2
        return 1 if output.strip() else 0

    # Try numeric comparison
    try:
        expected_num = float(expected)
        nums = re.findall(r'[-+]?\d*\.?\d+', output)
        for n in nums:
            if abs(float(n) - expected_num) < 0.01:
                # Correct number — check if explanation includes clean markers
                if any(clean in output_lower for clean in ["therefore", "so", "answer", "result", "correct"]):
                    # Check for messy hedging words
                    if any(messy in output_lower for messy in MESSY_KEYWORDS):
                        return 3
                return 4
        # Numbers present but none match
        return 2 if nums else 1
    except (ValueError, TypeError):
        # Non-numeric expected answer — string match
        if expected.lower() in output_lower:
            return 4
        return 1 if output.strip() else 0

# ============================================================
# SCORING: Consistency across re-runs (0-4)
# ============================================================

def score_consistency(run_outputs: list[str]) -> int:
    """
    4 = all outputs identical
    3 = minor wording differences (same meaning)
    2 = same structure, different phrasing
    1 = significantly different
    0 = contradictory answers
    """
    if not run_outputs:
        return 0

    # Normalize: lowercase, strip whitespace, collapse internal spaces
    normalized = []
    for out in run_outputs:
        norm = re.sub(r'\s+', ' ', out.lower().strip())
        normalized.append(norm)

    first = normalized[0]

    # Check exact match across all
    if all(n == first for n in normalized):
        return 4

    # Count how many match exactly
    exact_matches = sum(1 for n in normalized if n == first)
    if exact_matches >= len(normalized) * 0.8:
        return 3  # Minor variations

    # Check for length similarity + keyword overlap
    lengths = [len(n) for n in normalized]
    avg_len = sum(lengths) / len(lengths)
    length_consistent = all(abs(l - avg_len) < 20 for l in lengths)

    if length_consistent:
        return 2  # Same structure, different phrasing

    # Significant variation
    return 1

# ============================================================
# CORE: Ollama execution
# ============================================================

def run_single_prompt(model_config, prompt):
    """Execute one prompt via Ollama. Returns (output_text, latency_seconds)."""
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
# RE-RUN: Multiple independent executions
# ============================================================

def run_with_reruns(model_config, prompt, num_runs=RUNS_PER_TEST):
    """Execute the same prompt N times, returning list of {output, latency}."""
    runs = []
    for run_num in range(num_runs):
        output, latency = run_single_prompt(model_config, prompt)
        print(f"      [run {run_num + 1}/{num_runs}] latency={latency:.2f}s", flush=True)
        runs.append({"output": output, "latency": latency})
    return runs

# ============================================================
# SELF-REVIEW: Second-pass correction
# ============================================================

def run_self_review(model_config, original_output):
    """Send review prompt for second-pass correction. Returns (reviewed_output, latency)."""
    review_prompt = REVIEW_PROMPT_TEMPLATE.format(original_output=original_output)
    reviewed, latency = run_single_prompt(model_config, review_prompt)
    return reviewed, latency

# ============================================================
# CHECK-TWICE: Verification prompt modifier
# ============================================================

def apply_check_twice(prompt, enabled=CHECK_TWICE):
    """If enabled, append verification instructions to the prompt."""
    if enabled:
        return prompt + CHECK_TWICE_PROMPT_SUFFIX
    return prompt

# ============================================================
# TEST LOADING: .txt files per capability
# ============================================================

def load_tests_from_txt(capability):
    """
    Load .txt test prompts from config/tests/<capability>/
    Returns list of {"name": <stem>, "prompt": <contents>}
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
# MAIN: Evaluation with scoring
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation harness (re-run + self-review + check-twice + scoring)"
    )
    parser.add_argument(
        "--capability", "-c",
        choices=["structured_output", "reasoning"],
        help="Which capability to evaluate"
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

    num_runs = args.runs
    check_twice = args.check_twice

    config = load_config()
    models = config["models"]

    # Filter models
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
    # Per-model execution + scoring
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
            final_prompt = apply_check_twice(prompt, check_twice)

            print(f"  [{test_name}] ", end="", flush=True)

            test_result_entry = {
                "capability": capability,
                "test": test_name,
                "runs": [],
                "consistency_score": 0,
                "avg_initial_score": 0.0,
                "avg_reviewed_score": 0.0,
                "delta": 0.0
            }

            initial_scores = []
            reviewed_scores = []
            initial_outputs_for_consistency = []

            for run_idx in range(num_runs):
                # Stage 1: Initial
                initial_output, initial_latency = run_single_prompt(model_cfg, final_prompt)
                model_total_time += initial_latency

                # Score initial
                if capability == "structured_output":
                    initial_score = score_structured_output(initial_output)
                else:  # reasoning
                    initial_score = score_reasoning(initial_output, test_name, capability)
                initial_scores.append(initial_score)
                initial_outputs_for_consistency.append(initial_output)

                # Stage 2: Self-review
                reviewed_output, review_latency = run_self_review(model_cfg, initial_output)
                model_total_time += review_latency

                # Score reviewed
                if capability == "structured_output":
                    reviewed_score = score_structured_output(reviewed_output)
                else:
                    reviewed_score = score_reasoning(reviewed_output, test_name, capability)
                reviewed_scores.append(reviewed_score)

                run_data = {
                    "initial": initial_output,
                    "reviewed": reviewed_output,
                    "scores": {
                        "initial": initial_score,
                        "reviewed": reviewed_score
                    },
                    "initial_latency": round(initial_latency, 3),
                    "review_latency": round(review_latency, 3),
                    "total_latency": round(initial_latency + review_latency, 3)
                }
                test_result_entry["runs"].append(run_data)

            # Per-test aggregates
            test_result_entry["avg_initial_score"] = round(sum(initial_scores) / len(initial_scores), 2)
            test_result_entry["avg_reviewed_score"] = round(sum(reviewed_scores) / len(reviewed_scores), 2)
            test_result_entry["delta"] = round(test_result_entry["avg_reviewed_score"] - test_result_entry["avg_initial_score"], 2)

            # Consistency across all initial outputs
            test_result_entry["consistency_score"] = score_consistency(initial_outputs_for_consistency)

            # Check-twice comparison (run once with vs without, if flag is on)
            if check_twice and num_runs >= 1:
                # Run once with plain prompt
                plain_output, _ = run_single_prompt(model_cfg, prompt)
                plain_score = score_structured_output(plain_output) if capability == "structured_output" else score_reasoning(plain_output, test_name, capability)
                ct_score = initial_scores[0]  # First run already used check-twice prompt
                test_result_entry["check_twice_delta"] = round(ct_score - plain_score, 2)

            model_results["results"].append(test_result_entry)
            print("done")

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
            f.write(f"Model:           {model_name}\n")
            f.write(f"Capability:      {capability}\n")
            f.write(f"Timestamp:       {run_timestamp}\n")
            f.write(f"Run ID:          {run_id}\n")
            f.write(f"Tests:           {len(tests)}\n")
            f.write(f"Runs per test:   {num_runs}\n")
            f.write(f"Check-twice:     {check_twice}\n")
            f.write(f"Total time:      {model_total_time:.2f} s\n")
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("PER-TEST ANALYSIS\n")
            f.write("=" * 70 + "\n\n")

            for test_result in model_results["results"]:
                test_name = test_result["test"]
                f.write(f"Test: {test_name}\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Consistency:  {test_result['consistency_score']}/4\n")
                f.write(f"  Avg initial:  {test_result['avg_initial_score']:.2f}/4\n")
                f.write(f"  Avg reviewed: {test_result['avg_reviewed_score']:.2f}/4\n")
                f.write(f"  Delta:        {test_result['delta']:+.2f}\n")
                if "check_twice_delta" in test_result:
                    f.write(f"  CT delta:     {test_result['check_twice_delta']:+.2f}\n")
                f.write("\n")

                for i, run in enumerate(test_result["runs"], 1):
                    f.write(f"  Run {i}:\n")
                    f.write(f"    Initial  ({run['initial_latency']}s):  {run['initial'][:100]}\n")
                    f.write(f"    Reviewed ({run['review_latency']}s): {run['reviewed'][:100]}\n")
                    f.write(f"    Scores:  init={run['scores']['initial']}, rev={run['scores']['reviewed']}\n\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF SUMMARY\n")

        print(f"  → saved: {out_path}")
        print(f"  → summary: {summary_path}")

    print(f"\nAll done. Results in results/ directory.")

if __name__ == "__main__":
    main()
