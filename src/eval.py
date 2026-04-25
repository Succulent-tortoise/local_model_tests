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
OVERCONFIDENCE_LENGTH_THRESHOLD = 300  # chars; long + wrong = overconfident

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

# Expected answers per reasoning test (hardcoded)
EXPECTED_ANSWERS = {
    "reasoning/test_1.txt": "3",
    "reasoning/test_2.txt": "0.05",
}

# Keywords for reasoning quality heuristics
STEP_INDICATORS = ["step", "therefore", "so", "thus", "because", "reason", "calc", "equation", "solve"]
MESSY_KEYWORDS = ["um", "uh", "maybe", "i think", "probably", "perhaps", "like", "stuff", "approximately"]
CONTRADICTION_KEYWORDS = ["however", "actually", "wait", "no", "wrong", "incorrect", "contradict"]

# ============================================================
# UTILS: Clean output, extract answer
# ============================================================

def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from Ollama output."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def extract_final_answer(output: str) -> str:
    """
    Extract the final numeric or short answer from output.
    For reasoning: try to find a number or short phrase at the end.
    """
    clean = strip_ansi(output).strip()
    # Try to find a standalone number (integer or decimal)
    numbers = re.findall(r'[-+]?\d*\.?\d+', clean)
    if numbers:
        # Return the last number found (often the final answer)
        return numbers[-1].strip()
    # Fallback: first short phrase after common answer markers
    for marker in ["answer:", "result:", "therefore:", "so:"]:
        if marker in clean.lower():
            after = clean.lower().split(marker, 1)[1].strip()
            # Take up to first sentence or 30 chars
            after = after.split('.')[0][:30]
            return after.strip()
    # Last resort: first 30 chars of cleaned output
    return clean[:30].strip()

def is_contradictory(a: str, b: str) -> bool:
    """
    Determine if two answers are contradictory.
    Simple heuristic: both numeric and differ by > 50% relative, or opposite keywords.
    """
    a_clean = strip_ansi(a).lower().strip()
    b_clean = strip_ansi(b).lower().strip()

    # Try numeric comparison
    nums_a = re.findall(r'[-+]?\d*\.?\d+', a_clean)
    nums_b = re.findall(r'[-+]?\d*\.?\d+', b_clean)
    if nums_a and nums_b:
        try:
            val_a = float(nums_a[-1])
            val_b = float(nums_b[-1])
            if val_a == 0 and val_b == 0:
                return False
            # Relative difference > 50% considered contradictory
            rel_diff = abs(val_a - val_b) / max(abs(val_a), abs(val_b), 1)
            if rel_diff > 0.5:
                return True
        except (ValueError, TypeError):
            pass

    # Keyword opposition (simple)
    opposites = [("yes", "no"), ("safe", "unsafe"), ("correct", "wrong"), ("true", "false")]
    for a_word, b_word in opposites:
        if a_word in a_clean and b_word in b_clean:
            return True

    # Contradiction keywords in one but not other
    if any(k in a_clean for k in CONTRADICTION_KEYWORDS) and not any(k in b_clean for k in CONTRADICTION_KEYWORDS):
        return True

    return False

# ============================================================
# SCORING: Structured Output (0-4) — single dimension
# ============================================================

def score_structured_output(output: str) -> int:
    """
    4 = valid JSON dict, no extra text
    3 = valid JSON dict, minor issues (extra fields, formatting)
    2 = JSON with extra text before/after, or malformed but recognizable
    1 = attempted JSON but invalid
    0 = no JSON at all
    """
    output = strip_ansi(output).strip()

    # Try to extract first JSON object
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if not match:
        return 0

    json_str = match.group()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return 1  # Malformed but present

    before = output[:match.start()].strip()
    after = output[match.end():].strip()
    has_extras = bool(before or after)

    if not has_extras and isinstance(data, dict):
        return 4
    elif isinstance(data, dict):
        return 3 if not has_extras else 2
    else:
        return 1

# ============================================================
# SCORING: Reasoning — split into answer_correctness & reasoning_quality
# ============================================================

def score_reasoning_answer(output: str, test_name: str, capability: str) -> int:
    """
    4 = correct final answer (numeric or semantic match)
    3 = correct answer but messy/unclear presentation
    2 = partially correct (close numeric or partial logic)
    1 = incorrect but attempted
    0 = empty/error/nonsense
    """
    output_clean = strip_ansi(output).strip()
    if len(output_clean) > 2000 or not output_clean:
        return 0

    key = f"{capability}/{test_name}.txt"
    expected = EXPECTED_ANSWERS.get(key)

    if expected is None:
        # No ground truth — heuristic: presence of reasoning + any answer
        if any(ind in output_clean.lower() for ind in STEP_INDICATORS):
            return 2 if re.search(r'\d', output_clean) else 1
        return 0

    # Numeric expected?
    try:
        expected_num = float(expected)
        nums = re.findall(r'[-+]?\d*\.?\d+', output_clean)
        if not nums:
            return 1 if output_clean else 0
        # Find closest number
        closeness = [abs(float(n) - expected_num) for n in nums]
        min_diff = min(closeness)
        if min_diff < 0.01:
            return 4
        elif min_diff / max(abs(expected_num), 1) < 0.1:
            return 2
        else:
            return 1
    except (ValueError, TypeError):
        # String match
        if expected.lower() in output_clean.lower():
            return 4
        else:
            return 1 if output_clean else 0

def score_reasoning_quality(output: str) -> int:
    """
    Evaluate reasoning clarity and logical flow.
    4 = clear step-by-step, logical, no contradictions
    3 = mostly clear, minor messiness or slight inconsistency
    2 = some reasoning but weak/incomplete
    1 = very weak reasoning or confused
    0 = no reasoning (just answer) or nonsense
    """
    output_clean = strip_ansi(output).lower().strip()
    if not output_clean:
        return 0

    # Count step indicators
    step_count = sum(1 for ind in STEP_INDICATORS if ind in output_clean)

    # Penalize messy keywords
    messy_penalty = sum(1 for kw in MESSY_KEYWORDS if kw in output_clean)

    # Check for contradictions in the same output
    has_contradiction = any(kw in output_clean for kw in CONTRADICTION_KEYWORDS)

    if step_count >= 3 and not has_contradiction and messy_penalty == 0:
        return 4
    elif step_count >= 2 and messy_penalty <= 1 and not has_contradiction:
        return 3
    elif step_count >= 1:
        return 2
    elif len(output_clean) > 50:  # Long but no clear steps
        return 1
    else:
        return 0

# ============================================================
# SCORING: Consistency (improved)
# ============================================================

def score_consistency(run_outputs: list[str]) -> int:
    """
    Improved consistency scoring:
    4 = all outputs identical (after normalization)
    3 = minor differences (length diff < 10%, same answer)
    2 = same answer but different wording
    1 = significantly different answers
    0 = contradictory
    """
    if not run_outputs:
        return 0

    cleaned = [strip_ansi(out).strip() for out in run_outputs]
    normalized = [' '.join(c.lower().split()) for c in cleaned]

    first = normalized[0]

    # Exact match all
    if all(n == first for n in normalized):
        return 4

    # Extract final answers and compare
    answers = [extract_final_answer(out) for out in cleaned]
    ans_first = answers[0]
    all_same_ans = all(a == ans_first for a in answers)

    if all_same_ans:
        # Same answer, check length similarity
        lengths = [len(n) for n in normalized]
        avg_len = sum(lengths) / len(lengths)
        max_dev = max(abs(l - avg_len) / avg_len for l in lengths) if avg_len > 0 else 0
        if max_dev < 0.1:
            return 3
        else:
            return 2
    else:
        # Different answers — check for contradiction
        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                if is_contradictory(cleaned[i], cleaned[j]):
                    return 0
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
# MAIN: Evaluation with enhanced scoring
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation harness (re-run + self-review + check-twice + enhanced scoring)"
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
                "stability": "low",
                # Averages for structured_output: scores are 0-4 single; for reasoning: split dimensions
                "avg_initial_score": 0.0,
                "avg_reviewed_score": 0.0,
                "delta": 0.0,
                # Reasoning-specific: answer correctness & reasoning quality
                "avg_initial_answer_correctness": None,
                "avg_reviewed_answer_correctness": None,
                "delta_answer": None,
                "avg_initial_reasoning_quality": None,
                "avg_reviewed_reasoning_quality": None,
                "delta_reasoning": None,
            }

            initial_scores = []
            reviewed_scores = []
            initial_answer_scores = []
            reviewed_answer_scores = []
            initial_quality_scores = []
            reviewed_quality_scores = []
            initial_outputs_for_consistency = []

            for run_idx in range(num_runs):
                # Stage 1: Initial
                initial_output, initial_latency = run_single_prompt(model_cfg, final_prompt)
                model_total_time += initial_latency

                # Score initial (capability-specific)
                if capability == "structured_output":
                    initial_score = score_structured_output(initial_output)
                else:  # reasoning
                    initial_answer = score_reasoning_answer(initial_output, test_name, capability)
                    initial_quality = score_reasoning_quality(initial_output)
                    initial_score = (initial_answer + initial_quality) / 2  # composite for compatibility
                    initial_answer_scores.append(initial_answer)
                    initial_quality_scores.append(initial_quality)
                initial_scores.append(initial_score)
                initial_outputs_for_consistency.append(initial_output)

                # Detect overconfidence (only if reasoning and we have answer score)
                overconfidence = False
                if capability == "reasoning" and initial_answer <= 1 and len(initial_output) > OVERCONFIDENCE_LENGTH_THRESHOLD:
                    overconfidence = True

                # Stage 2: Self-review
                reviewed_output, review_latency = run_self_review(model_cfg, initial_output)
                model_total_time += review_latency

                # Score reviewed
                if capability == "structured_output":
                    reviewed_score = score_structured_output(reviewed_output)
                else:
                    reviewed_answer = score_reasoning_answer(reviewed_output, test_name, capability)
                    reviewed_quality = score_reasoning_quality(reviewed_output)
                    reviewed_score = (reviewed_answer + reviewed_quality) / 2
                    reviewed_answer_scores.append(reviewed_answer)
                    reviewed_quality_scores.append(reviewed_quality)
                reviewed_scores.append(reviewed_score)

                # Contradiction detection (compare extracted final answers)
                initial_ans_extracted = extract_final_answer(initial_output)
                reviewed_ans_extracted = extract_final_answer(reviewed_output)
                contradicts = is_contradictory(initial_output, reviewed_output)

                if capability == "structured_output":
                    # Structured output: single score maps to both dimensions for uniform schema
                    init_scores = {"answer_correctness": initial_score, "reasoning_quality": initial_score}
                    rev_scores = {"answer_correctness": reviewed_score, "reasoning_quality": reviewed_score}
                else:
                    init_scores = {"answer_correctness": initial_answer, "reasoning_quality": initial_quality}
                    rev_scores = {"answer_correctness": reviewed_answer, "reasoning_quality": reviewed_quality}

                run_data = {
                    "initial": initial_output,
                    "reviewed": reviewed_output,
                    "scores": {"initial": init_scores, "reviewed": rev_scores},
                    "initial_latency": round(initial_latency, 3),
                    "review_latency": round(review_latency, 3),
                    "total_latency": round(initial_latency + review_latency, 3),
                    "contradiction": contradicts
                }
                if overconfidence:
                    run_data["overconfidence"] = True
                test_result_entry["runs"].append(run_data)

            # Per-test aggregates
            test_result_entry["avg_initial_score"] = round(sum(initial_scores) / len(initial_scores), 2)
            test_result_entry["avg_reviewed_score"] = round(sum(reviewed_scores) / len(reviewed_scores), 2)
            test_result_entry["delta"] = round(test_result_entry["avg_reviewed_score"] - test_result_entry["avg_initial_score"], 2)

            # Consistency
            test_result_entry["consistency_score"] = score_consistency(initial_outputs_for_consistency)

            # Stability flag
            if test_result_entry["consistency_score"] >= 3:
                test_result_entry["stability"] = "high"
            elif test_result_entry["consistency_score"] == 2:
                test_result_entry["stability"] = "medium"
            else:
                test_result_entry["stability"] = "low"

            # Reasoning-specific: answer & quality breakdown
            if capability == "reasoning" and initial_answer_scores:
                test_result_entry["avg_initial_answer_correctness"] = round(sum(initial_answer_scores) / len(initial_answer_scores), 2)
                test_result_entry["avg_reviewed_answer_correctness"] = round(sum(reviewed_answer_scores) / len(reviewed_answer_scores), 2)
                test_result_entry["delta_answer"] = round(test_result_entry["avg_reviewed_answer_correctness"] - test_result_entry["avg_initial_answer_correctness"], 2)
                test_result_entry["avg_initial_reasoning_quality"] = round(sum(initial_quality_scores) / len(initial_quality_scores), 2)
                test_result_entry["avg_reviewed_reasoning_quality"] = round(sum(reviewed_quality_scores) / len(reviewed_quality_scores), 2)
                test_result_entry["delta_reasoning"] = round(test_result_entry["avg_reviewed_reasoning_quality"] - test_result_entry["avg_initial_reasoning_quality"], 2)

            # Check-twice comparison
            if check_twice and num_runs >= 1:
                plain_output, _ = run_single_prompt(model_cfg, prompt)
                if capability == "structured_output":
                    plain_score = score_structured_output(plain_output)
                    ct_score = initial_scores[0]
                else:
                    plain_ans = score_reasoning_answer(plain_output, test_name, capability)
                    plain_qual = score_reasoning_quality(plain_output)
                    plain_score = (plain_ans + plain_qual) / 2
                    ct_ans = initial_answer_scores[0] if initial_answer_scores else 0
                    ct_qual = initial_quality_scores[0] if initial_quality_scores else 0
                    ct_score = (ct_ans + ct_qual) / 2
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
                f.write(f"  Stability:    {test_result['stability']}\n")
                f.write(f"  Avg initial:  {test_result['avg_initial_score']:.2f}/4\n")
                f.write(f"  Avg reviewed: {test_result['avg_reviewed_score']:.2f}/4\n")
                f.write(f"  Delta:        {test_result['delta']:+.2f}\n")
                if capability == "reasoning":
                    f.write(f"  Answer correctness:  init={test_result.get('avg_initial_answer_correctness','?')}/4, rev={test_result.get('avg_reviewed_answer_correctness','?')}/4, Δ={test_result.get('delta_answer','?'):+.2f}\n")
                    f.write(f"  Reasoning quality:   init={test_result.get('avg_initial_reasoning_quality','?')}/4, rev={test_result.get('avg_reviewed_reasoning_quality','?')}/4, Δ={test_result.get('delta_reasoning','?'):+.2f}\n")
                if "check_twice_delta" in test_result:
                    f.write(f"  CT delta:     {test_result['check_twice_delta']:+.2f}\n")
                f.write("\n")

                for i, run in enumerate(test_result["runs"], 1):
                    # Strip ANSI for summary display
                    init_display = strip_ansi(run['initial'])[:100]
                    rev_display = strip_ansi(run['reviewed'])[:100]
                    f.write(f"  Run {i}:\n")
                    f.write(f"    Initial  ({run['initial_latency']}s):  {init_display}\n")
                    f.write(f"    Reviewed ({run['review_latency']}s): {rev_display}\n")
                    scores = run['scores']
                    if capability == "reasoning":
                        f.write(f"    Scores:  init=[ans={scores['initial']['answer_correctness']}, qual={scores['initial']['reasoning_quality']}], rev=[ans={scores['reviewed']['answer_correctness']}, qual={scores['reviewed']['reasoning_quality']}]\n")
                    else:
                        f.write(f"    Scores:  init={scores['initial']['answer_correctness']}/4, rev={scores['reviewed']['answer_correctness']}/4\n")
                    if run.get("contradiction"):
                        f.write(f"    ⚠️ Contradiction detected between initial and reviewed\n")
                    if run.get("overconfidence"):
                        f.write(f"    ⚠️ Overconfidence flag (long output, low score)\n")
                    f.write("\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF SUMMARY\n")

        print(f"  → saved: {out_path}")
        print(f"  → summary: {summary_path}")

    print(f"\nAll done. Results in results/ directory.")

if __name__ == "__main__":
    main()
