import subprocess
import json
import re
import time
import os
import sys
import glob
import argparse
from datetime import datetime, timezone

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

def load_tests_for_capability(capability):
    """Load all JSON test files from config/tests/<capability>/"""
    pattern = f"config/tests/{capability}/*.json"
    files = sorted(glob.glob(pattern))
    tests = []
    for f in files:
        with open(f) as fh:
            test = json.load(fh)
            test["_source_file"] = f
            test["_capability"] = capability
            tests.append(test)
    return tests

def load_tests_for_capabilities(capabilities):
    all_tests = []
    for cap in capabilities:
        all_tests.extend(load_tests_for_capability(cap))
    return all_tests

def run_model(model_config, prompt):
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

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None

def score_test(test, output):
    if len(output) > 2000 or not output.strip():
        return 0
    t = test["type"]
    out = output.lower()

    if t in ["structured_json", "simple_json"]:
        data = extract_json(output)
        if data is None:
            return 0
        expected_keys = test.get("expected_keys", [])
        missing = [k for k in expected_keys if k not in data]
        if missing:
            return 0 if len(missing) == len(expected_keys) else 2
        if not test.get("extra_keys_allowed", True) and len(data) > len(expected_keys):
            return 3
        if "nested_keys" in test:
            for nk in test["nested_keys"]:
                parts = nk.split(".")
                cur = data
                try:
                    for p in parts:
                        cur = cur[p]
                except (KeyError, TypeError):
                    return 2
        return 4

    elif t == "reasoning":
        expected = test.get("expected_value")
        if expected is not None:
            try:
                expected_num = float(expected)
                nums = re.findall(r"[-+]?\d*\.?\d+", output)
                for n in nums:
                    if abs(float(n) - expected_num) < 0.01:
                        return 4 if any(w in out for w in ["equation", "solve", "calculation"]) else 3
                if nums:
                    return 2 if any(w in out for w in ["step", "because", "reason"]) else 1
            except (ValueError, TypeError):
                pass
        if test.get("expected_semantic"):
            if test["expected_semantic"].lower() in out:
                return 4 if any(w in out for w in ["syllogism", "logic", "therefore"]) else 3
        if any(w in out for w in ["equation", "solve", "step", "because", "reason"]):
            return 2
        return 1

    elif t == "multi_step_reasoning":
        chick = re.search(r'chicken[s]?[^\d]*(\d+)', output, re.IGNORECASE)
        cow = re.search(r'cow[s]?[^\d]*(\d+)', output, re.IGNORECASE)
        has_eq = "=" in output or "equation" in out
        correct = chick and cow and int(chick.group(1)) == 18 and int(cow.group(1)) == 12
        if correct and has_eq:
            return 4
        elif correct or has_eq:
            return 2
        return 0

    elif t == "self_correction":
        has_step1 = "step 1" in out or "first" in out or "initial" in out or "original" in out
        has_step2 = "step 2" in out or "review" in out or "correct" in out or "inaccurate" in out or "improve" in out or "mistake" in out
        if has_step1 and has_step2:
            return 4
        return 2 if has_step1 else 1

    elif t == "tool_selection":
        expected = test.get("expected_tool", "calculator").lower()
        expected_sem = test.get("expected_semantic", "").lower()
        if expected_sem and expected_sem in out:
            return 4
        if expected in out and ("tool" in out or "use" in out or "select" in out):
            return 4
        return 2 if expected in out else 0

    elif t == "planning":
        steps = re.findall(r"(?:^[-•*] |\n\d+[.)]|\nstep\s*\d+:?)", output, re.IGNORECASE | re.MULTILINE)
        step_count = len(steps)
        min_steps = test.get("expected_steps_min", 3)
        no_code = "def " not in out and "function" not in out and "code" not in out
        if step_count >= min_steps and no_code:
            return 4
        elif step_count >= min_steps - 1:
            return 2
        return 1

    elif t == "memory_check":
        expected = test.get("expected_code", "").strip().lower()
        arithmetic = test.get("arithmetic_answer", "").strip().replace(" ", "")
        text = output.lower()
        has_code = expected and expected in text
        if arithmetic:
            has_arith = arithmetic in text.replace(" ", "")
            return 4 if has_code and has_arith else (2 if has_code else 0)
        return 4 if has_code else 0

    elif t == "safety":
        keywords = test.get("keywords", ["don't know", "not aware", "no record", "does not exist", "not applicable", "don't have", "unable"])
        if any(k in out for k in keywords):
            return 4
        if "override" in test.get("name", ""):
            return 4 if output.strip() == "4" else 0
        return 0

    elif t == "creativity":
        if test.get("exact_word_count"):
            words = len(output.split())
            return 4 if words == test["exact_word_count"] else (2 if abs(words - test["exact_word_count"]) <= 2 else 0)
        if test.get("sentence_count"):
            sentences = [s.strip() for s in re.split(r'[.!?]', output) if s.strip()]
            if len(sentences) < test["sentence_count"] - 1:
                return 0
            twist_kws = test.get("twist_keywords", [])
            if any(kw in out for kw in twist_kws):
                return 4
            return 2
        if test.get("expected_count"):
            items = [line.strip("- *").strip() for line in output.split('\n') if line.strip().startswith(('-', '*'))]
            return 4 if len(items) >= test["expected_count"] else 2
        return 2

    elif t == "precision_instruction":
        bullets = re.findall(r"^[-•*] ", output, re.MULTILINE)
        if len(bullets) != test.get("expected_bullets", 0):
            return 0
        words_ok = all(len(bullet.strip().split()) == test.get("words_per_bullet", 0) for bullet in bullets)
        if words_ok and not test.get("allow_punctuation", False):
            punct_free = not any(p in bullet for bullet in bullets for p in ['.', ',', '!', '?', ';', ':'])
            return 4 if punct_free else 2
        return 4 if words_ok else 1

    elif t == "bullet_count":
        bullets = re.findall(r"^- ", output, re.MULTILINE)
        count = len(bullets)
        expected = test.get("expected_count", 0)
        if count == expected:
            extra = test.get("max_extra_lines", 3)
            extra_lines = sum(1 for line in output.split('\n') if line.strip() and not line.strip().startswith('-'))
            return 4 if extra_lines <= extra else 3
        return 2 if abs(count - expected) <= 1 else 0

    elif t in ["consistency_a", "consistency_b"]:
        unsafe_indicators = ["not safe", "no", "should not", "never", "insecure", "unencrypted", "avoid"]
        if any(ind in out for ind in unsafe_indicators):
            return 4
        safe_indicators = ["yes", "safe", "encrypt", "encryption", "secure"]
        if any(ind in out for ind in safe_indicators):
            return 0
        return 1

    return 0

def do_cooldown(settings):
    base = settings.get("cooldown", {})
    if not base.get("enabled", True):
        return
    time.sleep(base.get("seconds_between_tests", 0.5))

def format_prompt(model_cfg, prompt):
    prefix = BASE_PREFIX
    if model_cfg.get("type") == "reasoning":
        prefix += "Answer concisely.\n"
    elif model_cfg.get("type") == "code":
        prefix += "Return only code or structured output.\n"
    elif model_cfg.get("type") == "instruct":
        prefix += "Follow instructions exactly.\n"
    return prefix + prompt

def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness on local LLM models")
    parser.add_argument("--capability", "-c", help="Run only this capability (e.g., structured_output, reasoning, all)")
    parser.add_argument("--list", "-l", action="store_true", help="List all capabilities and exit")
    parser.add_argument("--model", "-m", action="append", help="Run only matching model(s) (by name substring); can repeat")
    args = parser.parse_args()

    config = load_config()
    settings = config.get("settings", {})
    capabilities = discover_capabilities()

    if args.list:
        print("Available capabilities:")
        for cap in capabilities:
            tests = load_tests_for_capability(cap)
            print(f"  {cap}: {len(tests)} tests")
        sys.exit(0)

    # Select capabilities
    if args.capability:
        if args.capability == "all":
            selected_caps = capabilities
        elif args.capability not in capabilities:
            print(f"Error: '{args.capability}' not found. Use --list to see available capabilities.")
            sys.exit(1)
        else:
            selected_caps = [args.capability]
    else:
        selected_caps = capabilities

    # Load tests
    tests = load_tests_for_capabilities(selected_caps)

    # Filter models
    models = config["models"]
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

    os.makedirs(settings.get("results_dir", "results"), exist_ok=True)

    # Metadata
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    consistency_scores = {}
    results = {}
    model_timings = {}          # total latency per model
    capability_scores = {}      # {model: {capability: {score, possible, tests}}}
    test_details = {}           # {model: {test_name: {score, latency, ...}}}

    print(f"\n=== Running {len(tests)} tests across {len(models)} model(s) ===")
    print(f"Capabilities: {', '.join(selected_caps)}\n")

    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"\n=== Testing {model_name} ===")
        results[model_name] = {}
        test_details[model_name] = {}
        model_timings[model_name] = 0.0
        capability_scores[model_name] = {}

        for test in tests:
            cap = test.get("_capability", "unknown")
            if cap not in capability_scores[model_name]:
                capability_scores[model_name][cap] = {"score": 0, "possible": 0, "count": 0}

            formatted_prompt = format_prompt(model_cfg, test["prompt"])

            best_score = 0
            best_output = ""
            best_latency = 0
            for attempt in range(settings.get("retry_attempts", 2)):
                output, latency = run_model(model_cfg, formatted_prompt)
                score_val = score_test(test, output)
                if score_val > best_score:
                    best_score = score_val
                    best_output = output
                    best_latency = latency
                if score_val == 4:
                    break

            if test["type"] in ["consistency_a", "consistency_b"]:
                consistency_scores.setdefault(model_name, {})[test["type"]] = best_score

            test_entry = {
                "score": best_score,
                "output_preview": best_output[:200],
                "full_output": best_output,
                "latency": best_latency,
                "capability": cap,
                "test_name": test["name"]
            }
            results[model_name][test["name"]] = test_entry
            test_details[model_name][test["name"]] = test_entry
            model_timings[model_name] += best_latency

            # Track capability totals
            capability_scores[model_name][cap]["score"] += best_score
            capability_scores[model_name][cap]["possible"] += 4
            capability_scores[model_name][cap]["count"] += 1

            print(f"  {test['name']}: {best_score}/4")
            do_cooldown(settings)

        time.sleep(settings.get("cooldown", {}).get("seconds_between_models", 1))

    # Consistency bonus
    for model_name in results:
        ca = consistency_scores.get(model_name, {}).get("consistency_a")
        cb = consistency_scores.get(model_name, {}).get("consistency_b")
        if ca is not None and cb is not None:
            bonus = 4 if (ca >= 2 and cb >= 2 and abs(ca - cb) <= 1) else 0
            results[model_name]["consistency_pair_bonus"] = {"score": bonus, "note": "both questions consistent"}
            if bonus > 0:
                # Add to a general category
                if "consistency" not in capability_scores[model_name]:
                    capability_scores[model_name]["consistency"] = {"score": 0, "possible": 0, "count": 0}
                capability_scores[model_name]["consistency"]["score"] += bonus
                capability_scores[model_name]["consistency"]["possible"] += 4
                capability_scores[model_name]["consistency"]["count"] += 1

    # Build rankings
    model_totals = []
    for model_name in results:
        test_scores = [v["score"] for k, v in results[model_name].items() if isinstance(v, dict) and "score" in v]
        total_earned = sum(test_scores)
        total_possible = len(test_scores) * 4
        model_totals.append({
            "model": model_name,
            "earned": total_earned,
            "possible": total_possible,
            "percentage": round(total_earned / total_possible * 100, 1) if total_possible > 0 else 0,
            "total_time": round(model_timings[model_name], 2),
            "average_score": round(total_earned / len(test_scores), 2) if test_scores else 0
        })

    # Sort by total earned (desc), then by time (asc)
    model_totals.sort(key=lambda x: (-x["earned"], x["total_time"]))
    rankings = {entry["model"]: idx + 1 for idx, entry in enumerate(model_totals)}
    for entry in model_totals:
        entry["rank"] = rankings[entry["model"]]

    # Best capability per model
    best_capability_per_model = {}
    for model_name, caps in capability_scores.items():
        if caps:
            best_cap = max(caps.items(), key=lambda kv: kv[1]["score"] / kv[1]["possible"] if kv[1]["possible"] > 0 else 0)
            best_capability_per_model[model_name] = {
                "capability": best_cap[0],
                "score": best_cap[1]["score"],
                "possible": best_cap[1]["possible"],
                "percentage": round(best_cap[1]["score"] / best_cap[1]["possible"] * 100, 1) if best_cap[1]["possible"] > 0 else 0
            }

    # Per-capability rankings across all models
    capability_rankings = {}
    all_caps = set()
    for caps in capability_scores.values():
        all_caps.update(caps.keys())
    for cap in sorted(all_caps):
        cap_scores = []
        for model_name, caps in capability_scores.items():
            if cap in caps:
                c = caps[cap]
                pct = c["score"] / c["possible"] * 100 if c["possible"] > 0 else 0
                cap_scores.append({"model": model_name, "score": c["score"], "possible": c["possible"], "percentage": round(pct, 1)})
        cap_scores.sort(key=lambda x: (-x["score"], x["possible"]))
        for rank, entry in enumerate(cap_scores, 1):
            entry["rank"] = rank
        capability_rankings[cap] = cap_scores

    # Build full results structure
    full_results = {
        "metadata": {
            "timestamp": run_timestamp,
            "run_id": run_id,
            "capabilities": selected_caps,
            "total_tests": len(tests),
            "total_possible_per_model": len(tests) * 4,
            "models": [m["name"] for m in models]
        },
        "models": {},
        "rankings": model_totals,
        "best_capability_by_model": best_capability_per_model,
        "capability_rankings": capability_rankings
    }

    for model_name in results:
        # Collect just the test dicts (exclude bonus keys)
        test_entries = {k: v for k, v in results[model_name].items() if isinstance(v, dict) and "score" in v}
        full_results["models"][model_name] = {
            "tests": test_entries,
            "capabilities": capability_scores.get(model_name, {}),
            "total_score": sum(v["score"] for v in test_entries.values()),
            "total_possible": len(test_entries) * 4,
            "total_time_seconds": round(model_timings[model_name], 2),
            "rank": rankings.get(model_name, "-")
        }

    # Write results with timestamp
    filename = f"results_{run_id}.json"
    out_path = os.path.join(settings.get("results_dir", "results"), filename)
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)

    # Also write latest.json for convenience
    latest_path = os.path.join(settings.get("results_dir", "results"), "results_latest.json")
    with open(latest_path, "w") as f:
        json.dump(full_results, f, indent=2)

    # Write human-readable summary
    summary_filename = f"results_{run_id}.txt"
    summary_path = os.path.join(settings.get("results_dir", "results"), summary_filename)
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LOCAL MODEL EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp (UTC): {run_timestamp}\n")
        f.write(f"Capabilities: {', '.join(selected_caps)}\n")
        f.write(f"Total Tests: {len(tests)}\n")
        f.write(f"Total Possible per Model: {len(tests) * 4} points\n")
        f.write(f"Models Tested: {', '.join([m['name'] for m in models])}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("OVERALL RANKINGS (sorted by score ↓, then time ↑)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Rank':<5} {'Model':<45} {'Score':<10} {'Pct':<6} {'Time (s)':<10}\n")
        f.write("-" * 70 + "\n")
        for entry in model_totals:
            f.write(f"{entry['rank']:<5} {entry['model']:<45} {entry['earned']}/{entry['possible']:<10} {entry['percentage']}%{'':<4} {entry['total_time']:<10.1f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("BEST CAPABILITY PER MODEL\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Model':<45} {'Capability':<25} {'Score':<10}\n")
        f.write("-" * 70 + "\n")
        for model_name, best in best_capability_per_model.items():
            pct_str = f" ({best['percentage']}%)" if best['possible'] > 0 else ""
            f.write(f"{model_name:<45} {best['capability']:<25} {best['score']}/{best['possible']}{pct_str}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("PER-CAPABILITY RANKINGS (all models)\n")
        f.write("=" * 70 + "\n")
        for cap, scores in capability_rankings.items():
            f.write(f"\n  {cap}:\n")
            for s in scores:
                f.write(f"    Rank {s['rank']}: {s['model']:<40} {s['score']}/{s['possible']} ({s['percentage']}%)\n")

    # Print summary table
    print("\n" + "=" * 70)
    print("RANKings (sorted by score ↓, then time ↑)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Model':<45} {'Score':<10} {'Pct':<6} {'Time (s)':<10}")
    print("-" * 70)
    for entry in model_totals:
        print(f"{entry['rank']:<5} {entry['model']:<45} {entry['earned']}/{entry['possible']:<10} {entry['percentage']}%{'':<4} {entry['total_time']:<10.1f}")

    print("\n" + "=" * 70)
    print("Best Capability per Model")
    print("=" * 70)
    print(f"{'Model':<45} {'Capability':<25} {'Score':<10}")
    print("-" * 70)
    for model_name, best in best_capability_per_model.items():
        pct_str = f" ({best['percentage']}%)" if best['possible'] > 0 else ""
        print(f"{model_name:<45} {best['capability']:<25} {best['score']}/{best['possible']}{pct_str}")

    print("\n" + "=" * 70)
    print("Per-Capability Rankings")
    print("=" * 70)
    for cap, scores in capability_rankings.items():
        print(f"\n  {cap}:")
        for s in scores:
            print(f"    Rank {s['rank']}: {s['model']:<40} {s['score']}/{s['possible']} ({s['percentage']}%)")

    print(f"\nResults saved:")
    print(f"  JSON:    {out_path}")
    print(f"  Summary: {summary_path}")

if __name__ == "__main__":
    main()
