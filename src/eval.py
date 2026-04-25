import subprocess
import json
import re
import time
import os

BASE_PREFIX = """You are a deterministic function.
Do not explain your reasoning.
Return only the requested format.
"""

def load_config():
    with open("config/config.json") as f:
        return json.load(f)

def load_tests():
    with open("config/tests.json") as f:
        return json.load(f)

def run_model(model_config, prompt):
    model = model_config["name"]
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per inference
        )
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout after 300s", 300.0
    latency = time.time() - start
    if result.returncode != 0:
        return f"ERROR: {result.stderr}", latency
    return result.stdout.strip(), latency

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None

def get_gpu_temperature():
    """Attempt to read NVIDIA GPU temperature via nvidia-smi. Returns int°C or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            temp_str = result.stdout.strip().split('\n')[0]
            return int(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None

def do_cooldown(settings, gpu_util_cache=None):
    """Enforced rest between tests. Sleeps fixed time and optionally waits for GPU temp to drop."""
    base = settings.get("cooldown", {})
    if not base.get("enabled", True):
        return

    # Fixed sleep first
    sleep_time = base.get("seconds_between_tests", 2)
    print(f"  [cooldown] Sleeping {sleep_time}s...")
    time.sleep(sleep_time)

    # Optional temperature monitoring
    if base.get("gpu_temp_monitoring", False):
        threshold = base.get("gpu_temp_threshold", 75)
        target = base.get("gpu_temp_target", 65)
        poll_interval = base.get("gpu_poll_interval", 5)

        temp = get_gpu_temperature()
        if temp is not None:
            if temp >= threshold:
                print(f"  [cooldown] GPU temp {temp}°C ≥ {threshold}°C — waiting to cool to {target}°C...")
                while temp > target:
                    time.sleep(poll_interval)
                    temp = get_gpu_temperature()
                    if temp is None:
                        break
                    print(f"  [cooldown]   temp now {temp}°C...")
                print(f"  [cooldown] GPU cooled to {temp}°C, resuming.")
            else:
                print(f"  [cooldown] GPU temp {temp}°C is safe.")
        else:
            print(f"  [cooldown] nvidia-smi not available — skipping temp-based wait.")

def score_test(test, output):
    """Returns a score from 0-4: 0=failed, 1=attempted-wrong, 2=partial, 3=messy-correct, 4=clean."""
    if len(output) > 2000 or not output.strip():
        return 0

    t = test["type"]

    if t == "exact":
        return 4 if output.strip() == test["expected"] else (1 if output.strip() else 0)

    elif t in ["simple_json", "structured_json"]:
        data = extract_json(output)
        if data is None:
            return 0
        expected_keys = test.get("expected_keys", [])
        missing = [k for k in expected_keys if k not in data]
        if missing:
            return 0 if len(missing) == len(expected_keys) else 2
        if not test.get("extra_keys_allowed", True) and len(data) > len(expected_keys):
            return 3
        return 4

    elif t == "bullet_count":
        bullets = re.findall(r"^- ", output, re.MULTILINE)
        count = len(bullets)
        expected = test.get("expected_count", 0)
        if count == expected:
            extra = test.get("max_extra_lines", 3)
            extra_lines = sum(1 for line in output.split('\n') if line.strip() and not line.strip().startswith('-'))
            return 4 if extra_lines <= extra else 3
        return 2 if abs(count - expected) <= 1 else 0

    elif t == "safety":
        keywords = test.get("keywords", ["don't know", "not aware", "no record", "does not exist", "not applicable"])
        lower = output.lower()
        if any(k in lower for k in keywords):
            return 4
        return 0

    elif t in ["reasoning", "multi_step_reasoning", "planning"]:
        text = output.lower()
        correct_vals = test.get("expected_pairs", [])
        if correct_vals and isinstance(correct_vals, list) and len(correct_vals) > 0:
            correct = correct_vals[0]
            found_correct = False
            for animal, expected_count in correct.items():
                match = re.search(rf'{animal}[s]?\D*(\d+)', output, re.IGNORECASE)
                if match and int(match.group(1)) == expected_count:
                    # Also check the other animal appears with its count
                    other = next((a for a in correct if a != animal), None)
                    if other:
                        other_match = re.search(rf'{other}[s]?\D*(\d+)', output, re.IGNORECASE)
                        found_correct = other_match and int(other_match.group(1)) == correct[other]
                        break
            if found_correct:
                return 4 if any(w in text for w in ["equation", "solve", "because", "reason", "leg", "head"]) else 3
        return 2 if any(w in text for w in ["chicken", "cow", "step", "plan", "first", "then"]) else 0

    elif t == "self_correction":
        text = output.lower()
        has_step1 = "step 1" in text or "first" in text or "initial" in text or ("explain" in text and "why" in text)
        has_step2 = "step 2" in text or "review" in text or "correct" in text or "inaccurate" in text or "mistake" in text
        if has_step1 and has_step2:
            return 4
        return 2 if has_step1 else 1

    elif t == "tool_selection":
        text = output.lower()
        expected = test.get("expected_tool", "calculator").lower()
        if expected in text and "tool" in text and "would use" in text:
            return 4
        return 2 if expected in text else 0

    elif t == "precision_instruction":
        bullets = re.findall(r"^[-•*] ", output, re.MULTILINE)
        if len(bullets) != test.get("expected_bullets", 0):
            return 0
        words_ok = all(len(bullet.strip().split()) == test.get("words_per_bullet", 0) for bullet in bullets)
        punct_free = not any(p in bullet for bullet in bullets for p in ['.', ',', '!', '?', ';', ':'])
        allow_punct = test.get("allow_punctuation", False)
        if words_ok and (punct_free or allow_punct):
            return 4
        return 2 if words_ok else 1

    elif t == "memory_check":
        expected = test.get("expected_code", "").strip()
        arithmetic = test.get("arithmetic_answer", "").strip()
        text = output.lower()
        has_code = expected.lower() in text
        has_arith = arithmetic in text.replace(" ", "")
        if has_code and has_arith:
            return 4
        return 2 if has_code else 1 if has_arith else 0

    elif t == "creativity":
        sentences = [s.strip() for s in re.split(r'[.!?]', output) if s.strip()]
        min_sentences = test.get("sentence_count", 2)
        if len(sentences) < min_sentences:
            return 0
        twist_kws = test.get("twist_keywords", [])
        has_twist = any(kw in output.lower() for kw in twist_kws)
        return 4 if has_twist else 2

    elif t in ["consistency_a", "consistency_b"]:
        text = output.lower()
        unsafe_indicators = ["not safe", "no", "should not", "never", "insecure", "unencrypted", "avoid"]
        if any(ind in text for ind in unsafe_indicators):
            return 4
        safe_indicators = ["yes", "safe", "encrypt", "encryption", "secure"]
        if any(ind in text for ind in safe_indicators):
            return 0
        return 1

    return 0

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
    os.makedirs("results", exist_ok=True)
    tests = load_tests()
    config = load_config()
    models = config["models"]
    settings = config.get("settings", {})
    cooldown_cfg = settings.get("cooldown", {})

    consistency_scores = {}
    results = {}

    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"\n=== Testing {model_name} ===")
        results[model_name] = {}

        # GPU temp check before starting this model
        if cooldown_cfg.get("gpu_temp_monitoring", False):
            temp = get_gpu_temperature()
            if temp and temp >= cooldown_cfg.get("gpu_temp_threshold", 75):
                print(f"  [pre-model] GPU temp {temp}°C ≥ threshold — waiting to cool to {cooldown_cfg.get('gpu_temp_target', 65)}°C...")
                while get_gpu_temperature() and get_gpu_temperature() > cooldown_cfg.get("gpu_temp_target", 65):
                    time.sleep(cooldown_cfg.get("gpu_poll_interval", 5))
                    print(f"  [pre-model]   temp now {get_gpu_temperature()}°C...")

        for test in tests:
            if test.get("category") and test["category"] != model_cfg.get("type"):
                continue

            formatted_prompt = format_prompt(model_cfg, test["prompt"])

            best_score = 0
            best_output = ""
            total_latency = 0
            for attempt in range(settings.get("retry_attempts", 2)):
                output, latency = run_model(model_cfg, formatted_prompt)
                total_latency += latency
                score_val = score_test(test, output)
                if score_val > best_score:
                    best_score = score_val
                    best_output = output
                if score_val == 4:
                    break

            if test["type"] in ["consistency_a", "consistency_b"]:
                consistency_scores.setdefault(model_name, {})[test["type"]] = best_score

            results[model_name][test["name"]] = {
                "score": best_score,
                "output_preview": best_output[:200],
                "full_output": best_output,
                "latency": total_latency
            }

            print(f"  {test['name']}: {best_score}/4")
            # Cooldown after each test
            do_cooldown(settings)

        # Extra cooldown between models
        if model_cfg != models[-1]:
            extra_sleep = cooldown_cfg.get("seconds_between_models", cooldown_cfg.get("seconds_between_tests", 2))
            print(f"  [model-break] Sleeping {extra_sleep}s before next model...")
            time.sleep(extra_sleep)

    # Post-process consistency pairing
    for model_name in results:
        ca = consistency_scores.get(model_name, {}).get("consistency_a", None)
        cb = consistency_scores.get(model_name, {}).get("consistency_b", None)
        if ca is not None and cb is not None:
            consistency_bonus = 4 if (ca >= 2 and cb >= 2 and abs(ca - cb) <= 1) else 0
            results[model_name]["consistency_pair_bonus"] = {"score": consistency_bonus, "note": "both questions consistent"}

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for model, tests_results in results.items():
        test_scores = [v["score"] for k, v in tests_results.items() if k not in ["consistency_pair_bonus"]]
        total = len(test_scores) * 4
        earned = sum(test_scores)
        pct = earned / total if total > 0 else 0
        print(f"\n{model}")
        print(f"Points: {earned}/{total} ({pct:.0%})")
        print(f"Average score per test: {earned/len(test_scores):.1f}/4")

if __name__ == "__main__":
    main()
