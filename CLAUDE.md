# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

**Run the full evaluation harness:**
```bash
python src/eval.py
```

**Check available Ollama models:**
```bash
ollama list
```

**Pull a new model:**
```bash
ollama pull <model-name>
```

**Run a single model test interactively (debugging):**
```bash
ollama run <model-name> "Your prompt here"
```

**View latest results:**
```bash
cat results/results.json | python -m json.tool
```

## Project Overview

A Python-based evaluation harness for testing local LLM models via Ollama. The project defines a suite of tests (reasoning, JSON extraction, instruction following, safety) and runs them against multiple local models to compare performance, accuracy, and latency.

**Current models** (in `config/config.json`): 21 models spanning 5 categories:
- **Reasoning/agent**: openhermes:7b, phi3:3.8b-mini, mixtral:8x7b, llama3.1
- **Code**: deepseek-coder:6.7b-instruct, starcoder2:7b, codellama:7b-instruct, qwen2.5-coder series
- **Instruct**: llama3.1:8b-instruct, qwen2.5:7b-instruct, mistral-nemo:12b, phi3:3.8b-mini
- **General**: gemma2:2b, llama3.2:latest, yi:6b-chat, mistral:latest
- **Embedding**: nomic-embed-text:latest (marked `skip: true`)

See `config/config.json` for the full registry with temperatures, sizes, and notes.

## Directory Structure

```
local_model_tests/
├── config/
│   ├── config.json       # Model registry + global settings (cooldown, retries)
│   └── tests.json        # 13 test definitions with prompts and scoring criteria
├── src/
│   └── eval.py           # Main harness — config loading, Ollama subprocesses, scoring, reporting
├── results/
│   └── results.json      # Generated at runtime — full outputs, scores, latencies
└── docs/
    └── local_model_testing.md  # Original design doc with test ideas
```

## Architecture

### Core Flow (`eval.py`)

1. **Load configuration** — `config/config.json` (models + settings) and `config/tests.json` (test suite)
2. **Iterate models × tests** — For each model:
   - Filter tests by `category` (only run if `test.category == model.type` or no category)
   - Format prompt via `format_prompt()` with model-specific prefix
   - Execute `ollama run <model>` as subprocess (capture stdout/stderr)
   - Apply retry logic (default 2 attempts, keeps best score)
3. **Scoring** — `score_test()` uses a **0–4 rubric**:

| Score | Meaning                 |
|-------|-------------------------|
| 0     | completely failed       |
| 1     | attempted but incorrect |
| 2     | partially correct       |
| 3     | correct but messy       |
| 4     | clean + reliable        |

4. **Post-processing** — Compute `consistency_pair_bonus` for paired `consistency_a`/`consistency_b` tests
5. **Output** — Write `results/results.json`, print per-model summary (points, percentage, average)

### Scoring Types

The harness supports 13 test types with specialized scoring logic:

- **`structured_json` / `simple_json`** — JSON extraction with key validation, optional strict schema
- **`reasoning` / `multi_step_reasoning`** — Regex pattern matching for numeric answers, reasoning keywords
- **`self_correction** — Detects two-phase structure (initial answer + review/correction)
- **`tool_selection`** — Checks for expected tool name + justification language
- **`precision_instruction`** — Bullet count, word count, punctuation rules
- **`memory_check`** — Dual verification: code recall + arithmetic answer
- **`safety`** — Keyword matching for appropriate refusal/hallucination avoidance
- **`planning`** — Step count validation, code absence check
- **`creativity`** — Sentence count + twist keyword detection
- **`consistency_a` / `consistency_b`** — Safety-consistency pairing (sensitive/unsafe topics)
- **`bullet_count`** — Exact bullet count with optional extra line tolerance

### Prompt Formatting (`format_prompt`)

A `BASE_PREFIX` enforces deterministic output:
```
You are a deterministic function.
Do not explain your reasoning.
Return only the requested format.
```

Model-type modifiers are appended:
- `reasoning` → "Answer concisely."
- `code` → "Return only code or structured output."
- `instruct` → "Follow instructions exactly."

### Configuration Schema

**`config/config.json`:**
```json
{
  "models": [
    {
      "name": "model-name:tag",
      "type": "reasoning|code|instruct|general|embedding",
      "temperature": 0.1-0.4,
      "size": "XR GB",           // info only
      "notes": "description",     // info only
      "slow": false,              // optional: flag for long-running models
      "skip": false               // optional: exclude from evaluation
    }
  ],
  "settings": {
    "default_temperature": 0.2,
    "retry_attempts": 2,
    "max_output_length": 2000,
    "cooldown": {
      "enabled": true,
      "seconds_between_tests": 2,
      "seconds_between_models": 5,
      "gpu_temp_monitoring": true,
      "gpu_temp_threshold": 75,
      "gpu_temp_target": 65,
      "gpu_poll_interval": 5
    }
  }
}
```

**`config/tests.json`:**
Each test has:
```json
{
  "name": "test_identifier",
  "prompt": "exact prompt text",
  "type": "reasoning|json|bullet_count|safety|...",
  "category": "reasoning|code|json|instruction|memory|general", // optional, filters by model.type
  // Type-specific fields:
  "expected": "exact match",
  "expected_keys": ["key1", "key2"],
  "expected_count": 5,
  "expected_pairs": [{"chickens": 18, "cows": 12}],
  "expected_tool": "calculator",
  // ...plus other type-specific criteria
}
```

### GPU Cooldown System

The harness includes automatic thermal management to prevent GPU throttling (requires NVIDIA GPU + `nvidia-smi`):

1. **Fixed sleep** — Always waits `seconds_between_tests` after each test
2. **Between models** — Additional `seconds_between_models` when switching models
3. **Temperature monitoring** — If `gpu_temp_monitoring` is true and temp ≥ threshold, polls every `gpu_poll_interval` seconds until reaching target

To disable: `"cooldown": {"enabled": false}` in config.

### Results Format

`results/results.json` structure:
```json
{
  "model-name": {
    "test_name": {
      "score": 0-4,
      "output_preview": "first 200 chars",
      "full_output": "complete response",
      "latency": 1.23
    },
    "consistency_pair_bonus": {
      "score": 0-4,
      "note": "both questions consistent"
    }
  }
}
```

## Development Tasks

### Adding a New Model

1. **Check availability:** `ollama list` or `ollama pull <model-name>`
2. **Edit** `config/config.json` → add entry to `"models"` array:
```json
{
  "name": "new-model:tag",
  "type": "reasoning",
  "temperature": 0.2,
  "notes": "brief description"
}
```
3. **Adjust** temperature based on model type:
   - `code`: 0.1 (deterministic)
   - `reasoning`: 0.2-0.3 (some creativity)
   - `general`/`instruct`: 0.3-0.4 (balanced)
4. **Run** harness: `python src/eval.py`

### Adding a New Test

1. **Edit** `config/tests.json` — append new test object
2. **Choose** `type` and provide required fields for that type
3. **Optionally set** `category` to filter which model types take this test
4. **Run** harness and verify scoring works as expected
5. **Consider** adding to documentation if reusable

### Running a Single Test for Debugging

**Via Ollama directly:**
```bash
ollama run <model-name> "Your prompt here"
```

**Via harness with specific test only:**
```bash
# Temporarily edit tests.json to include only the test you want
python src/eval.py
```

**Interactive inspection:** Check the full output in `results/results.json` to understand scoring decisions.

### Modifying Scoring Logic

**Edit** `score_test()` in `src/eval.py`. Each `test["type"]` has its own branch. To add a new type:
1. Define new `type` string in tests.json
2. Add corresponding `elif t == "your_type":` in `score_test()`
3. Implement scoring heuristics (exact match, keyword presence, regex patterns, etc.)
4. Update this CLAUDE.md with the new type

### Tweaking Prompt Prefixes

**Edit** `format_prompt()` in `src/eval.py`. The `BASE_PREFIX` applies to all models; the `if model_cfg.get("type")` branches add modifiers.

### Adjusting Cooldown Settings

**Edit** `config/config.json` → `settings.cooldown`:
- Disable entirely: `"enabled": false`
- Shorten/lengthen sleeps: `seconds_between_tests`, `seconds_between_models`
- Tune thermal thresholds: `gpu_temp_threshold` (trigger), `gpu_temp_target` (resume)
- Change poll interval: `gpu_poll_interval` (5s default)

### Filtering Slow Models

Add `"slow": true` to large models in `config/config.json`. When iterating models, you can skip them conditionally:
```python
if model_cfg.get("slow"):
    print(f"  Skipping {model_name} (marked slow)")
    continue
```

### Exporting/Importing Results

Results are in `results/results.json`. To copy/archive:
```bash
cp results/results.json results/backup-2026-04-25.json
```

To compare across runs, load both JSON files and diff the scores.

## Common Patterns & Gotchas

### Output Length Limits

The harness enforces `max_output_length` (default 2000 chars). Outputs exceeding this auto-score 0. Adjust in `config/config.json` → `settings.max_output_length` if needed (be careful — long outputs often indicate failure).

### JSON Extraction

The harness uses a **regex-based extractor** to find `{...}` blocks in model output. This fails if:
- JSON is malformed (missing braces, trailing commas)
- Model wraps JSON in code blocks with language tags (```json ...```)
  - The regex will still match; but malformed JSON still fails `json.loads()`
- Model includes multiple JSON objects

**Fix** — ensure prompts say "Return ONLY valid JSON." and consider pre-processing to strip code fences.

### Category Filtering

Tests with a `category` field only run for models whose `type` matches. If `category` is omitted, the test runs for all models. Common categories: `reasoning`, `code`, `json`, `instruction`, `memory`, `general`.

Use this to:
- Reduce runtime (only relevant tests per model)
- Compare cross-category performance (omit category to force all models)

### Retry Logic

The harness runs each test up to `retry_attempts` times (default 2), keeping the best score. Best latency is **not** kept — total latency sums all attempts. If you need "first success" latency, modify `main()` to break on score 4 and store that attempt's latency only.

### Consistency Bonus

Paired tests (`consistency_a` and `consistency_b`) receive a `consistency_pair_bonus` of 4 if:
- Both scores ≥ 2
- Absolute difference ≤ 1

This bonus is separate from individual test scores and appears as an extra key in the model's results.

### GPU Temperature Monitoring

Requires NVIDIA GPU + `nvidia-smi` in PATH. If not available, the harness prints a message and continues without temperature-based waiting. Non-NVIDIA GPUs (Apple Silicon, AMD) are not currently supported.

## Known Issues & Future Work

- **Consistency scoring**: The `consistency_pair_bonus` is computed but not included in the summary percentage — could be folded into total
- **Model selection logic**: `pick_best_model()` per category is proposed but not implemented — currently runs all models on all tests
- **Streamlit dashboard**: Planned for visual side-by-side model comparison
- **Test coverage gaps**: Categories like `memory` and `planning` are only lightly defined
- **Large model warnings**: Models marked `"slow": true` take 5–10× longer; consider filtering for quick sweeps

## Reference

**Model types** → `reasoning`, `code`, `instruct`, `general`, `embedding`

**Test types** → `structured_json`, `simple_json`, `reasoning`, `multi_step_reasoning`, `self_correction`, `tool_selection`, `precision_instruction`, `memory_check`, `safety`, `planning`, `creativity`, `consistency_a`, `consistency_b`, `bullet_count`

**Results location** → `results/results.json` (created at runtime)

**Design doc** → `docs/local_model_testing.md` (original concept and 12 test ideas)
