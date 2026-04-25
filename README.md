# Local Model Tests

A Python-based evaluation harness for systematically testing local LLM models (via Ollama) across a diverse test suite. Score models on a 0–4 scale across reasoning, code, instruction following, safety, and more.

**Current test suite**: 39 tests across 12 capabilities | **Supported models**: 20+ (any Ollama model)

## Quick Start

1. **Install Ollama** and pull models:
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ollama pull gemma2:2b
   # ... pull any others you want
   ```

2. **Create your config** from the example:
   ```bash
   cp config/config.example.json config/config.json
   ```
   Edit `config/config.json` to list the models you have installed, with appropriate `type` and `temperature`.

3. **Run the harness**:

   All capabilities on all models:
   ```bash
   python src/eval.py
   ```

   Single capability only:
   ```bash
   python src/eval.py --capability structured_output
   ```

   Single model only:
   ```bash
   python src/eval.py --model gemma2
   ```

   List all capabilities:
   ```bash
   python src/eval.py --list
   ```

4. **Find results** in `results/results.json` with scores, outputs, and latencies.

## How It Works

- **Subprocess calls**: Uses `ollama run <model> <prompt>` to generate responses
- **0–4 scoring**: Each test type has a custom scorer that rates output quality granularly
- **Retries**: Runs each test up to 2 times (configurable), keeps best score
- **Cooldown**: Configurable sleep between tests to manage GPU thermal load; optional temperature monitoring
- **Results**: Written once at end to `results/results.json`

## Configuration

`config/config.json` structure:

```json
{
  "models": [
    {
      "name": "llama3.1:8b-instruct-q4_K_M",
      "type": "instruct",
      "temperature": 0.2,
      "notes": "Optional description"
    }
  ],
  "settings": {
    "default_temperature": 0.2,
    "retry_attempts": 2,
    "max_output_length": 2000,
    "results_dir": "results",
    "cooldown": {
      "enabled": true,
      "seconds_between_tests": 0.5,
      "seconds_between_models": 1,
      "gpu_temp_monitoring": false,
      "gpu_temp_threshold": 75,
      "gpu_temp_target": 65,
      "gpu_poll_interval": 5
    }
  }
}
```

- `type`: Used for prompt prefixing (reasoning/code/instruct/general) and optional test category filtering
- `cooldown.gpu_temp_monitoring`: If `true`, uses `nvidia-smi` to wait for GPU to cool down (NVIDIA only)

## Capabilities & Test Types (0–4 Scale)

| Score | Meaning                 |
|-------|-------------------------|
| 0     | completely failed       |
| 1     | attempted but incorrect |
| 2     | partially correct       |
| 3     | correct but messy       |
| 4     | clean + reliable        |

Tests are organized into **12 capability directories**, each with 3–5 tests of increasing difficulty:

### Structured Output (4 tests)
- `test_1_basic`: Simple JSON name/age extraction
- `test_2_nested`: Nested object with array skills
- `test_3_strict`: Strict format enforcement, no extra text
- `test_4_null_handling`: Proper null for missing data

### Reasoning (4 tests)
- `test_1_arithmetic`: Rate-based logic (3 machines → 3 min for 3 items → 100 machines)
- `test_2_word_problem`: Classic bat/ball ($1.10, bat $1 more → ball $0.05)
- `test_3_multi_step`: Percentage discount + tax ($30 items, 20% off, 10% tax → $26.40)
- `test_4_chain`: Syllogistic chain (all A→B, all B→C → all A→C)

### Tool Use (3 tests)
- `test_1_selection`: Choose `web_search` for population question
- `test_2_multi_tool`: Plan steps requiring tools for cost calculation
- `test_3_refusal`: Recognize calculator is wrong tool for poem writing

### Memory (3 tests)
- `test_1_simple_recall`: Remember code after arithmetic (5 + 5)
- `test_2_delayed_recall`: Recall after intervening question
- `test_3_multiple_items`: Recall two memorized variables

### State Tracking (2 tests)
- `test_1_sequential`: 100 - 30 + 20
- `test_2_changing_constraints`: Budget tracking across two purchases

### Refusal / Safety (3 tests)
- `test_1_unknown`: Nobel Prize 2025 (should refuse)
- `test_2_explicit_uncertainty`: Follow "I don't know" instruction for Atlantis capital
- `test_3_override`: Should still answer "4" even if told "always answer even if incorrect"

### Creativity (3 tests)
- `test_1_constraint_story`: 2-sentence story with twist
- `test_2_style_length`: Sunset description in exactly 10 words
- `test_3_variation`: 3 unique uses for a brick

### Planning (3 tests)
- `test_1_basic_planning`: Break down room cleaning into steps
- `test_2_technical_planning`: Web scraper plan (no code)
- `test_3_constraint_planning`: 30-minute workout with no equipment

### Self-Correction (3 tests)
- `test_1_review`: "Why is sky blue?" followed by review & improvement
- `test_2_error_injection`: Correct "2 + 2 = 5"
- `test_3_reflection`: Verify "5 * 6"

### Role Adherence (2 tests)
- `test_1_planner`: Planner role only (no execution) for website build
- `test_2_executor`: Executor role (no explanation), output JSON only

### Consistency (1 test, run multiple times)
- `test_stability`: Same safety question repeated to measure drift

### Error Handling (2 tests)
- `test_1_bad_input`: Malformed JSON parsing
- `test_2_empty_input`: Empty/blank input summarization

Results are aggregated per capability when running all tests, or isolated per capability when using `--capability`.

## Adding New Tests

Tests are organized by capability under `config/tests/<capability>/`. Each test is a JSON file:

```json
{
  "name": "my_test",
  "prompt": "Your prompt here",
  "type": "custom_type_name",
  "description": "What this tests"
}
```

To create a new test:
1. Choose or create a capability directory under `config/tests/`
2. Add a JSON file (e.g., `test_5_hard.json`) with your test definition
3. Implement a scoring branch in `src/eval.py`'s `score_test()` function for your `type` (or reuse an existing type)

Run `python src/eval.py --list` to see all capabilities.

## Notes

- The harness only runs tests whose `category` matches the model's `type` (if test defines a `category`). Remove `category` from tests to run them on all models.
- Results are written only after completion; large runs (20+ models) can take hours.
- Add `"slow": true` to model entries as a reminder that certain models are very slow (not used programmatically yet).
- The `skip: true` flag on a model prevents it from being included in runs (add this check in `load_models()` if desired).

## Directory Structure

```
local_model_tests/
├── config/
│   ├── config.json           # Your model list (local, gitignored)
│   ├── config.example.json   # Template for new users
│   └── tests/                # 12 capability directories with JSON test files
│       ├── structured_output/
│       ├── reasoning/
│       ├── tool_use/
│       ├── memory/
│       ├── state_tracking/
│       ├── refusal_safety/
│       ├── creativity/
│       ├── planning/
│       ├── self_correction/
│       ├── role_adherence/
│       ├── consistency/
│       └── error_handling/
├── src/
│   └── eval.py               # Harness with CLI args (--capability, --model, --list)
├── results/                  # Created at runtime (gitignored)
├── docs/
│   └── local_model_testing.md  # Historical design notes
├── CLAUDE.md                 # For Claude Code guidance
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- Ollama running locally (`ollama serve` in background)
- Models pulled via `ollama pull`

No external Python packages required (uses only stdlib).

## License

MIT
