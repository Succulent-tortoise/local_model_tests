# Local Model Tests

A Python-based evaluation harness for systematically testing local LLM models (via Ollama) across a diverse test suite. Score models on a 0–4 scale across reasoning, code, instruction following, safety, and more.

**Current test suite**: 13 tests | **Supported models**: 20+ (any Ollama model)

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
   ```bash
   python src/eval.py
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

## Test Types (0–4 Scale)

| Score | Meaning                 |
|-------|-------------------------|
| 0     | completely failed       |
| 1     | attempted but incorrect |
| 2     | partially correct       |
| 3     | correct but messy       |
| 4     | clean + reliable        |

### Test Descriptions

- `structured_output_nested` — strict JSON schema with array field
- `multi_step_reasoning` — word problem (chickens/cows), expects equations and correct answer (18 chickens, 12 cows)
- `self_correction` — two-step "write then review" in one prompt
- `tool_use_simulation` — selects correct tool (calculator) without answering directly
- `instruction_edge_case` — 3 bullets, 5 words each, no punctuation
- `context_retention` — remembers code `8472XQ` after arithmetic distraction
- `hallucination_trap_strong` — Nobel Prize 2025; expects "not applicable"
- `planning_decomposition` — web scraper planning steps, no code
- `controlled_creativity` — 2-sentence story with a twist
- `consistency_test_a/b` — same safety question in two wordings (paired consistency bonus)
- `instruction_following_bullets` — exactly 5 markdown bullets
- `json_extraction_simple` — JSON with expected keys

## Adding New Tests

Edit `config/tests.json`. Each test needs:

```json
{
  "name": "my_test",
  "prompt": "Your prompt here",
  "type": "custom_type_name",
  "description": "What this tests"
}
```

Then implement a scoring branch in `src/eval.py` in the `score_test()` function for your `type`.

## Notes

- The harness only runs tests whose `category` matches the model's `type` (if test defines a `category`). Remove `category` from tests to run them on all models.
- Results are written only after completion; large runs (20+ models) can take hours.
- Add `"slow": true` to model entries as a reminder that certain models are very slow (not used programmatically yet).
- The `skip: true` flag on a model prevents it from being included in runs (add this check in `load_models()` if desired).

## Directory Structure

```
local_model_tests/
├── config/
│   ├── config.json          # Your model list (local, not committed)
│   ├── config.example.json  # Template for new users
│   └── tests.json           # Test definitions
├── src/
│   └── eval.py              # Harness
├── results/                 # Created at runtime
├── docs/
│   └── local_model_testing.md  # Design history
├── CLAUDE.md                # For Claude Code guidance
└── README.md                # This file
```

## Requirements

- Python 3.8+
- Ollama running locally (`ollama serve` in background)
- Models pulled via `ollama pull`

No external Python packages required (uses only stdlib).

## License

MIT
