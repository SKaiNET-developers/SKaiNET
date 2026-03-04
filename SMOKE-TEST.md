# Smoke Test

Quick smoke test for loading LLMs via SKaiNET. Supports multiple runners
(kllama, kgemma, kbert) and model formats (GGUF, SafeTensors).

## Quick start

```bash
# Uses smoke-models.json in the same directory
./smoke-test.sh

# Use a custom config
./smoke-test.sh --config my-models.json

# Legacy: scan a directory for .gguf files
./smoke-test.sh /path/to/models

# Legacy: test specific .gguf files
./smoke-test.sh model1.gguf model2.gguf
```

## Configuration

Define models in `smoke-models.json`:

```json
{
  "defaults": {
    "prompt": "The capital of France is",
    "steps": 32,
    "temperature": 0.0
  },
  "models": [
    {
      "name": "Llama-3.2-1B-Q8",
      "runner": "kllama",
      "model": "~/.lmstudio/models/llama-3.2-1b/llama-3.2-1b-q8_0.gguf",
      "format": "gguf"
    }
  ]
}
```

### Model fields

| Field         | Required | Description                                    |
|---------------|----------|------------------------------------------------|
| `name`        | yes      | Display name for the summary table             |
| `runner`      | yes      | Which runner to use (`kllama`, `kgemma`, `kbert`) |
| `model`       | yes      | Path to the model file or directory (`~` is expanded) |
| `format`      | no       | `gguf` or `safetensors` (informational)        |
| `prompt`      | no       | Overrides the default prompt                   |
| `steps`       | no       | Overrides the default step count               |
| `temperature` | no       | Overrides the default temperature              |
| `doc`         | no       | Document text for kbert similarity comparisons |

### Runners

| Runner   | Gradle task                              | Use case              |
|----------|------------------------------------------|-----------------------|
| `kllama` | `:skainet-apps:skainet-kllama:run`       | GGUF LLMs (Llama, etc.) |
| `kgemma` | `:skainet-apps:skainet-kgemma:jvmRun`    | SafeTensors (Gemma)   |
| `kbert`  | `:skainet-apps:skainet-kbert-cli:run`    | BERT embeddings       |

## Adding a new model

1. Add an entry to the `models` array in `smoke-models.json`
2. Set `runner` to match the model architecture
3. Set `model` to the path on your machine
4. Optionally override `prompt`, `steps`, or `temperature`

## Adding a new runner

1. Add cases to `runner_task()`, `runner_compile_task()`, and `runner_args()` in `smoke-test.sh`
2. Use the new runner name in `smoke-models.json`

## Environment variables (legacy mode)

| Variable       | Default                      |
|----------------|------------------------------|
| `SMOKE_PROMPT` | `The capital of France is`   |
| `SMOKE_STEPS`  | `32`                         |
| `SMOKE_TEMP`   | `0.0`                        |
