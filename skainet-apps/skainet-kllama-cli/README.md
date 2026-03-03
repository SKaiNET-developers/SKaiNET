# KLlama CLI

Command-line interface for running LLaMA models with [KLlama](../skainet-kllama/README.md). Supports single-prompt generation, interactive chat, and agent mode with tool calling.

For embedding KLlama in your own application, see the [KLlama library README](../skainet-kllama/README.md).

---

## Building

Build the fat JAR from the project root (requires JDK 21+):

```bash
./gradlew :skainet-apps:skainet-kllama-cli:shadowJar
```

Output: `skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar`

---

## Usage

KLlama uses the Java Vector API for SIMD-accelerated inference. Always include the required JVM flags:

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -jar skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar [options] "<prompt>"
```

### Options

| Option | Description | Default |
|---|---|---|
| `-m, --model` | Path to `.gguf`, `.safetensors`, `.bin` model, or HuggingFace directory | *(required)* |
| `-t, --tokenizer` | Path to tokenizer file (auto-detected for GGUF and SafeTensors) | auto |
| `-s, --steps` | Number of tokens to generate | `64` |
| `-k, --temperature` | Sampling temperature | `0.8` |
| `-p, --systemprompt` | System prompt prepended to the conversation | built-in default |
| `--chat` | Interactive multi-turn chat mode | off |
| `--agent` | Interactive agent mode with tool calling | off |
| `--demo` | Tool calling demo (calculator + file listing) | off |
| `--template=NAME` | Chat template: `llama3` or `chatml` | `llama3` |
| `-h, --help` | Show help and exit | |

---

## Examples

### Single Prompt (GGUF)

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -jar kllama-all.jar \
     -m tinyllama-1.1b-q4.gguf \
     -s 96 -k 0.7 \
     -p "You are concise" \
     "Once upon a time"
```

### Interactive Chat

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -jar kllama-all.jar \
     -m model.gguf --chat
```

### Agent Mode (Tool Calling)

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -jar kllama-all.jar \
     -m model.gguf --agent --template=chatml
```

### Tool Calling Demo

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -jar kllama-all.jar \
     -m model.gguf --demo
```

### SafeTensors (HuggingFace Directory)

Point `-m` at a directory containing `model.safetensors`, `config.json`, and `tokenizer.json`:

```bash
java --enable-preview --add-modules jdk.incubator.vector \
     -jar kllama-all.jar \
     -m ./my-llama-model/ \
     "Hello, world!"
```

---

## Native Executable (macOS)

A Kotlin/Native executable is also available for macOS:

```bash
./skainet-apps/skainet-kllama/build/bin/macosArm64/releaseExecutable/kllama.kexe \
     <model_path> "<prompt>" [steps] [temperature] [--backend=cpu]
```

| Option | Description |
|---|---|
| `--backend=mlx` | Use MLX GPU backend (default on macOS) |
| `--backend=cpu` | Use CPU backend |
| `--list-backends` | Show available backends and exit |

```bash
# CPU backend
./kllama.kexe tinyllama-1.1b.gguf "Once upon a time" --backend=cpu

# List available backends
./kllama.kexe --list-backends
```
