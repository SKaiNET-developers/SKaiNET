[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)
[![GitHub Contributors](https://img.shields.io/github/contributors/SKaiNET-developers/SKaiNET)](https://github.com/SKaiNET-developers/SKaiNET/graphs/contributors)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-View%20Docs-blue?logo=readthedocs&logoColor=white)](https://deepwiki.com/SKaiNET-developers/SKaiNET)

<img src="docs/SKaiNET-logo.png" alt="SKaiNET logo" width="150">

### Vision

SKaiNET aims to democratize "Edge AI / On-device AI" by bridging the gap between high-level application development and low-level hardware optimization. We believe AI should be portable, type-safe, and developer-friendly, enabling seamless intelligence in everything from mobile apps to IoT devices without sacrificing performance.

> For architecture details see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quickstart

Add the core dependencies (Gradle Kotlin DSL):

```kotlin
dependencies {
    implementation("sk.ainet.core:SKaiNET-lang-core:0.15.3")
    implementation("sk.ainet.core:SKaiNET-backend-cpu:0.15.3")
}
```

> **Java / Maven users** — see [Java Getting Started](docs/java-getting-started.md) for BOM setup and JVM flags.

### Hello Neural Net

```kotlin
val model = nn {
    input(28 * 28)
    dense(out = 128)
    relu()
    dense(out = 10)
}
```

### Hello LLM

```kotlin
val ctx = DirectCpuExecutionContext()
val ingestion = LlamaIngestion(ctx)
val weights = ingestion.load { SystemFileSystem.source(Path("model.gguf")).buffered() }
val tokenizer = GGUFTokenizer.fromSource(SystemFileSystem.source(Path("model.gguf")).buffered())

val runtime = LlamaRuntime(ctx, weights)
runtime.generate(tokenizer.encode("Once upon a time"), steps = 64) { token ->
    print(tokenizer.decode(token))
}
```

> **More examples:** [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) | [SKaiNET-notebook](https://github.com/SKaiNET-developers/SKaiNET-notebook)

---

## Explore

| Goal | Start here |
|---|---|
| Examples and sample projects | [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) |
| Interactive notebooks | [SKaiNET-notebook](https://github.com/SKaiNET-developers/SKaiNET-notebook) |
| LLM inference (Llama, Gemma) | [docs/kllama-getting-started.md](docs/kllama-getting-started.md) |
| Java 21+ integration | [docs/java-getting-started.md](docs/java-getting-started.md) |
| Data loading and transforms | [docs/io-readers-guide.md](docs/io-readers-guide.md) |
| Graph DSL (ResNet, YOLO) | [docs/graph-dsl.md](docs/graph-dsl.md) |
| Edge AI / Arduino export | [docs/arduino-c-codegen.md](docs/arduino-c-codegen.md) |
| MLIR / StableHLO compiler | [docs/hlo-getting-started.md](docs/hlo-getting-started.md) |
| Architecture overview | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Features

### Kotlin Multiplatform

- Targets: JVM, macOS (Native), JS, WASM (Browser + WasmWasi)
- Single codebase shared across all platforms via Kotlin Multiplatform

### LLM Inference

- **KLlama**: Llama-family models from GGUF files with streaming generation
- **KGemma**: Gemma 3n models from SafeTensors with HuggingFace tokenizer
- **KBert**: BERT and Sentence-Transformers for embeddings
- JVM acceleration via MemorySegment tensors, SIMD GEMM, paged KV cache

### Agentic AI

- Function / tool calling via `skainet-kllama-agent`
- `AgentLoop` for multi-turn tool-use conversations
- Java facade: `JavaAgentLoop`

### Neural Network DSL

- **Sequential**: `nn { input(); dense(); relu(); dense() }`
- **DAG / Graph**: arbitrary wiring with `dag { }` for ResNet, YOLO-style architectures
- Layers: Dense, Conv1d/2d/3d, MaxPool, AvgPool, BatchNorm, Dropout, LeakyReLU, ELU
- KAN (Kolmogorov–Arnold Networks) layer (experimental)
- Autograd engine with reverse-mode gradients, SGD and Adam/AdamW optimizers

### Data and I/O

- Built-in loaders: MNIST, Fashion-MNIST, CIFAR-10
- Formats: GGUF, ONNX, SafeTensors, JSON, Image (JPEG, PNG)
- Type-safe transform DSL: resize, crop, normalize, toTensor

### Java 21+ Support

- `SKaiNET` entry point, `TensorJavaOps`, builder-pattern model definition
- `KLlamaJava` / `KBertJava` facades for blocking and async inference
- Maven BOM (`sk.ainet:skainet-bom`) for one-line version management
- Docs: [Getting Started](docs/java-getting-started.md) | [LLM Inference](docs/java-llm-inference.md) | [Model Training](docs/java-model-training.md)

### Edge AI: Arduino / C99 Export

- Export trained models to standalone, optimized C99 with static memory allocation
- Ready-to-use Arduino library output
- See [arduino-c-codegen.md](docs/arduino-c-codegen.md)

### Compiler: MLIR / StableHLO

- Lower Kotlin DSL to MLIR StableHLO dialect
- Optimization passes: constant folding, operation fusion, dead code elimination
- Valid IREE-compilable output with streaming API and public `HloGenerator`
- See [hlo-getting-started.md](docs/hlo-getting-started.md)

---

## What's New in 0.15.3

- **System prompt support (Java)** — added `systemPrompt` support to `KLlamaJava` and `KLlamaSession` for easier customization of agent behavior
- **Model extraction** — model-specific code is now isolated in dedicated `skainet-models` modules
- **Whisper HLO generation** — fixed StableHLO MLIR output for Whisper audio models
- **Smoke test improvements** — faster and more robust LLM loading verification with multi-runner support


See [CHANGELOG.md](CHANGELOG.md) for the full release history.

---

## Roadmap

- **Q1 2026**: Comprehensive documentation ✅
- **Q2 2026**: Reference-based validation of computation correctness
- **Q3 2026**: Agentic AI enhancements ✅ (tool calling shipped in 0.13.0; ongoing)
- **Q4 2026**: Federated learning support for multi-device training

---

## Contributing & Community

We love contributions! Whether it's a new operator, documentation, or a bug fix:

1. Read our [Contribution Guide](CONTRIBUTING.md).
2. Check the [Good First Issues](https://github.com/SKaiNET-developers/SKaiNET/labels/good%20first%20issue).
3. Open a discussion or issue on [GitHub](https://github.com/SKaiNET-developers/SKaiNET/issues).

Browse the full codebase documentation on [DeepWiki](https://deepwiki.com/SKaiNET-developers/SKaiNET).

### Contributors (0.14.0)

- **Dhia Chemingui** ([@dhiaspaner](https://github.com/dhiaspaner)) — Android KMP plugin migration (#385, #386)

---

## License

MIT — see [LICENCE](LICENCE).
