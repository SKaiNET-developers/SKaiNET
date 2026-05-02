[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)
[![GitHub Contributors](https://img.shields.io/github/contributors/SKaiNET-developers/SKaiNET)](https://github.com/SKaiNET-developers/SKaiNET/graphs/contributors)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-View%20Docs-blue?logo=readthedocs&logoColor=white)](https://deepwiki.com/SKaiNET-developers/SKaiNET)

<img src="docs/modules/ROOT/images/SKaiNET-logo.png" alt="SKaiNET logo" width="150">

### Vision

SKaiNET aims to democratize "Edge AI / On-device AI" by bridging the gap between high-level application development and low-level hardware optimization. We believe AI should be portable, type-safe, and developer-friendly, enabling seamless intelligence in everything from mobile apps to IoT devices without sacrificing performance.

> For architecture details see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quickstart

Add the core dependencies (Gradle Kotlin DSL):

```kotlin
dependencies {
    // Recommended: import the umbrella BOM and drop versions on the engine modules.
    implementation(platform("sk.ainet:skainet-bom:0.22.2"))

    implementation("sk.ainet.core:skainet-lang-core")
    implementation("sk.ainet.core:skainet-backend-cpu")
}
```

> The BOM was first correctly published to Maven Central in 0.22.2 — earlier versions
> shipped at the wrong coordinates and could not be imported. Pin versions directly if
> you need an older release.

### Hello Neural Net

```kotlin
val model = nn {
    input(28 * 28)
    dense(out = 128)
    relu()
    dense(out = 10)
}
```

### Core Tensor Ops

```kotlin
val a = tensor(shape(2, 2)) { float(1f, 2f, 3f, 4f) }
val b = tensor(shape(2, 2)) { float(5f, 6f, 7f, 8f) }

val c = a matMul b
val d = c.relu()
```

### GGUF Model Loading

```kotlin
// Recommended: streaming reader — memory-efficient, supports quantized types
val source = JvmRandomAccessSource.open("model.gguf")
StreamingGGUFReader.open(source).use { reader ->
    println("Tensors: ${reader.tensorCount}")
    
    // Load specific tensor on demand (no whole-file loading)
    val bytes = reader.loadTensor("token_embd.weight")
    
    // Or get a TensorStorage descriptor with encoding/placement metadata
    val storage = reader.loadTensorStorage("token_embd.weight")
}
```

> **More examples:** [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) | [SKaiNET-notebook](https://github.com/SKaiNET-developers/SKaiNET-notebook)

---

## Ecosystem

SKaiNET is a modular ecosystem. While this repository contains the core engine, specialized high-level libraries are maintained in standalone repositories:

| Project | Description |
|---|---|
| [SKaiNET-LLM](https://github.com/SKaiNET-developers/SKaiNET-LLM) | Llama, Gemma, and BERT inference runtimes |
| [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers) | Pre-built transformer architectures and layers |
| [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) | Sample projects and integration demos |

---

## Explore

| Goal | Start here |
|---|---|
| Examples and sample projects | [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) |
| Interactive notebooks | [SKaiNET-notebook](https://github.com/SKaiNET-developers/SKaiNET-notebook) |
| LLM inference (Llama, Gemma) | [SKaiNET-LLM](https://github.com/SKaiNET-developers/SKaiNET-LLM) |

---

## Features

### Kotlin Multiplatform

- Targets: JVM, macOS (Native), JS, WASM (Browser + WasmWasi)
- Single codebase shared across all platforms via Kotlin Multiplatform

### Optimized Execution

- **ComputeGraphExecutor**: Optimized engine with fusion passes and trace-to-DAG bridging.
- **SDPA & Gather**: High-performance Scaled Dot-Product Attention and indexing operations.
- **TurboQuant**: Runtime KV-cache compression (~8x at 4-bit) for long-context LLM inference. Presets: `safe-lowbit`, `balanced`, `experimental-max`. See `TurboQuantUsage` for integration guide.

### Agentic AI Infrastructure

- **ComputeGraph**: Unified framework for defining agentic workflows and tool-calling loops.
- Java facade: `JavaAgentLoop` (in `skainet-lang-java`)

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
- Maven BOM (`sk.ainet:skainet-bom`) for one-line version management

### Edge AI: Arduino / C99 Export

- Export trained models to standalone, optimized C99 with static memory allocation
- Ready-to-use Arduino library output

### Compiler: MLIR / StableHLO

- Lower Kotlin DSL to MLIR StableHLO dialect
- Optimization passes: constant folding, operation fusion, dead code elimination
- Valid IREE-compilable output with streaming API and public `HloGenerator`

---

## What's New in 0.22.2

- **`sk.ainet:skainet-bom` now resolves from Maven Central.** The umbrella BOM was previously published at the wrong coordinates (`sk.ainet.core:skainet-bom`), so consumers following the standard `platform(...)` import pattern — and downstream BOMs like `sk.ainet.transformers:skainet-transformers-bom` that import it transitively — got 404s from Central. Hotfix; no API or behavior changes. (Issue #584)

## What's New in 0.22.0

- **Native (FFM) CPU kernel provider — M5 milestone closed.** New `skainet-backend-native-cpu` module bundles a hand-tuned C shared library (`-O3 -ffast-math` auto-vectorized into AVX2 / NEON FMA) reachable via FFM downcalls. **4.17×–5.87× faster than Panama Vector on Q4_K matmul** at LLM-typical 1024²–4096² shapes; **1.55×–1.77× faster on FP32 SGEMM** at 256³–1024³. Auto-registers via ServiceLoader; `KernelRegistry.bestAvailable()` routes through native when the lib loads, falls through cleanly to the priority-50 Panama provider otherwise.
- **Zero-copy MemSeg path for mmap'd Q4_K weights** — JVM-only `Q4KMemSegMatmulKernel` SPI sibling skips the staged `ByteArray → MemorySegment` copy that costs +20% wall-clock at 4096² shapes.
- **Cross-arch shipping** — published JAR carries native libs for `linux-x86_64`, `macos-arm64`, and `windows-x86_64`. Linux ARM64 consumers cleanly fall back to Panama (Kotlin/Native host limitation tracked).

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

---

## Roadmap

- **Q1 2026**: Comprehensive documentation ✅
- **Q2 2026**: TurboQuant KV-cache compression ✅ (shipped in 0.18.0); Qwen/LLaMA tokenizers ✅ (shipped in 0.20.0)
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
