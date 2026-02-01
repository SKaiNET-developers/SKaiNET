# KLlama

**Pure Kotlin LLaMA Inference Runtime powered by SKaiNET**

KLlama is a Kotlin Multiplatform implementation of LLaMA model inference that runs on JVM, native platforms (macOS, Linux), and WebAssembly - all from a single codebase. No JNI bindings, no llama.cpp dependency, just pure Kotlin.

## What is KLlama?

KLlama enables you to run LLaMA-architecture language models directly in your Kotlin applications across multiple platforms. It's built on top of [SKaiNET](https://github.com/anthropics/skainet), a device-first AI framework for Kotlin Multiplatform.

### Key Features

- **Pure Kotlin** - No native bindings or external C/C++ dependencies
- **Cross-Platform** - Single codebase compiles to JVM, native binaries, and WASM
- **Karpathy Format** - Full support for llama2.c `.bin` checkpoint format
- **GGUF Support** - Load quantized models in GGUF format (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and K-quants)
- **Streaming Generation** - Token-by-token output with callback API
- **KV Cache** - Efficient autoregressive generation with key-value caching

## Supported Platforms

| Platform | Target | Status | Output |
|----------|--------|--------|--------|
| JVM | `jvm` | ✅ Ready | JAR / Run with `java` |
| macOS (Apple Silicon) | `macosArm64` | ✅ Ready | Native binary |
| Linux (x64) | `linuxX64` | ✅ Ready | Native binary |
| Linux (ARM64) | `linuxArm64` | ✅ Ready | Native binary |
| Browser | `wasmJs` | ✅ Ready | WebAssembly |
| Android | `android` | ✅ Ready | AAR library |
| iOS | `iosArm64` | 🚧 Planned | Framework |

## Powered by SKaiNET

KLlama leverages SKaiNET's core components:

- **skainet-lang-core** - Type-safe tensor DSL and operations
- **skainet-backend-cpu** - CPU execution backend with SIMD support (JVM)
- **skainet-io-gguf** - GGUF model format parser and quantization handling

SKaiNET provides the foundational tensor operations, memory management, and cross-platform abstractions that make KLlama possible.

## Getting Models

### Recommended: Karpathy's TinyStories Models

Small models perfect for testing and development:

```bash
# Stories 15M (~60MB) - Fastest, good for testing
curl -L -o stories15m.bin \
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"

# Stories 42M (~167MB) - Better quality
curl -L -o stories42m.bin \
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin"

# Stories 110M (~438MB) - Best quality in this series
curl -L -o stories110m.bin \
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin"
```

### GGUF Models

For production use, GGUF models offer better compression via quantization:

```bash
# TinyLlama 1.1B Q4_K_M (~670MB)
curl -L -o tinyllama-1.1b-q4.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# TinyLlama 1.1B Q8_0 (~1.2GB) - Higher quality
curl -L -o tinyllama-1.1b-q8.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
```

### Model Sources

| Source | Models | Format | Notes |
|--------|--------|--------|-------|
| [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) | Stories 15M/42M/110M | `.bin` | Tiny models for testing |
| [TheBloke](https://huggingface.co/TheBloke) | Many LLaMA variants | `.gguf` | Quantized models |
| [Qwen](https://huggingface.co/Qwen) | Qwen2 0.5B-72B | `.gguf` | Official GGUF releases |

## Getting Tokenizers

### For Karpathy `.bin` Models

The tokenizer must match the model's vocabulary:

```bash
# Standard LLaMA tokenizer (32K vocab) - for stories15m/42m/110m
curl -L -o tokenizer.bin \
  "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"
```

### For GGUF Models

GGUF files embed their tokenizer metadata. KLlama extracts vocabulary from the GGUF file automatically, but you still need a tokenizer.bin for BPE encoding. Use the LLaMA tokenizer above, or convert from the model's original tokenizer.

## Building

### Prerequisites

- JDK 21 or later
- Gradle 8.x

### Build Commands

```bash
# Build all targets
./gradlew :skainet-apps:skainet-kllama:build

# Build optimized JVM Fat JAR (recommended for CLI)
./gradlew :skainet-apps:skainet-kllama:shadowJar -PbuildFatJar

# Build specific platforms
./gradlew :skainet-apps:skainet-kllama:jvmJar                           # JVM JAR
./gradlew :skainet-apps:skainet-kllama:linkReleaseExecutableMacosArm64  # macOS ARM64
./gradlew :skainet-apps:skainet-kllama:linkReleaseExecutableLinuxX64    # Linux x64
./gradlew :skainet-apps:skainet-kllama:linkReleaseExecutableLinuxArm64  # Linux ARM64
./gradlew :skainet-apps:skainet-kllama:wasmJsBrowserDistribution        # WASM/Browser
```

## Running

### Command Line Usage

```
kllama <model> <tokenizer> <prompt> [steps=64] [temperature=0.8]

Arguments:
  model           Path to model file (.gguf or .bin)
  tokenizer       Path to tokenizer.bin file (required for .bin, optional for .gguf)
  prompt          Text prompt to start generation
  steps           Number of tokens to generate (default: 64)
  temperature     Sampling temperature, 0=greedy (default: 0.8)
```

### JVM

#### Running via Gradle (Development)
```bash
# Running Karpathy .bin model
./gradlew :skainet-apps:skainet-kllama:jvmRun \
  --args="stories15m.bin tokenizer.bin 'Once upon a time' 64 0.8"

# Running GGUF model (embedded tokenizer)
./gradlew :skainet-apps:skainet-kllama:jvmRun \
  --args="tinyllama-1.1b-q4.gguf 'Once upon a time' 64 0.8"
```

#### Running the Fat JAR (Production)
For best performance, use the Fat JAR with SIMD (Vector API) enabled:
```bash
# Karpathy .bin model
java --add-modules jdk.incubator.vector --enable-preview -Xmx8g \
  -jar skainet-apps/skainet-kllama/build/libs/kllama-fat.jar \
  stories15m.bin tokenizer.bin "Once upon a time" 64 0.8

# GGUF model
java --add-modules jdk.incubator.vector --enable-preview -Xmx8g \
  -jar skainet-apps/skainet-kllama/build/libs/kllama-fat.jar \
  tinyllama-1.1b-q4.gguf "Once upon a time" 64 0.8
```
*Note: Adjust `-Xmx` (heap size) based on your model size (e.g., `-Xmx20g` for larger models).*

### macOS (Apple Silicon)

```bash
./skainet-apps/skainet-kllama/build/bin/macosArm64/releaseExecutable/kllama.kexe \
  stories15m.bin tokenizer.bin "Once upon a time" 64 0.8
```

### Linux

```bash
./skainet-apps/skainet-kllama/build/bin/linuxX64/releaseExecutable/kllama.kexe \
  stories15m.bin tokenizer.bin "Once upon a time" 64 0.8
```

### Browser (WASM)

```bash
# Build distribution
./gradlew :skainet-apps:skainet-kllama:wasmJsBrowserDistribution

# Serve locally
cd skainet-apps/skainet-kllama/build/dist/wasmJs/productionExecutable
python3 -m http.server 8080
# Open http://localhost:8080
```

For WASM, place model files in `src/wasmJsMain/resources/models/`.

## Example Output

```
$ ./kllama.kexe stories15m.bin tokenizer.bin "Once upon a time" 64 0.8
Loading model from stories15m.bin...
Loading tokenizer from tokenizer.bin...
Generating 64 tokens with temperature=0.8...
---
Once upon a time, there was a little girl named Lily. She loved to play
outside in the sunshine. One day, she found a beautiful butterfly in the
garden. The butterfly had pretty colors on its wings.
---
tok/s: 45.2
```

## Architecture

```
sk.ainet.apps.kllama/
├── LlamaRuntime.kt       # Core inference: attention, FFN, sampling
├── LlamaIngestion.kt     # Model loading facade
├── LlamaLoadConfig.kt    # Configuration for loading
├── Tokenizer.kt          # Tokenizer interface
├── TokenizerImpl.kt      # BPE tokenizer implementation
├── TokenizerUtils.kt     # Binary tokenizer loader
└── cli/
    └── Main.kt           # CLI entry point (per-platform)
```

### Runtime Components

- **RMSNorm** - Root Mean Square Layer Normalization
- **RoPE** - Rotary Position Embeddings
- **Multi-Head Attention** - With KV cache for efficient generation
- **SwiGLU FFN** - Feed-forward network with SiLU activation
- **Temperature Sampling** - Greedy or stochastic token selection

## Standalone Extraction

KLlama is designed to be extractable as a standalone application. The core dependencies are:

```kotlin
dependencies {
    implementation("sk.ainet:skainet-lang-core:$version")
    implementation("sk.ainet:skainet-backend-cpu:$version")
    implementation("sk.ainet:skainet-io-gguf:$version")
    implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.5.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
}
```

## Execution Backends

KLlama supports multiple execution backends for different performance profiles:

| Backend | Platform | Description |
|---------|----------|-------------|
| JVM CPU (Default) | JVM | Pure Kotlin loops, baseline CPU execution |
| JVM Vector API | JVM | SIMD-accelerated via Java Vector API (JDK 21+) |
| Native CPU | macOS/Linux | Kotlin/Native CPU execution |
| Native MLX | macOS ARM64 | GPU-accelerated via Apple MLX framework |

### Backend Selection

**JVM Vector API** (enabled by default on JDK 21+):
```bash
# Disable Vector API to use scalar CPU
export SKAINET_CPU_VECTOR_ENABLED=false
java -jar kllama-all.jar model.gguf "prompt"

# Enable Vector API (default)
export SKAINET_CPU_VECTOR_ENABLED=true
java --enable-preview --add-modules jdk.incubator.vector -jar kllama-all.jar model.gguf "prompt"
```

**Native MLX** (macOS Apple Silicon):
```bash
# Run with MLX GPU acceleration
./kllama.kexe model.gguf "prompt" --backend mlx

# Run with CPU only
./kllama.kexe model.gguf "prompt" --backend cpu
```

### MLX Backend Setup

The MLX backend provides GPU acceleration on Apple Silicon Macs using Apple's [MLX framework](https://github.com/ml-explore/mlx).

#### Prerequisites

1. **macOS with Apple Silicon** (M1/M2/M3/M4)
2. **MLX installed** via Homebrew or from source

#### Install MLX via Homebrew (Recommended)

```bash
brew install mlx
```

This installs MLX to `/opt/homebrew/opt/mlx` which is auto-detected.

#### Install MLX from Source

```bash
git clone https://github.com/ml-explore/mlx.git
cd mlx
export MLX_PREFIX="$HOME/.local/mlx"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${MLX_PREFIX}"
cmake --build build --config Release
cmake --install build
export MLX_ROOT=$MLX_PREFIX
```

#### Building with MLX

```bash
# Set MLX_ROOT if not using Homebrew default location
export MLX_ROOT=/opt/homebrew/opt/mlx

# Verify MLX is detected
ls $MLX_ROOT/include/mlx/mlx.h

# Build native executable with MLX support
./gradlew :skainet-apps:skainet-kllama:linkReleaseExecutableMacosArm64
```

## Performance Notes

| Platform | Notes |
|----------|-------|
| JVM Vector API | SIMD-accelerated, best JVM performance |
| JVM CPU | Scalar loops, baseline JVM |
| Native MLX | GPU-accelerated on Apple Silicon - fastest |
| Native CPU | Pure Kotlin loops |
| WASM | Single-threaded browser execution - slowest |

Current limitations:
- All weights dequantized to FP32 at load time
- Single-token generation (no batching)

## Benchmarking

A benchmark script is provided to compare performance across all backends:

```bash
# Run benchmark with your model
./benchmark.sh path/to/model.gguf

# Custom prompt and token count
./benchmark.sh model.gguf "Hello world" 128

# Skip rebuild (if already built)
SKIP_BUILD=true ./benchmark.sh model.gguf
```

### Sample Output

```
╔══════════════════════════════════════════════════════════════╗
║                    Benchmark Results                         ║
╚══════════════════════════════════════════════════════════════╝

Model: tinyllama-1.1b-q8.gguf
Prompt: "Once upon a time"
Steps: 64
Runs: 3

Backend                       tok/s         Speedup
------------------------- --------------- ---------------
JVM CPU (Scalar)                   12.5             1.0x
JVM Vector API                     28.3            2.26x
Native CPU                         15.2            1.22x
Native MLX                         89.4            7.15x

Benchmark complete!
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLX_ROOT` | Path to MLX installation | `/opt/homebrew/opt/mlx` |
| `SKIP_BUILD` | Skip build step if set to `true` | `false` |
| `SKAINET_CPU_VECTOR_ENABLED` | Enable JVM Vector API | `true` |

## Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| Large models may OOM on mobile | ⚠️ Limitation | Use smaller quantized models |
| WASM bundle size is large | ⚠️ Limitation | Use smaller models for browser |

**Tested working:**
- Karpathy `.bin` format (stories15m, stories42m, stories110m)
- GGUF F32 tensors
- GGUF quantized tensors (Q4_0, Q4_K, Q8_0, etc.)

## Roadmap

- [ ] Quantized inference (skip dequantization)
- [ ] Top-k / Top-p sampling
- [ ] Android app with UI
- [ ] iOS app with SwiftUI
- [x] MLX acceleration (Apple Silicon GPU)
- [ ] Batch inference for prompt processing

## License

Part of the SKaiNET project. See repository root for license information.

## Links

- [SKaiNET](https://github.com/anthropics/skainet) - Parent framework
- [llama2.c](https://github.com/karpathy/llama2.c) - Inspiration and tokenizer source
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Model format specification
- [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) - Small test models
