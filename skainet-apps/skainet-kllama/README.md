# KLlama

**Pure Kotlin LLaMA Inference Runtime powered by SKaiNET**

KLlama is a Kotlin Multiplatform implementation of LLaMA model inference that runs on JVM, native platforms (macOS, Linux), and WebAssembly - all from a single codebase. No JNI bindings, no llama.cpp dependency, just pure Kotlin.

## What is KLlama?

KLlama enables you to run LLaMA-architecture language models directly in your Kotlin applications across multiple platforms. It's built on top of [SKaiNET](https://github.com/anthropics/skainet), a device-first AI framework for Kotlin Multiplatform.

### Key Features

- **Pure Kotlin** - No native bindings or external C/C++ dependencies
- **Cross-Platform** - Single codebase compiles to JVM, native binaries, and WASM
- **GGUF Support** - Load quantized models in GGUF format (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and K-quants)
- **Karpathy Format** - Compatible with llama2.c `.bin` checkpoint format
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
| Android | `android` | 🚧 Planned | AAR library |
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
kllama <model-path> <tokenizer-path> <prompt> [steps=64] [temperature=0.8]

Arguments:
  model-path      Path to model file (.gguf or .bin)
  tokenizer-path  Path to tokenizer.bin file
  prompt          Text prompt to start generation
  steps           Number of tokens to generate (default: 64)
  temperature     Sampling temperature, 0=greedy (default: 0.8)
```

### JVM

```bash
./gradlew :skainet-apps:skainet-kllama:jvmRun \
  --args="stories15m.bin tokenizer.bin 'Once upon a time' 64 0.8"
```

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

## Performance Notes

| Platform | Notes |
|----------|-------|
| JVM | Uses Vector API (incubator) for SIMD - fastest |
| Native | Pure Kotlin loops - good performance |
| WASM | Single-threaded browser execution - slowest |

Current limitations:
- All weights dequantized to FP32 at load time
- No GPU acceleration (CPU only)
- Single-token generation (no batching)

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
- [ ] Metal acceleration (Apple)
- [ ] Batch inference for prompt processing

## License

Part of the SKaiNET project. See repository root for license information.

## Links

- [SKaiNET](https://github.com/anthropics/skainet) - Parent framework
- [llama2.c](https://github.com/karpathy/llama2.c) - Inspiration and tokenizer source
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Model format specification
- [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) - Small test models
