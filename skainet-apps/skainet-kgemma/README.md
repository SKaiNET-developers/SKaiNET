# skainet-kgemma

Kotlin Multiplatform runtime for Gemma 3n E2B multimodal models.

## Overview

`skainet-kgemma` provides a pure Kotlin implementation for running Google's Gemma 3n E2B model, following the existing `kllama` module pattern but with Gemma-specific architecture support.

## Key Features

- **Hybrid Attention**: 4 local sliding-window + 1 global full attention layers (repeating pattern)
- **GELU Activation**: Uses GELU instead of SiLU
- **Variable FFN Dimensions**: MatFormer architecture with per-layer FFN sizes
- **KV Cache Sharing**: Last 15 layers share KV cache to reduce memory
- **Dual RoPE Frequencies**: 10k for local attention, 1M for global attention
- **Multiplatform**: JVM, Android, Native (Linux/macOS), JavaScript, WebAssembly

## Architecture

```
Gemma 3n E2B:
- Hidden size: 2048
- Per-layer embedding: 256
- Layers: 35
- Attention heads: 8 (query) / 2 (kv)
- Head dimension: 256
- Vocabulary: 262400
- Context: 8192 tokens
- Sliding window: 512 tokens
```

## Usage

### Basic Inference

```kotlin
val ctx = CpuExecutionContext()
val ingestion = Gemma3nIngestion<FP32>(ctx, FP32::class)

// Load model
val runtime = ingestion.loadRuntimeStreaming {
    FileRandomAccessSource(File("gemma-3n-E2B-it-Q8_0.gguf"))
}

// Generate text
val tokenizer = GGUFTokenizer.fromRandomAccessSource(
    FileRandomAccessSource(File("gemma-3n-E2B-it-Q8_0.gguf"))
)

val prompt = tokenizer.encode("Hello, how are you?")
runtime.generate(prompt, steps = 100, temperature = 0.8f) { token ->
    print(tokenizer.decode(intArrayOf(token)))
}
```

### Configuration

```kotlin
// Custom config
val config = Gemma3nConfig(
    hiddenSize = 2048,
    numLayers = 35,
    numAttentionHeads = 8,
    numKvHeads = 2,
    headDim = 256,
    slidingWindow = 512,
    ropeBaseLocal = 10000f,
    ropeBaseGlobal = 1000000f,
    kvSharedLayers = 15
)
```

## CLI Tool

The `kgemma` CLI provides a simple way to run Gemma 3n models from the command line.

### Building

```bash
# Build JVM version
./gradlew :skainet-apps:skainet-kgemma:jvmJar

# Build native Linux executable
./gradlew :skainet-apps:skainet-kgemma:linuxX64Binaries

# Build native macOS executable
./gradlew :skainet-apps:skainet-kgemma:macosArm64Binaries
```

### Running

**JVM (recommended for development):**
```bash
./gradlew :skainet-apps:skainet-kgemma:jvmRun \
    --args='<model-path> "<prompt>" [steps] [temperature]'
```

**Native executable:**
```bash
./build/bin/linuxX64/releaseExecutable/kgemma \
    <model-path> "<prompt>" [steps] [temperature]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `model-path` | Yes | - | Path to model (GGUF file, SafeTensors directory, or index.json) |
| `prompt` | Yes | - | Text prompt to complete |
| `steps` | No | 64 | Number of tokens to generate |
| `temperature` | No | 0.8 | Sampling temperature (0.0 = greedy, higher = more random) |
| `--tokenizer=path` | No | auto | Path to tokenizer (GGUF file or auto-detected tokenizer.json) |

### Supported Model Formats

| Format | Path Example | Tokenizer |
|--------|--------------|-----------|
| GGUF | `model.gguf` | Embedded in GGUF |
| SafeTensors (directory) | `models/` | Auto-loads `tokenizer.json` from directory |
| SafeTensors (index) | `model.safetensors.index.json` | Auto-loads `tokenizer.json` from same directory |

### Examples

**GGUF model:**
```bash
./gradlew :skainet-apps:skainet-kgemma:jvmRun \
    --args='models/gemma-3n-E2B-it-Q8_0.gguf "Hello, how are you?" 32 0.7'
```

**HuggingFace SafeTensors model:**
```bash
# Download model
huggingface-cli download google/gemma-3n-E2B-it --local-dir models/

# Run (tokenizer.json auto-detected)
./gradlew :skainet-apps:skainet-kgemma:jvmRun \
    --args='models/ "The meaning of life is" 64'
```

**With explicit tokenizer:**
```bash
./gradlew :skainet-apps:skainet-kgemma:jvmRun \
    --args='models/ "Hello" 32 --tokenizer=tokenizer.gguf'
```

### Memory Requirements

| Model | Minimum RAM |
|-------|-------------|
| Gemma 3n E2B (FP32) | 16 GB |
| Gemma 3n E2B (Q8) | 8 GB |

The JVM version automatically allocates 24GB heap. For native builds, ensure sufficient system RAM.

## Model Download

```bash
# GGUF format (~4GB Q8 quantized)
huggingface-cli download ggml-org/gemma-3n-E2B-it-GGUF \
    gemma-3n-E2B-it-Q8_0.gguf \
    --local-dir models/

# SafeTensors format (original HuggingFace)
huggingface-cli download google/gemma-3n-E2B-it --local-dir models/
```

## Module Structure

```
skainet-kgemma/
├── src/commonMain/kotlin/sk/ainet/apps/kgemma/
│   ├── Gemma3nRuntime.kt           # Main inference runtime
│   ├── Gemma3nConfig.kt            # Model configuration
│   ├── Gemma3nIngestion.kt         # GGUF loading facade
│   ├── Gemma3nAttentionBackend.kt  # Hybrid attention implementation
│   ├── Gemma3nKvCache.kt           # KV cache with layer sharing
│   ├── AttentionBackend.kt         # Strategy interface
│   └── multimodal/
│       ├── VisionEncoder.kt        # MobileNetV5 vision encoder (planned)
│       └── AudioEncoder.kt         # USM audio encoder (planned)
└── build.gradle.kts
```

## Dependencies

The module depends on:
- `skainet-llm` - Tokenizer interface and HuggingFace BPE tokenizer
- `skainet-lang-core` - Tensor operations
- `skainet-compile-core` - Graph compilation
- `skainet-backend-cpu` - CPU execution context
- `skainet-io-gguf` - GGUF file parsing and weight loading
- `skainet-io-safetensors` - HuggingFace SafeTensors format support

## Implementation Status

### Phase 1: Core Text Inference ✅
- [x] Module structure and build configuration
- [x] Gemma3n weight loader (GGUF)
- [x] Gemma3nConfig data class
- [x] Gemma3nRuntime with GELU activation
- [x] Hybrid attention backend
- [x] KV cache with layer sharing
- [x] CLI tool with JVM and native targets

### Phase 1.5: HuggingFace Support ✅
- [x] SafeTensors weight loader (sharded)
- [x] HuggingFace config.json parser
- [x] HuggingFace tokenizer.json loader (BPE)
- [x] Auto-detection of model format

### Phase 2: Optimizations (Planned)
- [ ] Off-heap KV cache for JVM
- [ ] GPU attention backend
- [ ] Compiled graph acceleration

### Phase 3: Multimodal (Planned)
- [ ] Vision encoder (MobileNetV5)
- [ ] Audio encoder (USM)
- [ ] Soft token injection

## License

See the root project LICENSE file.
