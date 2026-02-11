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

## Model Download

```bash
# E2B model (~4GB Q8 quantized)
huggingface-cli download ggml-org/gemma-3n-E2B-it-GGUF \
    gemma-3n-E2B-it-Q8_0.gguf \
    --local-dir models/
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
- `skainet-llm` - Tokenizer interface
- `skainet-lang-core` - Tensor operations
- `skainet-compile-core` - Graph compilation
- `skainet-backend-cpu` - CPU execution context
- `skainet-io-gguf` - GGUF file parsing and weight loading

## Implementation Status

### Phase 1: Core Text Inference ✅
- [x] Module structure and build configuration
- [x] Gemma3n weight loader
- [x] Gemma3nConfig data class
- [x] Gemma3nRuntime with GELU activation
- [x] Hybrid attention backend
- [x] KV cache with layer sharing

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
