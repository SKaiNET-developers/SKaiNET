# KLlama

A pure Kotlin LLaMA inference library for JVM, Native, JS, and WebAssembly. Part of the [SKaiNET](../../README.md) framework.

KLlama lets you load and run LLaMA-family models directly in your application — no Python, no native binaries, no external servers.

> **Early Stage Development**: The project is in active development. While it supports various formats and quantizations, you may encounter edge cases. We appreciate your feedback and bug reports!

Looking for the command-line interface? See the [KLlama CLI README](../skainet-kllama-cli/README.md).

---

## Supported Formats & Quantization

| Format | Description |
|---|---|
| **GGUF** | Full support with embedded tokenizer |
| **SafeTensors** | HuggingFace format (`model.safetensors` + `config.json` + `tokenizer.json`) |
| **Karpathy .bin** | Legacy llama2.c format |

**Quantization:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K). Quantized weights are dequantized on the fly for maximum cross-platform compatibility.

---

## Quick Start (Java)

Add the dependency (version managed by [BOM](../../docs/java-getting-started.md)):

```xml
<dependency>
    <groupId>sk.ainet</groupId>
    <artifactId>skainet-kllama-jvm</artifactId>
</dependency>
<dependency>
    <groupId>sk.ainet</groupId>
    <artifactId>skainet-backend-cpu-jvm</artifactId>
</dependency>
```

Load a GGUF model and generate text:

```java
import sk.ainet.apps.kllama.java.GenerationConfig;
import sk.ainet.apps.kllama.java.KLlamaJava;
import sk.ainet.apps.kllama.java.KLlamaSession;
import java.nio.file.Path;

public class Example {
    public static void main(String[] args) {
        try (KLlamaSession session = KLlamaJava.loadGGUF(Path.of("model.gguf"))) {
            GenerationConfig config = GenerationConfig.builder()
                    .maxTokens(128)
                    .temperature(0.7f)
                    .build();

            // Stream tokens to stdout as they are generated
            session.generate("The capital of France is", config,
                    token -> System.out.print(token));
            System.out.println();
        }
    }
}
```

Run with JVM flags (required for Vector API / SIMD):

```bash
java --enable-preview --add-modules jdk.incubator.vector -cp <classpath> Example
```

See the [Java LLM Inference Guide](../../docs/java-llm-inference.md) for streaming, async, SafeTensors, BERT, and agent/tool-calling examples. See [Building a Java CLI App](../../docs/java-cli-app.md) for a complete standalone project walkthrough.

---

## Embedding in Kotlin

### 1. Add Dependencies

```kotlin
dependencies {
    implementation("sk.ainet:skainet-kllama-jvm:0.15.0")
    implementation("sk.ainet:skainet-backend-cpu-jvm:0.15.0")
}
```

### 2. High-Level API

The simplest way from Kotlin is to use the Java facade — it handles context creation, weight loading, quantization dispatch, and tokenizer setup:

```kotlin
import sk.ainet.apps.kllama.java.KLlamaJava
import sk.ainet.apps.kllama.java.GenerationConfig
import java.nio.file.Path

fun main() {
    KLlamaJava.loadGGUF(Path.of("model.gguf")).use { session ->
        val config = GenerationConfig.builder()
            .maxTokens(128)
            .temperature(0.7f)
            .build()

        session.generate("The capital of France is", config) { token ->
            print(token)
        }
        println()
    }
}
```

### 3. Low-Level API

For full control over the loading pipeline (e.g., custom execution context, quantization policy, weight conversion):

```kotlin
import sk.ainet.apps.kllama.*
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.dequant.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    val modelPath = "path/to/model.gguf"
    val quantArena = Arena.ofShared()
    val memSegFactory = MemorySegmentTensorDataFactory()
    val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

    // 1. Load weights with quantization support
    val ingestion = LlamaIngestion<FP32>(
        ctx = ctx,
        dtype = FP32::class,
        config = LlamaLoadConfig(
            quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
            allowQuantized = true
        )
    )
    val rawWeights = ingestion.loadStreaming {
        JvmRandomAccessSource.open(modelPath)
    }
    val weights = if (rawWeights.quantTypes.isNotEmpty()) {
        MemSegWeightConverter.convert(rawWeights, ctx, quantArena)
    } else {
        rawWeights
    }

    // 2. Create runtime and tokenizer
    val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
    val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)
    val tokenizer = JvmRandomAccessSource.open(modelPath).use { source ->
        GGUFTokenizer.fromRandomAccessSource(source)
    }

    // 3. Generate text
    val prompt = "The capital of France is"
    val tokens = tokenizer.encode(prompt)

    runtime.generate(tokens, steps = 64, temperature = 0.8f) { tokenId ->
        print(tokenizer.decode(tokenId))
    }
    println()

    // 4. Cleanup
    quantArena.close()
    memSegFactory.close()
}
```

---

## Custom Backend Integration

KLlama is hardware-agnostic. You can inject your own attention and acceleration backends:

- **`AttentionBackend<T>`** — provides RoPE application and KV cache management for your hardware.
- **`GraphAccelerator<T>`** *(optional)* — provides fused operations (RMSNorm + QKV matmuls) to reduce synchronization overhead.

```kotlin
val customBackend = MyGpuAttentionBackend(ctx, weights)
val accelerator = MyGpuAccelerator(ctx)

val runtime = LlamaRuntime<FP32>(
    ctx = ctx,
    weights = weights,
    attentionBackend = customBackend,
    dtype = FP32::class,
    graphAccelerator = accelerator
)
```

---

## Performance

For best JVM performance, always run with Java Vector API flags:

```
--enable-preview --add-modules jdk.incubator.vector
```

This enables SIMD instructions for accelerated tensor operations.

---

## Reporting Bugs

1. **Model**: The exact model name and source (e.g., "TinyLlama-1.1B-Chat-v1.0-GGUF").
2. **Stack trace**: Full exception output if the app crashes.
3. **Environment**: OS, architecture (Intel/Apple Silicon), and Java version.

Report issues on our [GitHub Issue Tracker](https://github.com/anthropics/skainet/issues).
