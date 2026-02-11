# KLlama Getting Started Guide

Welcome to **KLlama**, the pure Kotlin LLaMA inference runtime. This guide will help you get started with the CLI and show you how to embed LLaMA models directly into your own Kotlin applications using **SKaiNET 0.11.0**.

⚠️ **Early Stage Development**: The whole project is in early development. While it supports various formats and quantizations, you may encounter edge cases. We appreciate your feedback and bug reports!

## 🚀 Quick Start with CLI

The CLI is the fastest way to test your LLaMA models on your machine.

### Building

To build the executable fat JAR, run the following command from the project root with Java 21+ as the default JDK:

```bash
./gradlew :skainet-apps:skainet-kllama-cli:shadowJar
```

The JAR will be located at: `skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar`

### Running

KLlama leverages the **Java Vector API** for high-performance CPU inference. You must enable it via JVM flags:

```bash
java --enable-preview --add-modules jdk.incubator.vector -jar skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar -m <model_path> [-t <tokenizer_path>] [-s <steps>] [-k <temperature>] [-p <system_prompt>] "<prompt>"
```

*   **`<model_path>`**: Path to `.gguf` or `.bin` (Karpathy) model.
*   **`[tokenizer_path]`**: Required for `.bin` models. Optional for `.gguf` if the tokenizer is embedded.
*   **`[steps]`**: Optional generation step count (default: `64`).
*   **`[temperature]`**: Optional sampling temperature (default: `0.8`).
*   **`[system_prompt]`**: Optional system prompt prepended as context.
*   **`"<prompt>"`**: Your text prompt.

**Example (GGUF with embedded tokenizer):**
```bash
java --enable-preview --add-modules jdk.incubator.vector -jar skainet-apps/skainet-kllama-cli/build/libs/kllama-all.jar -m tinyllama-1.1b-q4.gguf -s 96 -k 0.7 -p "You are concise" "Once upon a time"
```

---

### Running

```bash
./skainet-apps/skainet-kllama/build/bin/macosArm64/releaseExecutable/kllama.kexe <model_path> "<prompt>" [steps] [temperature] [--backend=cpu]
```

**Options:**
*   `--backend=mlx` - Use MLX GPU backend (default on macOS)
*   `--backend=cpu` - Use CPU backend
*   `--list-backends` - Show available backends and exit

**Example:**
```bash
# Fall back to CPU backend
./kllama.kexe tinyllama-1.1b.gguf "Once upon a time" --backend=cpu

# List available backends
./kllama.kexe --list-backends
# Output: Available: cpu
```

---

## 🛠 Embedding KLlama in Your App

You can easily integrate KLlama into any Kotlin project.

### 1. Add Dependencies

Add the following to your `build.gradle.kts` (ensure you are using version `0.11.0`):

```kotlin
dependencies {
    implementation("sk.ainet.apps:skainet-kllama:0.11.0")
    // For JVM SIMD support
    implementation("sk.ainet.core:skainet-backend-cpu:0.11.0")
}
```

### 2. Basic Usage

Here is a minimal example of how to load a GGUF model and generate text:

```kotlin
import sk.ainet.apps.kllama.*
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    val modelPath = "path/to/model.gguf"
    val ctx = DirectCpuExecutionContext()

    // 1. Load the model (supports GGUF quantization via on-the-fly dequantization)
    val ingestion = LlamaIngestion(ctx)
    val runtimeWeights = ingestion.loadStreaming {
        JvmRandomAccessSource.open(modelPath)
    }

    // 2. Initialize the runtime and tokenizer
    val runtime = LlamaRuntime(ctx, runtimeWeights)
    val tokenizer = JvmRandomAccessSource.open(modelPath).use { source ->
        GGUFTokenizer.fromRandomAccessSource(source)
    }

    // 3. Generate text
    val prompt = "The capital of France is"
    val promptTokens = tokenizer.encode(prompt)
    
    println("Response:")
    runtime.generate(promptTokens, steps = 32) { tokenId ->
        print(tokenizer.decode(tokenId))
    }
}
```

### 3. Custom Backend Integration (Advanced)

KLlama is hardware-agnostic by design. If you are developing a project that uses a specialized hardware backend (e.g., `mxl-backend`), you can inject your own implementations into the `LlamaRuntime`:

1.  **`AttentionBackend`**: Implement this interface to provide custom RoPE application and KV cache management for your hardware.
2.  **`GraphAccelerator`**: (Optional) Implement this to provide fused operations (like RMSNorm + QKV matmuls) to bypass individual operator calls and reduce synchronization overhead.

**Example: Injecting a custom backend**

```kotlin
val mxlCtx = MxlExecutionContext() 
val customBackend = MxlAttentionBackend(mxlCtx, weights)
val mxlAccelerator = MxlGraphAccelerator(mxlCtx)

val runtime = LlamaRuntime(
    ctx = mxlCtx,
    weights = runtimeWeights,
    attentionBackend = customBackend,
    dtype = FP32::class,
    graphAccelerator = mxlAccelerator 
)
```

---

## 📦 Supported Formats & Quantization

KLlama is designed to be flexible with model formats:

*   **GGUF**: Full support for loading GGUF models.
*   **Quantization**: Supports **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)**.
    *   *Note: Currently, quantized weights are dequantized to FP32 during loading for maximum compatibility across platforms.*
*   **Karpathy .bin**: Legacy support for llama2.c format.

---

## 🐛 Reporting Bugs

Since we are in early development, your bug reports are invaluable. If you encounter an issue:

1.  **Check the model**: Tell us the exact model name and where you downloaded it (e.g., "TinyLlama-1.1B-Chat-v1.0-GGUF").
2.  **Provide a Stack Trace**: If the app crashes, please include the full exception stack trace.
3.  **Environment**: Mention your OS, Architecture (Intel/Apple Silicon), and Java version.

Report issues on our [GitHub Issue Tracker](https://github.com/anthropics/skainet/issues).

---

## 💡 Pro Tip: Performance
For the best performance on JVM, always run with:
`--enable-preview --add-modules jdk.incubator.vector`
This enables SIMD instructions, which can significantly speed up tensor operations.
