# KBert Getting Started Guide

Welcome to **KBert**, the pure Kotlin BERT/Sentence-Transformer embedding runtime. This guide will help you get started with the CLI and show you how to embed BERT-based models directly into your Kotlin applications using **SKaiNET 0.11.0**.

⚠️ Early Stage Development: The project is under active development. While it already works for several common Sentence-Transformer/BERT layouts, there may be edge cases. Please report issues with exact model details and logs.

## 🚀 Quick Start with CLI

The CLI is the fastest way to generate embeddings and try similarity search on your machine.

### Building

From the project root, build the executable fat JAR (Java 21+ as default JDK):

```bash
./gradlew :skainet-apps:skainet-kbert-cli:shadowJar
```

The JAR will be located at: `skainet-apps/skainet-kbert-cli/build/libs/kbert-all.jar`

### Running

KBert leverages the Java Vector API for high‑performance CPU execution. Enable it via JVM flags:

```bash
java --enable-preview --add-modules jdk.incubator.vector \
  -jar skainet-apps/skainet-kbert-cli/build/libs/kbert-all.jar <model_dir> "query text" ["doc text"]
```

- `<model_dir>`: Directory containing `model.safetensors` (or any `*.safetensors`), `vocab.txt`, and optionally `config.json`.
- `"query text"`: The text to encode.
- `"doc text"` (optional): A second text. If provided, the CLI will also compute cosine similarity between the two embeddings.

Example:
```bash
java --enable-preview --add-modules jdk.incubator.vector \
  -jar skainet-apps/skainet-kbert-cli/build/libs/kbert-all.jar ./models/all-MiniLM-L6-v2 "The cat sits on the mat" "A feline rests on a rug"
```

Output includes embedding dimension, first few values, and (if second text provided) cosine similarity.

---

## 🛠 Embedding KBert in Your App

You can integrate KBert in any Kotlin project.

### 1. Add Dependencies

Add the following to your `build.gradle.kts` (ensure you are using version `0.11.0`):

```kotlin
dependencies {
    implementation("sk.ainet.apps:skainet-bert:0.11.0")
    // For CPU SIMD-accelerated ops
    implementation("sk.ainet.core:skainet-backend-cpu:0.11.0")
    // For loading SafeTensors files
    implementation("sk.ainet.io:skainet-io-safetensors:0.11.0")
}
```

### 2. Basic Usage (load BERT/Sentence-Transformer and encode)

```kotlin
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.bert.*
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32

fun main() = runBlocking {
    val modelDir = "./models/all-MiniLM-L6-v2"

    // 1) Tokenizer (expects vocab.txt in modelDir)
    val vocabTxt = java.nio.file.Files.readString(java.nio.file.Path.of(modelDir, "vocab.txt"))
    val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocabTxt)

    // 2) Detect/prepare config (config.json optional; reasonable defaults if missing)
    val config = MDBR_LEAF_IR_CONFIG // or parse from your config.json if present

    // 3) Load weights from .safetensors
    val ctx = DirectCpuExecutionContext()
    val ingestion = BertIngestion<FP32>(ctx, FP32::class, config)
    val safetensorsPath = java.nio.file.Path.of(modelDir, "model.safetensors")
    val loader = SafeTensorsParametersLoader(
        sourceProvider = { JvmRandomAccessSource.open(safetensorsPath.toString()) }
    )
    val weights = ingestion.load(loader)

    // 4) Create runtime
    val runtime = BertRuntime(ctx, weights, FP32::class)

    // 5) Encode text(s)
    val textA = "The capital of France is Paris."
    val metaA = tokenizer.encodeWithMetadata(textA)
    val embA = runtime.encode(metaA.inputIds, metaA.attentionMask, metaA.tokenTypeIds)

    val vecA = (embA.data.copyToFloatArray()).toList()
    println("Embedding dim: ${vecA.size}")
    println("First 8 values: ${vecA.take(8).joinToString(", ") { "%.6f".format(it) }}")
}
```

### 3. Pairwise similarity (optional)

```kotlin
val textB = "Paris is the capital city of France."
val metaB = tokenizer.encodeWithMetadata(textB)
val embB = runtime.encode(metaB.inputIds, metaB.attentionMask, metaB.tokenTypeIds)

fun cosine(a: List<Float>, b: List<Float>): Float {
    require(a.size == b.size)
    var dot = 0f; var na = 0f; var nb = 0f
    for (i in a.indices) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
    return if (denom > 0f) dot/denom else 0f
}

val sim = cosine(vecA, (embB.data.copyToFloatArray()).toList())
println("Cosine similarity: %.6f".format(sim))
```

---

## 📦 Supported Files & Model Layouts

- Weights: `*.safetensors` (e.g., `model.safetensors`, `pytorch_model.safetensors`).
- Tokenizer: `vocab.txt` (HuggingFace-style WordPiece/BERT vocab).
- Config: `config.json` is optional; defaults similar to MiniLM are used when absent. Projection heads used by many Sentence-Transformers are supported via `projectionDim` when present.

Notes:
- If multiple `*.safetensors` files exist, KBert will pick the first one unless you select explicitly.
- Sentence-Transformers models like `all-MiniLM-L6-v2` should work out of the box.

---

## 🐛 Reporting Bugs

If you encounter an issue:
1. Provide the exact model name and where you downloaded it (e.g., "sentence-transformers/all-MiniLM-L6-v2").
2. Include full stack traces if any crash occurs.
3. Mention OS, architecture (Intel/Apple Silicon), Java version, and whether you used the CLI or embedded API.

Report issues on our GitHub Issue Tracker: https://github.com/anthropics/skainet/issues

---

## 💡 Pro Tip: Performance

For best CPU performance on JVM, always run with:

```
--enable-preview --add-modules jdk.incubator.vector
```

This enables the Java Vector API (SIMD), significantly accelerating tensor operations.