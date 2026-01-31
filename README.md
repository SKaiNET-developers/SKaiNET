[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)
[![GitHub Contributors](https://img.shields.io/github/contributors/SKaiNET-developers/SKaiNET)](https://github.com/SKaiNET-developers/SKaiNET/graphs/contributors)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-View%20Docs-blue?logo=readthedocs&logoColor=white)](https://deepwiki.com/SKaiNET-developers/SKaiNET)

<img src="docs/SKaiNET-logo.png" alt="SKaiNET logo" width="150">

**SKaiNET** is an open-source deep learning framework written in Kotlin, designed with developers in mind to enable the creation modern AI powered applications with ease. 

### 🌟 Vision
SKaiNET aims to democratize "Edge AI/ on device AI" by bridging the gap between high-level application development and low-level hardware optimization. We believe AI should be portable, type-safe, and developer-friendly, enabling seamless intelligence in everything from mobile apps to IoT devices without sacrificing performance.

> [!IMPORTANT]
> **About the name**
>
> “SKaiNET” is a **working project name** chosen early in the project’s life as part of a personal learning and experimentation effort, before any trademark considerations were known.
>
> The name is **not intended to reference, infringe, or imply association with any existing trademarks, companies, or products**. It is not a commercial brand and is **not claimed or assignable** to any company or organization that contributors may be affiliated with.
>
> If a naming conflict arises, the project name may be changed in the future.

### 🏗️ Architecture
SKaiNET uses a hybrid backend strategy that separates development iteration from production deployment.

![Architecture diagram of SKaiNET compiler](docs//SKaiNET-compiler.svg)

## Key features at a glance

### SKaiNET is Data

- **Built-in Data Loaders**: `MNIST`, `Fashion-MNIST`, `CIFAR-10`
- **I/O Formats**: `GGUF`, `ONNX`, `JSON`, `Image` (JPEG, PNG, etc.)
- **Transformation DSL**: Compose complex preprocessing pipelines including image resizing, normalization, and tensor conversion, using a type-safe Kotlin DSL.

```kotlin
// Data Transformation Pipeline
val transform = transforms<PlatformBitmapImage, Tensor<FP32, Float>> {
    resize(224, 224)
    centerCrop(200, 200)
    toTensor(ctx)
    normalize(ctx, mean = floatArrayOf(0.485f, 0.456f, 0.406f), std = floatArrayOf(0.229f, 0.224f, 0.225f))
}

val processedTensor = transform.apply(rawImage)
```

```kotlin
// data loaders
val ds = MNIST.load(train = true)
val (x, y) = ds.nextBatch(64)

// Type-safe tensor creation via tensor DSL
val ctx = DefaultNeuralNetworkExecutionContext()
val mask = data<FP32, Float>(ctx) {
    tensor{
        shape(3, 3) {
            from(
                1f, 0f, 0f,
                1f, 1f, 0f,
                1f, 1f, 1f,
            )
        }
    }
}

val t = tensor<FP32, Float>(ctx, FP32::class) {
    tensor {
        shape(2, 3) {
            from(
                0f, 1f, 2f,
                10f, 11f, 12f
            )
        }
    }
}
println("shape=${t.shape} first=${t.data[0,0]}")
```

### SKaiNET is Language

- Kotlin DSLs for Data, Neural Nets, Graphs, and Pipelines

#### Neural network DSL (Sequential)

```kotlin
val model = nn {
    input(28 * 28)
    dense(out = 128)
    relu()
    dense(out = 10)
}
```

#### Graph DSL (Functional/DAG)

For complex architectures with arbitrary wiring (like YOLO or ResNet), use the Graph DSL:

```kotlin
val program = dag {
    val x = input<FP32>("input", spec)
    val c1 = conv2d(x, w1, b1, padding = 1 to 1)
    val c2 = conv2d(c1, w2, b2, padding = 1 to 1)
    val sum = add(x, c2)
    output(relu(sum))
}
```

Read the [Graph DSL Documentation](docs/graph-dsl.md) for more details.

### SKaiNET is Tools

- Kotlin Notebook support Explorer and Notebook-friendly APIs

```kotlin
// Works smoothly in Kotlin Notebooks
display(model.summary())
println(ds.describe())
```

### SKaiNET is Compiler

- **MLIR/StableHLO Backend**: Lowering from high-level Kotlin DSL to MLIR StableHLO dialect.
- **Optimization Passes**: Extensible transformation API for optimizing the compiled IR.
    - `ConstantFoldingPass`: Folds arithmetic operations with constant operands.
    - `OperationFusionPass`: Fuses multiple ops (e.g., Add + ReLU) into efficient kernels.
    - `DeadCodeEliminationPass`: Removes unused computations.

```kotlin
// Applying Compiler Optimizations
val optimizer = StableHloOptimizer.createDefault()
val optimizedModule = optimizer.optimize(mlirModule)
```

- **Arduino C Code Generation**: Export models to standalone, optimized C99 code with static memory allocation.

```kotlin
// Export model to an Arduino library
val facade = CCodegenFacade()
facade.exportToArduinoLibrary(
    model = model,
    forwardPass = { ctx -> model.forward(input, ctx) },
    outputPath = "build/arduino",
    libraryName = "MyModel"
)
```

Read the [Deep Technical Explanation](docs/arduino-c-codegen.md) for more details.

### SKaiNET is for Developers

- Clean APIs, growing docs, Maven Central artifacts
- Get productive in minutes with minimal deps

```kotlin
dependencies {
    implementation("sk.ainet.core:SKaiNET-lang-core:0.9.2")
    implementation("sk.ainet.core:SKaiNET-backend-cpu:0.9.2")
}
// Ready to build & run in ~8 minutes
```

### SKaiNET is for Generative AI

Generate text with just a few lines of code using any Llama-based GGUF model:

```kotlin
val ctx = DirectCpuExecutionContext()
val ingestion = LlamaIngestion(ctx)

// Load model and tokenizer
val weights = ingestion.load { SystemFileSystem.source(Path("model.gguf")).buffered() }
val tokenizer = GGUFTokenizer.fromSource(SystemFileSystem.source(Path("model.gguf")).buffered())

// Generate!
val runtime = LlamaRuntime(ctx, weights)
runtime.generate(tokenizer.encode("Once upon a time"), steps = 64) { token ->
    print(tokenizer.decode(token))
}
```

## Use it

- From Kotlin code in apps, libraries, CLIs
- In Kotlin Notebooks for quick exploration
- With sample projects to learn patterns

## Quick start

Gradle (Kotlin DSL):

```kotlin
dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

dependencies {
    // minimal dependency with simple CPU backend
    implementation("sk.ainet.core:SKaiNET-lang-core:0.9.2")
    implementation("sk.ainet.core:SKaiNET-backend-cpu:0.9.2")

    // simple model zoo
    implementation("sk.ainet.core:SKaiNET-lang-models:0.9.2")

    // Optional I/O (e.g., GGUF loader, JSON)
    implementation("sk.ainet.core:SKaiNET-io-core:0.9.2")
    implementation("sk.ainet.core:SKaiNET-io-gguf:0.9.2")
}
```

Maven:

```xml
<dependency>
  <groupId>sk.ainet.core</groupId>
  <artifactId>SKaiNET-lang-core</artifactId>
  <version>0.9.2</version>
</dependency>
```

## Examples and notebooks

- [See examples](https://github.com/SKaiNET-developers/SKaiNET-examples)
- Kotlin Notebook: https://github.com/SKaiNET-developers/SKaiNET-notebook

## 0.9.2 highlights

- **SKaiNET for Generative AI**: Simplified API for text generation with Llama GGUF models.
- **Improved GGUF Loading**: Fixed critical bugs with column-major storage and added support for more quantization formats.
- **Better Tokenization**: Automatic detection of tokenizer strategies and improved UTF-8 decoding.
- **Runtime Fixes**: Fixed missing attention output projection in Llama models.

## 0.9.1 highlights

- **SafeTensors**: Native support for the SafeTensors format for secure and fast model loading.
- **Generalized Weight Loading**: Improved I/O pipeline with `WeightMapper` and progress tracking.
- **JVM Vector API**: Optimized tensor kernels for JVM using SIMD instructions.
- **Llama & GGUF**: Enhanced tokenizer and ingestion logic for Llama-based models.

```kotlin
// Example: Loading SafeTensors weights
val loader = SafeTensorsParametersLoader(ctx)
loader.load("model.safetensors", model)
```

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## 0.8.3 highlights (with tiny snippets)

- **KLlama (Llama 2 port)**: Initial version supporting GGUF models with `mmap` for zero-copy loading.
- **Quantization & BitNet**: Support for `Q8_0`, `Q4_K`, and BitNet/Ternary (`TQ1_0`, `TQ2_0`) formats.
- **Streaming & I/O**: Added streaming support for GGUF/ONNX and improved GGUF metadata loading.
- **Advanced Operations**: Added `LeakyReLU`, `ELU`, `AvgPool2d`, `Conv1d`, and `Conv3d`.
- **Optimizers & Metrics**: New `Adam`, `AdamW` optimizers and `Accuracy` metrics.
- **Datasets & Transforms**: Support for `CIFAR-10`, `Fashion-MNIST`, and a new `Data Transform API`.

```kotlin
// Example: Streaming inference with KLlama (GGUF)
val llama = KLlama.load("path/to/model.gguf")
llama.generate("Once upon a time") { token ->
    print(token) // streaming output
}
```

- **WASM/JS**: Initial support for web-based deployments.
- **GGUF-only**: Simplified I/O by focusing on GGUF (removed legacy formats).

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## 0.7.1 highlights (with tiny snippets)

- **Autograd Engine**: Initial support for automatic differentiation and reverse-mode gradients using `DefaultGradientTape`.
- **Optimization & Training**: New `SgdOptimizer` and training DSL to build and run training loops.
- **Loss Functions**: Added `MSELoss` and `CrossEntropyLoss` with configurable reduction strategies.

```kotlin
// Example training step with Autograd
val loss = MSELoss()
val optimizer = sgd(lr = 0.01)

val (tape, l) = record { loss.forward(model.forward(x, ctx), y, ctx) }
tape.computeGradients(targets = listOf(l), sources = model.parameters())
optimizer.step()
```

- Improved **Graph DSL** with better wiring and recording support.
- Stability improvements for StableHLO and CUDA backends.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## 0.6.3 highlights (with tiny snippets)

- StableHLO and CUDA support via IREE

```kotlin
// Compile model to StableHLO and run on CUDA
val ir = Compile.toStableHlo(model)
println(ir.pretty())
```

- Arduino C99 code generation

```kotlin
// Export model to an Arduino library
val facade = CCodegenFacade()
facade.exportToArduinoLibrary(
    model = model,
    forwardPass = { ctx -> model.forward(input, ctx) },
    outputPath = "build/arduino",
    libraryName = "MyModel"
)
```

- KSP-based TracingOps generation for recording pipelines.
- Improved HLO implementation and CUDA backend strategy.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## 0.5.0 highlights (with tiny snippets)

- Kolmogorov–Arnold Networks (KAN/AKN) preview in the NN DSL

```kotlin
val model = nn {
    input(64)
    dense(out = 64)
    // KAN layer (preview) with residual when dims match
    kanLayer(outputDim = 64, gridSize = 16, useResidual = true)
    dense(out = 10)
}
```

- Training/Eval phases made easy

```kotlin
val base = DefaultNeuralNetworkExecutionContext() // default = EVAL
val yTrain = train(base) { ctx -> model.forward(x, ctx) }
val yEval  = eval(base)  { ctx -> model.forward(x, ctx) }
```

- Dropout and BatchNorm layers

```kotlin
val y = x
    .let { dropout(p = 0.1).forward(it, ctx) }
    .let { batchNorm(numFeatures = 64).forward(it, ctx) }
```

- Conv2D + MaxPool in the NN DSL

```kotlin
val model = nn {
    conv2d(outChannels = 16, kernel = 3)
    maxPool2d(kernel = 2)
    dense(out = 10)
}
```

- Data API with MNIST loader and JSON dataset support

```kotlin
val ds = MNIST.load(train = true) // platform-aware loader
val (batchX, batchY) = ds.nextBatch(64)
```

- GGUF model loading (initial)

```kotlin
val gguf = GGUF.read("/path/to/model.gguf")
println("Tensors: ${gguf.tensors.size}")
```

- SIMD/Vector API acceleration on JVM; MatMul, tril, pooling ops; forward hooks and simple tape recording; unified tensor creation contexts; nested data blocks returning tensors.

## Experimental: Kolmogorov–Arnold Networks (KAN)

SKaiNET includes an initial KAN layer implementation that you can wire into the NN DSL. A KAN layer expands each input feature by a learnable grid of basis coefficients and then mixes them with a linear projection, with optional bias and residual connection.

- Current status: experimental/preview. API and behavior may change.
- Forward path uses broadcasted basis expansion and a matmul mixing step.
- `gridSize`, `useBias`, `useResidual`, and a custom `baseActivation` are supported. The `degree` parameter is reserved for future spline/basis functions and is not yet used.

Quick usage example:

```kotlin
val model = nn {
    input(64)
    dense(out = 64)
    // Add a KAN layer that keeps the same dimensionality and uses a residual connection
    kanLayer(outputDim = 64, gridSize = 16, useResidual = true)
    dense(out = 10)
}
```

Notes and limitations:
- Works with the default CPU backend; performance tuning and specialized kernels may arrive later.
- Residuals are applied only when `outputDim == inputDim`.
- You can customize initializers for the mixing weights, basis, and bias via the DSL block.

See source for details:
- SKaiNET-lang/SKaiNET-kan/src/commonMain/kotlin/sk/ainet/lang/kan/KanDsl.kt
- SKaiNET-lang/SKaiNET-kan/src/commonMain/kotlin/sk/ainet/lang/kan/KanLayer.kt

### 🚀 Sample Usage: Autograd
Minimize cosine distance between tensors with just a few lines:

```kotlin
skainet(ctx) {
    val a = tensor(1f, 0f, 0f).withRequiresGrad()
    val b = tensor(0f, 1f, 0f)

    // Record and compute gradients
    val (tape, distance) = record { a.cosineDistance(b) }
    tape.computeGradients(targets = listOf(distance), sources = listOf(a))

    // Optimize
    val optimizer = sgd(lr = 0.5)
    optimizer.addParameter(a)
    optimizer.step()
    
    println("Distance decreased to: ${a.cosineDistance(b).data.get()}")
}
```

### 🗺️ Roadmap
- **Q1 2026**: Full StableHLO coverage for Vision & NLP ops.
- **Q2 2026**: K2 Compiler Plugin for implicit context propagation.
- **Q3 2026**: IREE-based WebGPU backend for high-performance browser AI.
- **Q4 2026**: Federated learning support for multi-device training.

### 🤝 Join the Community
*   **GitHub Discussions**: [Ask questions & suggest features as issue](https://github.com/SKaiNET-developers/SKaiNET/issues)

### 🛠️ Contributing
We love contributions! Whether it's a new operator, documentation, or a bug fix:
1.  Read our [Contribution Guide](CONTRIBUTING.md).
2.  Check the [Good First Issues](https://github.com/SKaiNET-developers/SKaiNET/labels/good%20first%20issue).

## License

MIT — see LICENCE.
