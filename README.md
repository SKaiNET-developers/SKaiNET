[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/SKaiNET-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)

<img src="docs//SKaiNET-logo.png" alt="SKaiNET logo" width="150">

**SKaiNET** is an open-source deep learning framework written in Kotlin Multiplatform, designed with developers in mind to enable the creation modern AI powered applications with ease.

## Key features at a glance

### SKaiNET is Data

- Data loaders (MNIST, JSON datasets, image helpers)
- Type‑safe tensors across JVM/JS/Native

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

- Kotlin DSLs for Data, Neural Nets, and Pipelines

```kotlin
// Neural network DSL
val model = nn {
    input(28 * 28)
    dense(out = 128)
    relu()
    dense(out = 10)
}
```

### SKaiNET is Tools

- Kotlin Notebook support Explorer and Notebook-friendly APIs

```kotlin
// Works smoothly in Kotlin Notebooks
display(model.summary())
println(ds.describe())
```

### SKaiNET is Compiler

- MLIR/StableHLO based lowering (modules provided in `SKaiNET-compile-*`)

```kotlin
// Illustrative: export graph to JSON/StableHLO IR
val ir = Compile.toStableHlo(model)
println(ir.pretty())
```

### SKaiNET is for Developers

- Clean APIs, growing docs, Maven Central artifacts
- Get productive in minutes with minimal deps

```kotlin
dependencies {
    implementation("sk.ainet.core:SKaiNET-lang-core:0.5.0")
    implementation("sk.ainet.core:SKaiNET-backend-cpu:0.5.0")
}
// Ready to build & run in ~8 minutes
```

## Use it

- From Kotlin code in apps, libraries, CLIs
- In Kotlin Notebooks for quick exploration
- With sample projects to learn patterns

See also CHANGELOG for what’s new in 0.5.0.

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
    implementation("sk.ainet.core:SKaiNET-lang-core:0.5.0")
    implementation("sk.ainet.core:SKaiNET-backend-cpu:0.5.0")
    
    // simple model zoo
    implementation("sk.ainet.core:SKaiNET-lang-models:0.5.0")
    
    // Optional I/O (e.g., GGUF loader, JSON)
    implementation("sk.ainet.core:SKaiNET-io-core:0.5.0")
    implementation("sk.ainet.core:SKaiNET-io-gguf:0.5.0")
}
```

Maven:

```xml
<dependency>
  <groupId>sk.ainet.core</groupId>
  <artifactId>SKaiNET-lang-core</artifactId>
  <version>0.5.0</version>
</dependency>
```

## Samples and notebooks

- Sample app: https://github.com/sk-ai-net/SKaiNET-samples/tree/feature/MNIST/SinusApproximator
- Kotlin Notebook: https://github.com/sk-ai-net/SKaiNET-notebook

## Architecture

![Architecture diagram of SKaiNET compiler](docs//SKaiNET-compiler.svg)

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

See CHANGELOG.md for the full list.

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

## License

MIT — see LICENCE.
