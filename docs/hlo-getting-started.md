# Getting Started with HLO in SKaiNET

## What is HLO?

HLO (High-Level Operations) is SKaiNET's intermediate representation for neural network computations, based on [StableHLO](https://github.com/openxla/stablehlo) - the portable high-level operation set for machine learning. HLO serves as a bridge between SKaiNET's Kotlin DSL and various execution backends, enabling optimizations and cross-platform deployment.

## Why MLIR/XLA Instead of Direct Backends?

SKaiNET uses the MLIR/XLA compilation approach rather than implementing separate backends for each hardware target. This design choice provides several key advantages:

**Single Implementation Path**: Write operations once in Kotlin, compile to StableHLO MLIR, then let XLA handle hardware-specific optimizations. No need to maintain separate CUDA, Metal, or ROCm implementations.

**Automatic Optimization**: XLA provides sophisticated optimizations like operator fusion, memory layout optimization, and hardware-specific kernel selection without manual tuning.

**Future-Proof**: New hardware targets (like future GPU architectures) are automatically supported when XLA adds support, without requiring SKaiNET updates.

**Ecosystem Integration**: Full compatibility with JAX, TensorFlow, and other MLIR-based frameworks enables model sharing and toolchain reuse.

### Key Benefits

- **Portability**: Write once, compile to any XLA-supported hardware (CPU, GPU, TPU)
- **Optimization**: Leverage XLA's advanced compiler optimizations and operator fusion
- **Interoperability**: Full compatibility with XLA, JAX, TensorFlow, and MLIR ecosystems
- **Performance**: Hardware-specific optimizations without manual kernel development
- **No Backend Lock-in**: Single compilation target supports all hardware through XLA

## Architecture Overview

SKaiNET's HLO compilation pipeline transforms high-level Kotlin DSL operations into hardware-optimized executable code through the MLIR/XLA ecosystem:

```mermaid
graph TD
    A[Kotlin DSL] --> B[Compute Graph]
    B --> C[HLO Converter]
    C --> D[StableHLO MLIR]
    D --> E[XLA Compiler]
    E --> F[Hardware Executables]
    
    subgraph "SKaiNET Core"
        A
        B
        C
    end
    
    subgraph "MLIR/XLA Pipeline"
        D
        E
    end
    
    subgraph "Target Hardware"
        F --> G[CPU x86/ARM]
        F --> H[NVIDIA GPU]
        F --> I[AMD GPU]
        F --> J[Google TPU]
        F --> K[Mobile GPU]
    end
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style F fill:#e8f5e8
```

### Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Input Layer"
        DSL[Kotlin DSL Code]
        Tensor[Tensor Operations]
    end
    
    subgraph "Compilation Layer"
        DAG[Compute Graph DAG]
        Conv[HLO Converters]
        Opt[Optimization Passes]
    end
    
    subgraph "Execution Layer"
        MLIR[MLIR Representation]
        Backend[Target Backend]
        Runtime[Runtime Execution]
    end
    
    DSL --> DAG
    Tensor --> DAG
    DAG --> Conv
    Conv --> Opt
    Opt --> MLIR
    MLIR --> Backend
    Backend --> Runtime
    
    style DSL fill:#bbdefb
    style Conv fill:#c8e6c9
    style MLIR fill:#ffcdd2
```

## Building Blocks

### 1. HLO Converters

Converters transform SKaiNET operations into StableHLO operations:

- **MathOperationsConverter**: Basic arithmetic operations
- **LinalgOperationsConverter**: Linear algebra operations  
- **ActivationOperationsConverter**: Neural network activations
- **NeuralNetOperationsConverter**: High-level NN operations
- **ConstantOperationsConverter**: Constant value operations

### 2. Type System

HLO uses a strict type system for tensors:

```kotlin
// SKaiNET tensor type
Tensor<Float32, Shape4D> // Batch, Channel, Height, Width

// Converts to HLO type
tensor<1x3x224x224xf32> // StableHLO representation
```

### 3. Optimization Framework

The optimization pipeline includes:

- **Shape inference and propagation**
- **Constant folding and dead code elimination**
- **Operation fusion for performance**
- **Memory layout optimization**

## Practical Example: RGB to Grayscale Conversion

Let's walk through converting a color image tensor `Tensor<B,C,H,W>` to grayscale using matrix multiplication.

### Step 1: Define the Operation in Kotlin DSL

```kotlin
// From: skainet-lang/skainet-lang-models/src/commonMain/kotlin/sk/ainet/lang/model/compute/Rgb2GrayScaleMultiply.kt
fun Tensor<Float32, Shape4D>.rgb2GrayScaleMatMul(): Tensor<Float32, Shape4D> {
    // RGB to grayscale weights: [0.299, 0.587, 0.114]
    val grayWeights = constant(
        floatArrayOf(0.299f, 0.587f, 0.114f),
        Shape1D(3)
    ).reshape(Shape2D(3, 1))
    
    // Reshape input from [B,C,H,W] to [B,H,W,C] for matrix multiplication
    val reshaped = this.transpose(intArrayOf(0, 2, 3, 1))
    
    // Matrix multiply: [B,H,W,3] × [3,1] = [B,H,W,1]
    val gray = reshaped.matmul(grayWeights)
    
    // Reshape back to [B,1,H,W]
    return gray.transpose(intArrayOf(0, 3, 1, 2))
}
```

### Step 2: HLO Conversion Process

The conversion pipeline transforms this operation:

```mermaid
sequenceDiagram
    participant DSL as Kotlin DSL
    participant DAG as Compute Graph
    participant Conv as HLO Converter
    participant HLO as StableHLO IR
    participant Opt as Optimizer
    
    DSL->>DAG: rgb2GrayScaleMatMul()
    DAG->>Conv: MatMul + Transpose ops
    Conv->>HLO: stablehlo.dot_general
    Conv->>HLO: stablehlo.transpose
    HLO->>Opt: Unoptimized IR
    Opt->>HLO: Optimized IR
    
    Note over Conv,HLO: Type inference:<br/>tensor<BxCxHxWxf32> → tensor<Bx1xHxWxf32>
```

### Step 3: Generated StableHLO IR

The converter produces MLIR code like this:

```mlir
func.func @rgb2grayscale(%input: tensor<?x3x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // Define grayscale conversion weights
  %weights = stablehlo.constant dense<[[0.299], [0.587], [0.114]]> : tensor<3x1xf32>
  
  // Transpose input: [B,C,H,W] -> [B,H,W,C]
  %transposed = stablehlo.transpose %input, dims = [0, 2, 3, 1] : 
    (tensor<?x3x?x?xf32>) -> tensor<?x?x?x3xf32>
  
  // Matrix multiplication: [B,H,W,3] × [3,1] -> [B,H,W,1]
  %gray = stablehlo.dot_general %transposed, %weights, 
    contracting_dims = [3] x [0] : 
    (tensor<?x?x?x3xf32>, tensor<3x1xf32>) -> tensor<?x?x?x1xf32>
  
  // Transpose back: [B,H,W,1] -> [B,1,H,W]
  %result = stablehlo.transpose %gray, dims = [0, 3, 1, 2] : 
    (tensor<?x?x?x1xf32>) -> tensor<?x1x?x?xf32>
  
  return %result : tensor<?x1x?x?xf32>
}
```

## Hardware Target Compilation via XLA

SKaiNET uses the MLIR/XLA compilation pipeline to target different hardware platforms without requiring separate backend implementations. The StableHLO IR serves as a portable intermediate representation that XLA can compile to optimized code for various targets.

### Supported Hardware Targets

- **CPU**: x86_64, ARM64 (via XLA CPU backend)
- **GPU**: NVIDIA CUDA, AMD ROCm (via XLA GPU backend)  
- **TPU**: Google TPUs (via XLA TPU backend)
- **Mobile**: iOS Metal, Android GPU (via XLA mobile backends)

### Prerequisites for GPU Compilation

1. **XLA with GPU support**: [Installation guide](https://www.tensorflow.org/xla/tutorials/compile)
2. **NVIDIA CUDA Toolkit** (for NVIDIA GPUs): [Download here](https://developer.nvidia.com/cuda-downloads)
3. **ROCm** (for AMD GPUs): [Installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

### Step 1: Generate StableHLO IR

```bash
# Build SKaiNET HLO compiler
./gradlew :skainet-compile:skainet-compile-hlo:build

# Convert your model to StableHLO MLIR
./gradlew :skainet-compile:skainet-compile-hlo:generateHlo \
  -Pmodel=rgb2grayscale \
  -Poutput=rgb2grayscale.mlir
```

### Step 2: Compile with XLA for Target Hardware

```bash
# Compile to GPU executable (NVIDIA CUDA)
xla_compile \
  --input_format=mlir \
  --output_format=executable \
  --platform=gpu \
  --gpu_backend=cuda \
  --input_file=rgb2grayscale.mlir \
  --output_file=rgb2grayscale_cuda.so

# Compile to CPU executable  
xla_compile \
  --input_format=mlir \
  --output_format=executable \
  --platform=cpu \
  --input_file=rgb2grayscale.mlir \
  --output_file=rgb2grayscale_cpu.so

# Compile to TPU executable
xla_compile \
  --input_format=mlir \
  --output_format=executable \
  --platform=tpu \
  --input_file=rgb2grayscale.mlir \
  --output_file=rgb2grayscale_tpu.so
```

### Step 3: Runtime Execution

```bash
# Execute on target hardware using XLA runtime
xla_run \
  --executable=rgb2grayscale_cuda.so \
  --input=image.jpg \
  --output=gray.jpg \
  --device=gpu:0
```

### Jetson and Edge Device Deployment

For NVIDIA Jetson and other edge devices, the same MLIR → XLA compilation approach applies:

```bash
# Cross-compile for ARM64 with CUDA support
xla_compile \
  --input_format=mlir \
  --output_format=executable \
  --platform=gpu \
  --gpu_backend=cuda \
  --target_triple=aarch64-linux-gnu \
  --input_file=rgb2grayscale.mlir \
  --output_file=rgb2grayscale_jetson.so

# Deploy to Jetson device
scp rgb2grayscale_jetson.so jetson@192.168.1.100:~/models/

# Execute on Jetson
ssh jetson@192.168.1.100
cd ~/models
xla_run --executable=rgb2grayscale_jetson.so --device=gpu:0
```

## Advanced Topics

### Custom HLO Operations

Extend SKaiNET with custom operations:

```kotlin
// Define custom operation
@HloOperation("custom.rgb_enhance")
class RgbEnhanceOp : HloConverter {
    override fun convert(context: ConversionContext): String {
        return """
        %enhanced = custom_call @rgb_enhance(%input) : 
          (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xf32>
        """
    }
}
```

### Debugging HLO

Use SKaiNET's built-in debugging tools:

```kotlin
// Enable HLO debugging
val optimizer = StableHloOptimizer(debugMode = true)
val optimizedHlo = optimizer.optimize(hloModule)

// Visualize computation graph
optimizer.dumpGraphviz("rgb2gray.dot")
```

## Resources and References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [XLA Compilation Guide](https://www.tensorflow.org/xla)
- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [SKaiNET HLO Examples](./examples/hlo/)

## Next Steps

1. **Explore Examples**: Check `skainet-compile/skainet-compile-hlo/src/commonMain/kotlin/sk/ainet/compile/hlo/examples/`
2. **Run Tests**: Execute `./gradlew :skainet-compile:skainet-compile-hlo:test`
3. **Contribute**: Add new HLO converters for missing operations
4. **Optimize**: Profile and optimize your models using HLO tools

For more detailed information, see the [HLO Optimization Guide](./OPTIMIZATION.md) and [API Documentation](https://docs.skainet.sk/hlo/).