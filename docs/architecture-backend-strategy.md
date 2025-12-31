# SKaiNET Backend Architecture Strategy

## Overview

SKaiNET uses a hybrid backend approach combining direct execution for development with MLIR/XLA compilation for production deployment.

## Architecture Layers

### Layer 1: Development Backend (CPU)
- **Purpose**: Fast iteration, testing, and debugging
- **Implementation**: `skainet-backend-cpu` with direct Kotlin implementations
- **Use Cases**: Unit tests, local development, CI/CD validation
- **Platforms**: All Kotlin Multiplatform targets (JVM, Native, JS, WASM)

### Layer 2: Production Compilation (MLIR-based)
- **Purpose**: High-performance production deployment
- **Implementation**: StableHLO MLIR → Multiple compiler backends
- **Compiler Options**: 
  - **XLA**: Google's mature compiler for datacenter deployment
  - **IREE**: Modern MLIR compiler optimized for edge and mobile deployment
- **Use Cases**: Production inference, training, edge deployment
- **Platforms**: Any MLIR-supported hardware (CPU, GPU, TPU, mobile accelerators)

## Why This Hybrid Approach?

### Direct CPU Backend Benefits
1. **Fast Development Cycle**: No compilation step needed for testing
2. **Multiplatform Support**: Runs on all Kotlin targets including JS/WASM
3. **Debugging**: Standard Kotlin debugging tools work directly
4. **Reference Implementation**: Validates correctness of MLIR compilation

### MLIR Compilation Benefits
1. **Hardware Portability**: Single compilation target for all hardware
2. **Automatic Optimization**: Advanced compiler optimizations and fusion
3. **Ecosystem Integration**: Compatible with JAX, TensorFlow, PyTorch (via ONNX)
4. **Future-Proof**: New hardware automatically supported via compiler updates

## XLA vs IREE: Choosing the Right Compiler

SKaiNET supports both XLA and IREE as MLIR compilation targets, each optimized for different deployment scenarios:

### XLA (Accelerated Linear Algebra)
**Best for**: Datacenter deployment, integration with Google ecosystem

**Strengths**:
- Mature and battle-tested in production (TensorFlow, JAX)
- Excellent optimization for large-scale training workloads
- Strong TPU support and Google Cloud integration
- Comprehensive operator coverage
- Stable API and extensive documentation

**Limitations**:
- Heavier runtime dependencies
- Less optimized for mobile/edge deployment
- Primarily Google-controlled development
- Limited customization for specialized hardware

### IREE (Intermediate Representation Execution Environment)
**Best for**: Edge deployment, mobile applications, standalone executables

**Strengths**:
- **Lightweight Runtime**: Minimal dependencies, suitable for embedded systems
- **Standalone Executables**: No heavy runtime required for deployment
- **Mobile-First Design**: Optimized for power and memory constraints
- **Flexible Backends**: CPU, GPU (Vulkan/Metal/CUDA), WebGPU, bare-metal
- **Modern Architecture**: Built from ground-up with lessons from XLA
- **Open Governance**: Part of OpenXLA with broader community input

**Advantages for SKaiNET**:
- **Multiplatform Alignment**: Better suited for Kotlin Multiplatform's diverse targets
- **WebGPU Support**: Enables high-performance browser execution for JS/WASM
- **Bare-Metal Support**: Can target embedded systems without OS
- **Smaller Footprint**: Critical for mobile and edge applications

## Compilation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     SKaiNET Kotlin DSL                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Compute Graph (DAG)                      │
└─────────┬───────────────────────────────────────┬───────────┘
          │                                       │
          │ Development Path                      │ Production Path
          ▼                                       ▼
┌─────────────────────┐              ┌──────────────────────────┐
│  CPU Backend        │              │  StableHLO Converter     │
│  Direct Execution   │              │  MLIR Generation         │
└─────────────────────┘              └──────────┬───────────────┘
                                                 │
                                                 ▼
                                     ┌──────────────────────────┐
                                     │    Compiler Choice       │
                                     │   XLA    │    IREE       │
                                     └─────┬────┴────┬──────────┘
                                           │         │
                                           ▼         ▼
                                   ┌──────────┐ ┌──────────────┐
                                   │   XLA    │ │    IREE      │
                                   │ Compiler │ │  Compiler    │
                                   └─────┬────┘ └──────┬───────┘
                                         │             │
                                         ▼             ▼
                                   ┌──────────┐ ┌──────────────┐
                                   │ Hardware │ │  Standalone  │
                                   │Executables│ │ Executables  │
                                   │CPU|GPU|TPU│ │CPU|GPU|WebGPU│
                                   └──────────┘ └──────────────┘
```

## Deployment Strategy by Use Case

### Datacenter & Cloud Deployment
**Recommended**: XLA compilation
```kotlin
val model = neuralNetwork { /* DSL */ }
val executable = model.toStableHlo().compileWithXla(
    target = XlaTarget.GPU_CUDA,
    optimization = OptimizationLevel.AGGRESSIVE
)
```

### Mobile & Edge Deployment  
**Recommended**: IREE compilation
```kotlin
val model = neuralNetwork { /* DSL */ }
val executable = model.toStableHlo().compileWithIree(
    target = IreeTarget.CPU_ARM64,
    optimization = OptimizationLevel.SIZE
)
```

### Web Applications
**Recommended**: IREE with WebGPU
```kotlin
val model = neuralNetwork { /* DSL */ }
val executable = model.toStableHlo().compileWithIree(
    target = IreeTarget.WEBGPU,
    optimization = OptimizationLevel.LATENCY
)
```

### Development & Testing
**Recommended**: CPU Backend
```kotlin
val model = neuralNetwork { /* DSL */ }
val result = model.execute(input, CpuBackend())
```

## When to Use Each Path

### Use CPU Backend When:
- Running unit tests
- Developing new operators
- Debugging model behavior
- Targeting JS/WASM platforms without WebGPU
- Quick prototyping without compilation overhead

### Use XLA Compilation When:
- Deploying to datacenter/cloud environments
- Targeting TPU hardware
- Integrating with TensorFlow/JAX ecosystems
- Large-scale training workloads
- Maximum performance on server-class hardware

### Use IREE Compilation When:
- Deploying to mobile devices (iOS/Android)
- Targeting edge devices with limited resources
- Building standalone applications
- Using WebGPU in browsers
- Deploying to embedded/bare-metal systems
- Optimizing for binary size and startup time

## Hardware Backend Support Comparison

| Target Platform | CPU Backend | XLA | IREE |
|----------------|-------------|-----|------|
| **CPU (x64/ARM)** | ✅ Direct | ✅ Optimized | ✅ Lightweight |
| **NVIDIA GPU** | ❌ | ✅ CUDA | ✅ CUDA/Vulkan |
| **AMD GPU** | ❌ | ✅ ROCm | ✅ ROCm/Vulkan |
| **Apple GPU** | ❌ | ❌ | ✅ Metal |
| **Intel GPU** | ❌ | ✅ Level Zero | ✅ Vulkan |
| **TPU** | ❌ | ✅ Native | ❌ |
| **WebGPU** | ❌ | ❌ | ✅ WGSL |
| **Mobile GPU** | ❌ | ❌ | ✅ Vulkan/Metal |
| **Bare Metal** | ✅ Limited | ❌ | ✅ LLVM |

## No Separate Hardware Backends Needed

Unlike frameworks that implement separate backends for each hardware target (CUDA, Metal, ROCm, etc.), SKaiNET relies on MLIR compilers for hardware targeting. This means:

- **No `skainet-backend-cuda`**: XLA/IREE handle NVIDIA GPU compilation
- **No `skainet-backend-metal`**: IREE handles Apple GPU compilation  
- **No `skainet-backend-rocm`**: XLA/IREE handle AMD GPU compilation
- **No `skainet-backend-vulkan`**: IREE handles cross-platform GPU via Vulkan

The only exception is the CPU backend, which serves as a reference implementation and development tool rather than a production execution engine.

## Implementation Roadmap

### Phase 1: StableHLO Foundation (Current)
- ✅ Basic StableHLO export for core operations
- ✅ Integration with existing ComputeGraph infrastructure
- 🔄 Comprehensive operation coverage (in progress)

### Phase 2: XLA Integration
- 🔄 XLA compiler integration and runtime
- 🔄 Performance benchmarking vs CPU backend
- ⏳ Production deployment tooling

### Phase 3: IREE Integration  
- ⏳ IREE compiler integration
- ⏳ Mobile-optimized compilation profiles
- ⏳ WebGPU backend for browser deployment
- ⏳ Bare-metal deployment for embedded systems

### Phase 4: Advanced Features
- ⏳ Dynamic shape support
- ⏳ Mixed precision compilation
- ⏳ Custom operator integration
- ⏳ Distributed execution support

## Migration Path

For existing code using the CPU backend:

```kotlin
// Development: Direct execution
val result = model.execute(input, CpuBackend())

// Production (Datacenter): Compile with XLA
val mlir = model.toStableHlo()
val executable = XlaCompiler.compile(mlir, target = XlaTarget.GPU_CUDA)
val result = executable.run(input)

// Production (Mobile): Compile with IREE
val mlir = model.toStableHlo()  
val executable = IreeCompiler.compile(mlir, target = IreeTarget.CPU_ARM64)
val result = executable.run(input)

// Web Deployment: IREE with WebGPU
val mlir = model.toStableHlo()
val executable = IreeCompiler.compile(mlir, target = IreeTarget.WEBGPU)
val result = executable.run(input)
```

## Performance Characteristics

### Compilation Time
- **CPU Backend**: Instant (no compilation)
- **XLA**: Moderate (optimized for throughput)
- **IREE**: Fast (optimized for deployment)

### Runtime Performance  
- **CPU Backend**: Good (reference implementation)
- **XLA**: Excellent (datacenter workloads)
- **IREE**: Excellent (mobile/edge workloads)

### Binary Size
- **CPU Backend**: Small (Kotlin bytecode)
- **XLA**: Large (includes runtime)
- **IREE**: Small (standalone executables)

### Memory Usage
- **CPU Backend**: Moderate (JVM overhead)
- **XLA**: High (optimization metadata)
- **IREE**: Low (minimal runtime)

## Future Considerations

### Potential Additional Direct Backends
- **WebGPU**: For browser-based GPU acceleration when IREE compilation overhead is too high
- **Custom Hardware**: For specialized accelerators without MLIR support

These would follow the same pattern as the CPU backend: direct implementation for specific use cases where MLIR compilation isn't suitable.

### Emerging Technologies
- **WebAssembly SIMD**: Enhanced performance for browser deployment
- **RISC-V**: Support for open hardware architectures
- **Neuromorphic Hardware**: Specialized AI chips with unique programming models

## References

- [IREE Project Homepage](https://iree.dev/)
- [IREE vs XLA Comparison](https://iree.dev/reference/glossary/#xla)
- [StableHLO Specification](https://github.com/openxla/stablehlo)
- [XLA Documentation](https://www.tensorflow.org/xla)
- [MLIR Documentation](https://mlir.llvm.org/)
- [HLO Getting Started Guide](./hlo-getting-started.md)
