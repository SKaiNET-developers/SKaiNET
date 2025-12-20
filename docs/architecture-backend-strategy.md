# SKaiNET Backend Architecture Strategy

## Overview

SKaiNET uses a hybrid backend approach combining direct execution for development with MLIR/XLA compilation for production deployment.

## Architecture Layers

### Layer 1: Development Backend (CPU)
- **Purpose**: Fast iteration, testing, and debugging
- **Implementation**: `skainet-backend-cpu` with direct Kotlin implementations
- **Use Cases**: Unit tests, local development, CI/CD validation
- **Platforms**: All Kotlin Multiplatform targets (JVM, Native, JS, WASM)

### Layer 2: Production Compilation (MLIR/XLA)
- **Purpose**: High-performance production deployment
- **Implementation**: StableHLO MLIR → XLA compiler → hardware executables
- **Use Cases**: Production inference, training, edge deployment
- **Platforms**: Any XLA-supported hardware (CPU, GPU, TPU, mobile accelerators)

## Why This Hybrid Approach?

### Direct CPU Backend Benefits
1. **Fast Development Cycle**: No compilation step needed for testing
2. **Multiplatform Support**: Runs on all Kotlin targets including JS/WASM
3. **Debugging**: Standard Kotlin debugging tools work directly
4. **Reference Implementation**: Validates correctness of MLIR compilation

### MLIR/XLA Compilation Benefits
1. **Hardware Portability**: Single compilation target for all hardware
2. **Automatic Optimization**: XLA handles fusion, layout, and kernel selection
3. **Ecosystem Integration**: Compatible with JAX, TensorFlow, PyTorch (via ONNX)
4. **Future-Proof**: New hardware automatically supported via XLA updates

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
                                     │  XLA Compiler            │
                                     │  Hardware Optimization   │
                                     └──────────┬───────────────┘
                                                 │
                                                 ▼
                                     ┌──────────────────────────┐
                                     │  Hardware Executables    │
                                     │  CPU | GPU | TPU         │
                                     └──────────────────────────┘
```

## When to Use Each Path

### Use CPU Backend When:
- Running unit tests
- Developing new operators
- Debugging model behavior
- Targeting JS/WASM platforms
- Quick prototyping without compilation overhead

### Use MLIR/XLA When:
- Deploying to production
- Targeting GPU/TPU hardware
- Optimizing for performance
- Deploying to edge devices (Jetson, mobile)
- Integrating with other MLIR-based frameworks

## No Separate Hardware Backends Needed

Unlike frameworks that implement separate backends for each hardware target (CUDA, Metal, ROCm, etc.), SKaiNET relies on XLA for hardware targeting. This means:

- **No `skainet-backend-cuda`**: XLA handles NVIDIA GPU compilation
- **No `skainet-backend-metal`**: XLA handles Apple GPU compilation  
- **No `skainet-backend-rocm`**: XLA handles AMD GPU compilation

The only exception is the CPU backend, which serves as a reference implementation and development tool rather than a production execution engine.

## Migration Path

For existing code using the CPU backend:

```kotlin
// Development: Direct execution
val result = model.execute(input, CpuBackend())

// Production: Compile to MLIR, then use XLA runtime
val mlir = model.toStableHlo()
val executable = XlaCompiler.compile(mlir, target = GpuTarget.CUDA)
val result = executable.run(input)
```

## Future Considerations

### Potential Additional Direct Backends
- **WebGPU**: For browser-based GPU acceleration without compilation
- **Mobile GPU**: For iOS/Android when XLA overhead is too high
- **Custom Hardware**: For specialized accelerators without XLA support

These would follow the same pattern as the CPU backend: direct implementation for specific use cases where MLIR/XLA compilation isn't suitable.

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo)
- [XLA Documentation](https://www.tensorflow.org/xla)
- [MLIR Documentation](https://mlir.llvm.org/)
- [HLO Getting Started Guide](./hlo-getting-started.md)
