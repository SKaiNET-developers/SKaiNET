# TurboQuant Metal Backend — Implementation Task

> Covers TQ-023 (Metal/Apple Silicon backend) and TQ-024 (Fused dequant+attention kernels)
> Status: TODO — requires Metal Shading Language + Kotlin/Native interop

---

## Objective

Implement TurboQuant KV-cache compression and decompression as Metal
compute shaders for Apple Silicon, enabling zero-copy unified-memory
KV cache and fused dequant+attention execution.

## Why Metal

- Apple Silicon unified memory eliminates CPU↔GPU copies for KV cache
- Metal Performance Shaders (MPS) provides optimized SDPA primitives
- Most on-device inference for SKaiNET targets macOS/iOS (Apple Silicon)
- TurboQuant decode is embarrassingly parallel — ideal for GPU compute

## Prerequisites

All prerequisites are complete:
- [x] TurboQuant encoding types (`TensorEncoding.TurboQuantPolar`, `TurboQuantPolarQjl`)
- [x] CPU reference kernels (rotation, quantize, bit-pack, QJL, codec)
- [x] `KvCacheStore` interface with `TurboQuantKvCacheStore`
- [x] `CompressedKvAttention` bridge with `RAW_STORAGE` extension point
- [x] `Placement` model with `DeviceKind.GPU`, `MemoryDomain.UNIFIED`
- [x] `BufferHandle.DeviceResident` for backend-managed buffers

## Scope

### In scope
- Metal compute shaders for TurboQuant encode/decode
- Fused dequant+SDPA Metal kernel
- Unified-memory KV cache (no CPU↔GPU copy)
- Kotlin/Native Metal interop for macOS/iOS targets
- Integration with existing `TensorOps.scaledDotProductAttention()`

### Out of scope
- General-purpose Metal backend for all TensorOps (separate effort)
- CUDA/Vulkan backends
- Training support (inference only)

---

## Architecture

### Module structure

```
skainet-backends/
  skainet-backend-metal/                   # New module
    build.gradle.kts                        # KMP config: macosArm64, iosArm64
    src/
      commonMain/kotlin/sk/ainet/exec/metal/
        MetalTurboQuantOps.kt               # Public API
        MetalKvCacheStore.kt                # Metal-backed KvCacheStore
        MetalBufferPool.kt                  # MTLBuffer lifecycle management
      nativeMain/kotlin/sk/ainet/exec/metal/
        MetalDevice.kt                      # MTLDevice + command queue wrapper
        MetalShaderLibrary.kt               # Compile & cache .metal shaders
        MetalBufferHandle.kt                # BufferHandle.DeviceResident for Metal
      nativeMain/resources/
        turboquant.metal                    # Metal compute shaders
      nativeTest/
        MetalTurboQuantOpsTest.kt           # Correctness vs CPU reference
```

### Key interfaces to implement

```kotlin
// MetalKvCacheStore: KvCacheStore backed by MTLBuffer in unified memory
class MetalKvCacheStore(
    config: KvCacheConfig,
    keyConfig: TurboQuantConfig,
    valueConfig: TurboQuantConfig,
    device: MetalDevice
) : KvCacheStore {
    // KV data lives in MTLBuffer (unified memory)
    // appendToken: GPU-side TurboQuant encode
    // readKeys/readValues: GPU-side decode or zero-copy raw access
}

// MetalTurboQuantOps: dispatch TurboQuant kernels to Metal GPU
class MetalTurboQuantOps(device: MetalDevice) {
    fun encode(input: MTLBuffer, config: TurboQuantConfig): MTLBuffer
    fun decode(encoded: MTLBuffer, config: TurboQuantConfig): MTLBuffer
    fun fusedDequantAttention(
        query: MTLBuffer, keyCache: MTLBuffer, valueCache: MTLBuffer,
        config: TurboQuantConfig, scale: Float
    ): MTLBuffer
}
```

### Integration with CompressedKvAttention

The `RAW_STORAGE` dequant strategy in `CompressedKvAttention` is the
extension point. The Metal backend:
1. Returns raw `TensorStorage` with `BufferHandle.DeviceResident` pointing to MTLBuffer
2. The Metal SDPA kernel reads compressed K/V directly and fuses dequant

```kotlin
// In MetalAttentionOps (extends or replaces scaledDotProductAttention)
override fun scaledDotProductAttention(query, key, value, mask, scale, causal): Tensor {
    val keyStorage = compressedKv.loadKeyStorageRaw(layer)
    if (keyStorage.buffer is BufferHandle.DeviceResident) {
        // Dispatch fused Metal kernel
        return metalOps.fusedDequantAttention(query, keyStorage, valueStorage, ...)
    }
    // Fallback to CPU
    return super.scaledDotProductAttention(query, key, value, mask, scale, causal)
}
```

---

## Metal Shaders

### File: `turboquant.metal`

```metal
// Required compute kernels:

// 1. turboquant_encode
//    Per-thread: rotate → quantize → pack one head's vector
//    Threadgroup: shared memory for Walsh-Hadamard butterfly
kernel void turboquant_encode(
    device const float* input          [[buffer(0)]],  // [numHeads, headDim]
    device uchar* packed_output        [[buffer(1)]],  // packed codes
    device half* scales_output         [[buffer(2)]],  // per-group scales
    constant TQParams& params          [[buffer(3)]],  // bits, headDim, seed
    uint tid                           [[thread_position_in_grid]]
);

// 2. turboquant_decode
//    Per-thread: unpack → dequantize → inverse rotate one head's vector
kernel void turboquant_decode(
    device const uchar* packed_input   [[buffer(0)]],
    device const half* scales_input    [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant TQParams& params          [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
);

// 3. turboquant_fused_sdpa (highest value kernel)
//    Fuses: KV dequant + Q@K^T scaling + softmax + @V
//    Avoids materializing decompressed K/V in global memory
kernel void turboquant_fused_sdpa(
    device const float* query          [[buffer(0)]],  // [nHeads, seqLen, headDim]
    device const uchar* key_packed     [[buffer(1)]],  // compressed keys
    device const half* key_scales      [[buffer(2)]],
    device const uchar* value_packed   [[buffer(3)]],  // compressed values
    device const half* value_scales    [[buffer(4)]],
    device float* output               [[buffer(5)]],  // [nHeads, seqLen, headDim]
    constant SDPAParams& params        [[buffer(6)]],
    uint2 tid                          [[thread_position_in_grid]],
    uint2 tgid                         [[threadgroup_position_in_grid]]
);

// 4. walsh_hadamard_transform
//    Threadgroup-cooperative WHT for rotation stage
//    Uses threadgroup memory for butterfly communication
kernel void walsh_hadamard_transform(
    device float* data                 [[buffer(0)]],
    constant uint& log2_n             [[buffer(1)]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup(0)]]
);
```

### Shader parameters

```metal
struct TQParams {
    uint bits;          // 2, 3, 4, or 8
    uint headDim;       // dimension per head
    uint numHeads;      // heads in this batch
    uint seed;          // rotation seed
    uint groupSize;     // quantization group size (32)
    bool useQjl;        // whether QJL residual is present
    uint residualBits;  // QJL residual bits (1-4)
};

struct SDPAParams {
    uint nHeads;
    uint nKVHeads;
    uint seqLen;
    uint kvLen;
    uint headDim;
    float scale;        // 1/sqrt(headDim)
    uint keyBits;
    uint valueBits;
    bool causal;
};
```

---

## Implementation Plan

### Phase 1: Metal infrastructure (no TurboQuant yet)

| Task | Description | Files |
|---|---|---|
| M-001 | Create `skainet-backend-metal` module | `build.gradle.kts`, `settings.gradle.kts` |
| M-002 | `MetalDevice` wrapper (MTLDevice, command queue) | `MetalDevice.kt` |
| M-003 | `MetalShaderLibrary` (compile .metal, cache pipelines) | `MetalShaderLibrary.kt` |
| M-004 | `MetalBufferHandle` → `BufferHandle.DeviceResident` | `MetalBufferHandle.kt` |
| M-005 | `MetalBufferPool` (reusable MTLBuffer pool) | `MetalBufferPool.kt` |
| M-006 | Kotlin/Native cinterop for Metal.framework | `metal.def`, build config |

### Phase 2: TurboQuant encode/decode shaders

| Task | Description | Files |
|---|---|---|
| M-010 | `turboquant_encode` shader | `turboquant.metal` |
| M-011 | `turboquant_decode` shader | `turboquant.metal` |
| M-012 | `walsh_hadamard_transform` cooperative shader | `turboquant.metal` |
| M-013 | `MetalTurboQuantOps` Kotlin dispatch | `MetalTurboQuantOps.kt` |
| M-014 | Correctness tests vs CPU reference | `MetalTurboQuantOpsTest.kt` |

### Phase 3: Metal KV cache store

| Task | Description | Files |
|---|---|---|
| M-020 | `MetalKvCacheStore` with unified-memory buffers | `MetalKvCacheStore.kt` |
| M-021 | GPU-side append (encode on GPU, no CPU round-trip) | shader + Kotlin |
| M-022 | GPU-side read (decode on GPU for raw access) | shader + Kotlin |
| M-023 | Integration with `CompressedKvAttention.RAW_STORAGE` | bridge code |

### Phase 4: Fused dequant+SDPA

| Task | Description | Files |
|---|---|---|
| M-030 | `turboquant_fused_sdpa` shader | `turboquant.metal` |
| M-031 | Tiled attention with on-the-fly dequant | shader optimization |
| M-032 | Causal mask support in fused kernel | shader |
| M-033 | GQA (grouped-query attention) support | shader |
| M-034 | End-to-end benchmark vs CPU decode+SDPA | benchmark suite |

### Phase 5: Integration & optimization

| Task | Description | Files |
|---|---|---|
| M-040 | Wire Metal backend into `PlatformCpuOpsFactory` for macOS/iOS | factory impl |
| M-041 | Fallback to CPU when Metal unavailable | graceful degradation |
| M-042 | Unified-memory placement resolution in `MemoryPlanner` | planner update |
| M-043 | `@KvCache(device = GPU)` annotation handling | annotation processor |
| M-044 | Performance tuning: threadgroup sizes, occupancy | shader tuning |

---

## Kotlin/Native Metal Interop

### cinterop definition (`metal.def`)

```
language = Objective-C
headers = Metal/Metal.h MetalPerformanceShaders/MetalPerformanceShaders.h
compilerOpts = -framework Metal -framework MetalPerformanceShaders
linkerOpts = -framework Metal -framework MetalPerformanceShaders -framework Foundation
```

### Key ObjC types to bridge

| Metal Type | Kotlin Usage |
|---|---|
| `MTLDevice` | GPU device handle |
| `MTLCommandQueue` | Serial command submission |
| `MTLCommandBuffer` | Batch of GPU commands |
| `MTLComputeCommandEncoder` | Dispatch compute kernels |
| `MTLBuffer` | GPU/unified memory buffer |
| `MTLComputePipelineState` | Compiled shader pipeline |
| `MTLLibrary` | Compiled shader library |

### Unified memory pattern

```kotlin
// Allocate in unified memory — accessible from both CPU and GPU
val buffer = device.newBuffer(
    length = sizeInBytes,
    options = MTLResourceStorageModeShared  // unified memory
)

// CPU can read/write directly (no copy needed)
val ptr = buffer.contents()

// GPU kernel reads/writes same memory
encoder.setBuffer(buffer, offset = 0, index = 0)
encoder.dispatchThreads(...)
```

---

## Performance Targets

| Metric | CPU Reference | Metal Target |
|---|---|---|
| TurboQuant encode (128d, 4-bit) | ~10 μs | < 2 μs |
| TurboQuant decode (128d, 4-bit) | ~8 μs | < 1 μs |
| Fused dequant+SDPA (8 heads, 128d, 1024 KV) | N/A (separate) | < 100 μs |
| KV cache memory (4-bit vs FP32) | 8x compression | 8x compression |
| CPU↔GPU copies for KV cache | N/A | 0 (unified memory) |

## Acceptance Criteria

- [ ] Metal shaders compile and run on Apple Silicon (M1+)
- [ ] Encode/decode correctness matches CPU reference within tolerance
- [ ] Fused dequant+SDPA produces correct attention output
- [ ] Zero CPU↔GPU copies for KV cache in unified memory mode
- [ ] Graceful fallback to CPU when Metal is unavailable
- [ ] Benchmark shows meaningful speedup over CPU reference path
- [ ] Works on both macOS (macosArm64) and iOS (iosArm64)

## References

- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [TurboQuant paper (arXiv)](https://arxiv.org/html/2504.19874v1)
- SKaiNET existing backend: `skainet-backends/skainet-backend-cpu/`
- SKaiNET CPU SIMD kernels: `JvmQuantizedVectorKernels.kt`, `JvmTurboQuantKernels.kt`
- SKaiNET TurboQuant reference: `skainet-lang/.../ops/turboquant/`
