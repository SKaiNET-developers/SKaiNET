# KLlama Enterprise Roadmap

> Making KLlama the #1 choice for JVM, Kotlin, JS, iOS, and Android developers

## Executive Summary

KLlama is a Kotlin Multiplatform LLM inference runtime. This document outlines the path to making it the premier choice for developers across all platforms, directly competing with jlama (JVM) while offering unique multiplatform capabilities.

## Current State Analysis

### What We Have

| Feature | Status |
|---------|--------|
| Kotlin Multiplatform | ✅ JVM, Native, JS, WasmJS, iOS, Android |
| GGUF model loading | ✅ Full metadata + tensor parsing |
| Dequantization | ✅ F16, BF16, Q4_0, Q4_1, Q8_0, Q8_1, Q2-Q6_K, IQ4 |
| LLaMA architecture | ✅ With GQA support |
| Embedded tokenizer | ✅ BPE from GGUF |
| CLI interface | ✅ Native + JVM |
| Shadow JAR | ✅ Single fat JAR distribution |

### Competitive Comparison

| Feature | KLlama | jlama | llama.cpp |
|---------|--------|-------|-----------|
| Pure JVM/Kotlin | ✅ | ✅ | ❌ |
| Multiplatform | ✅ | ❌ | ❌ |
| iOS/Android native | ✅ | ❌ | Via bindings |
| Browser (Wasm) | ✅ | ❌ | Via bindings |
| Quantized inference | ✅ Q8_0, Q4_K | ✅ | ✅ |
| **BitNet/Ternary native** | ✅ TQ1_0/TQ2_0 dequant + ternary matmul | ❌ | Partial |
| SIMD optimization | ✅ JVM Vector API | ✅ | ✅ |
| Memory-mapped I/O | ✅ (JVM) | ✅ | ✅ |
| Multiple architectures | ❌ | ✅ | ✅ |
| GPU acceleration | ❌ | ❌ | ✅ |

### Key Gaps

1. ~~**Performance**: Dequantizes all weights to FP32, losing quantization benefits~~ ✅ Direct Q8_0/Q4_K matmul implemented
2. ~~**Memory**: Loads entire model into heap~~ ✅ JVM mmap + off-heap KV cache implemented
3. **API**: Low-level only, no chat/conversation abstractions
4. **Models**: LLaMA-only, missing Mistral/Phi/Gemma/Qwen
5. **Distribution**: Not on Maven Central, no package managers

---

## Core Architecture: Multi-Backend Storage Abstraction

**Goal**: Enable zero-copy loading and seamless GPU acceleration across all backends (CPU, MLX, Metal, WebGPU, CUDA)

### The Problem

Different compute backends have different memory models:

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory Landscape                         │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   CPU       │   MLX       │   Metal     │   WebGPU         │
├─────────────┼─────────────┼─────────────┼──────────────────┤
│ FloatArray  │ mlx.array   │ MTLBuffer   │ GPUBuffer        │
│ mmap OK ✓   │ unified mem │ GPU memory  │ GPU memory       │
│             │ mmap OK* ✓  │ needs copy  │ needs copy       │
└─────────────┴─────────────┴─────────────┴──────────────────┘

* MLX on Apple Silicon shares physical memory with CPU - zero copy possible!
```

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    High-Level API                            │
│         model.generate("Hello") → "Hello world!"            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tensor Operations                         │
│              matmul, softmax, attention, etc.               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Compute Backend                           │
│         CpuBackend, MlxBackend, MetalBackend, etc.          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tensor Storage                            │
│     HostStorage (mmap), DeviceStorage (GPU), Unified        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Source                               │
│              MappedGGUF, Network, Generated                  │
└─────────────────────────────────────────────────────────────┘
```

### Tensor Data Abstraction

Separate data representation from operations, enabling zero-copy and pluggable backends:

```kotlin
/**
 * Abstract tensor data storage - decoupled from compute operations.
 * Enables zero-copy mmap, quantized storage, and lazy loading.
 */
sealed interface TensorData<T> {
    val size: Int
    operator fun get(index: Int): T
    fun slice(offset: Int, length: Int): TensorData<T>
    fun toArray(): Array<T>  // Only when truly needed
}

/** Heap-backed data (current implementation) */
class HeapFloatData(private val array: FloatArray) : TensorData<Float> {
    override val size = array.size
    override fun get(index: Int) = array[index]
    override fun slice(offset: Int, length: Int) =
        HeapFloatData(array.copyOfRange(offset, offset + length))
}

/** Memory-mapped data - zero copy! */
class MappedFloatData(
    private val buffer: MappedByteBuffer,
    private val offset: Long,
    override val size: Int
) : TensorData<Float> {

    override fun get(index: Int): Float {
        // Direct access to mapped memory - no copy
        return buffer.getFloat((offset + index * 4).toInt())
    }

    override fun slice(offset: Int, length: Int): TensorData<Float> {
        // Just a view - no copy!
        return MappedFloatData(buffer, this.offset + offset * 4, length)
    }
}

/** Quantized data - dequantizes on access */
class QuantizedData(
    private val buffer: MappedByteBuffer,
    private val offset: Long,
    override val size: Int,
    private val quantType: GGMLQuantizationType
) : TensorData<Float> {

    override fun get(index: Int): Float = dequantizeAt(index)

    // For bulk operations - dequantize a block at a time
    fun getBlock(blockIndex: Int, output: FloatArray) {
        dequantizeBlock(blockIndex, output)
    }
}
```

### Storage Abstraction

```kotlin
/**
 * Where tensor data physically lives - decoupled from how it's accessed.
 */
sealed interface TensorStorage {
    val sizeBytes: Long
    val device: Device

    // Transfer to another device (may be zero-copy!)
    fun transferTo(target: Device): TensorStorage
}

sealed interface Device {
    object CPU : Device
    data class Metal(val deviceId: Int) : Device
    data class MLX(val stream: Any?) : Device
    data class WebGPU(val adapter: Any) : Device
    data class CUDA(val deviceId: Int) : Device
}

/** Host memory - can be heap or mmap */
class HostStorage(
    val memory: HostMemory,
    override val sizeBytes: Long
) : TensorStorage {
    override val device = Device.CPU

    override fun transferTo(target: Device): TensorStorage = when (target) {
        Device.CPU -> this  // Already there
        is Device.MLX -> MlxStorage.fromHost(this)  // Zero-copy on Apple Silicon!
        is Device.Metal -> MetalStorage.fromHost(this)
        is Device.WebGPU -> WebGPUStorage.fromHost(this)
        is Device.CUDA -> CudaStorage.fromHost(this)
    }
}

/** MLX unified memory storage */
class MlxStorage(
    val mlxArray: Any,
    override val sizeBytes: Long
) : TensorStorage {
    override val device = Device.MLX()

    companion object {
        fun fromHost(host: HostStorage): MlxStorage {
            // On Apple Silicon: potentially zero-copy!
            return if (host.memory is MappedMemory) {
                createFromMappedMemory(host.memory)  // No copy
            } else {
                createFromHeap(host.memory)  // Must copy
            }
        }
    }
}
```

### Compute Backend Interface

```kotlin
/**
 * Backend that executes tensor operations.
 * Each backend knows how to handle its storage type optimally.
 */
interface ComputeBackend {
    val device: Device

    // Create storage appropriate for this backend
    fun allocate(shape: Shape, dtype: DType): TensorStorage

    // Import from host (may be zero-copy)
    fun import(host: HostStorage, hint: ImportHint = ImportHint.DEFAULT): TensorStorage

    // Core operations
    fun matmul(a: Tensor, b: Tensor): Tensor
    fun add(a: Tensor, b: Tensor): Tensor
    fun softmax(x: Tensor, axis: Int): Tensor
    fun rmsNorm(x: Tensor, weight: Tensor, eps: Float): Tensor

    // Quantized operations (backend decides optimal strategy)
    fun matmulQuantized(input: Tensor, weights: QuantizedStorage): Tensor
}

enum class ImportHint {
    DEFAULT,          // Backend decides
    PREFER_ZERO_COPY, // Keep in host memory if possible
    PREFER_DEVICE,    // Copy to device memory
    LAZY              // Transfer on first use
}
```

### Backend Implementations

```kotlin
/** Pure CPU backend with mmap support */
class CpuBackend : ComputeBackend {
    override val device = Device.CPU

    override fun import(host: HostStorage, hint: ImportHint): TensorStorage {
        return host  // CPU can always use host storage directly - true zero copy!
    }

    override fun matmulQuantized(input: Tensor, weights: QuantizedStorage): Tensor {
        // Block-wise dequantization from mmap'd data
        return quantizedMatmulCpu(input, weights)
    }
}

/** MLX backend for Apple Silicon */
class MlxBackend : ComputeBackend {
    override val device = Device.MLX()

    override fun import(host: HostStorage, hint: ImportHint): TensorStorage {
        return when {
            // Apple Silicon unified memory: zero-copy possible!
            host.memory is MappedMemory && isAppleSilicon() -> {
                MlxStorage.zeroCopyFromMapped(host.memory)
            }
            hint == ImportHint.LAZY -> LazyMlxStorage(host)
            else -> MlxStorage.copyFrom(host)
        }
    }
}

/** Metal backend for explicit GPU control */
class MetalBackend(val metalDevice: MTLDevice) : ComputeBackend {
    override fun import(host: HostStorage, hint: ImportHint): TensorStorage {
        return when {
            isAppleSilicon() && host.memory is MappedMemory -> {
                // Shared storage mode - GPU accesses host memory directly
                val buffer = metalDevice.makeBuffer(
                    bytesNoCopy = host.memory.pointer,
                    length = host.sizeBytes,
                    options = .storageModeShared
                )
                MetalStorage(buffer)
            }
            else -> {
                // Copy to GPU private memory
                metalDevice.makeBuffer(bytes = host.toByteArray())
            }
        }
    }
}
```

### Zero-Copy Flow on Apple Silicon

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Apple Silicon (M1/M2/M3/M4)                       │
│                     Unified Memory Architecture                       │
│                                                                       │
│   ┌─────────────┐                      ┌─────────────────────────┐   │
│   │  GGUF File  │──── mmap() ────────▶ │   Physical Memory       │   │
│   │   (SSD)     │                      │   ┌─────────────────┐   │   │
│   └─────────────┘                      │   │  Tensor Data    │   │   │
│                                        │   │  (weights)      │   │   │
│                                        │   └────────┬────────┘   │   │
│                                        │            │            │   │
│   ┌─────────────┐                      │     ┌──────┴──────┐     │   │
│   │  CPU Cores  │◀─────────────────────┼─────│  Shared     │     │   │
│   └─────────────┘     same memory!     │     │  View       │     │   │
│                                        │     └──────┬──────┘     │   │
│   ┌─────────────┐                      │            │            │   │
│   │  GPU Cores  │◀─────────────────────┼────────────┘            │   │
│   │  (MLX/Metal)│     zero copy!       │                         │   │
│   └─────────────┘                      └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

Result: Load 70B model, GPU inference, ~0 bytes copied!
```

### Platform-Specific Mmap Implementation

```kotlin
// Common interface
expect class MappedMemory(path: Path) : Closeable {
    val size: Long
    fun getFloat(offset: Long): Float
    fun getPointer(): Pointer  // For zero-copy GPU import
}

// JVM
actual class MappedMemory(path: Path) : Closeable {
    private val channel = FileChannel.open(path, READ)
    private val buffer = channel.map(READ_ONLY, 0, channel.size())

    actual fun getFloat(offset: Long) = buffer.getFloat(offset.toInt())
}

// Native (macOS/Linux/iOS)
actual class MappedMemory(path: String) : Closeable {
    private val fd = open(path, O_RDONLY)
    private val size = lseek(fd, 0, SEEK_END)
    private val ptr = mmap(null, size, PROT_READ, MAP_PRIVATE, fd, 0)!!

    actual fun getFloat(offset: Long): Float {
        return (ptr + offset)!!.reinterpret<FloatVar>().pointed.value
    }

    actual fun getPointer() = ptr  // Direct pointer for Metal/MLX
}

// Wasm/JS - chunked loading (no true mmap, but similar API)
actual class MappedMemory(arrayBuffer: ArrayBuffer) : Closeable {
    private val view = DataView(arrayBuffer)
    actual fun getFloat(offset: Long) = view.getFloat32(offset.toInt(), littleEndian = true)
}
```

### Backend Compatibility Matrix

| Feature | CPU | MLX | Metal | WebGPU | CUDA |
|---------|-----|-----|-------|--------|------|
| Mmap zero-copy load | ✅ | ✅* | ✅* | ❌ | ❌ |
| Unified memory | N/A | ✅ | ✅* | ❌ | ❌ |
| Lazy transfer | ✅ | ✅ | ✅ | ✅ | ✅ |
| Quantized matmul | ✅ | ✅ | ✅ | ✅ | ✅ |
| Block dequant | ✅ | ✅ | ✅ | ✅ | ✅ |

*On Apple Silicon only

### Key Benefits

| Aspect | Heap (Current) | Mmap + Backend Abstraction |
|--------|---------------|----------------------------|
| Loading 7B model | ~14GB heap | ~0 heap (OS manages) |
| Tensor slicing | Copy data | View (zero-copy) |
| Memory pressure | OOM crash | OS pages out unused |
| GPU transfer | Always copy | Zero-copy on unified memory |
| Backend switch | Rewrite code | Change one line |

### Usage Example

```kotlin
// Load with optimal backend for platform
val backend = ComputeBackend.optimal()  // Auto-detects MLX, Metal, CPU, etc.
val model = KLlama.load("llama3-70b-q4.gguf", backend)

// On Apple Silicon with MLX: literally zero copies from disk to GPU!
model.generate("Hello")

// Explicit backend selection
val cpuModel = KLlama.load("model.gguf", CpuBackend())
val mlxModel = KLlama.load("model.gguf", MlxBackend())
val metalModel = KLlama.load("model.gguf", MetalBackend())
```

### Integration with Quantized Inference

The real power: mmap + backend abstraction + block-wise dequantization:

```kotlin
// Quantized matmul - never fully dequantizes!
fun ComputeBackend.matmulQuantized(
    input: Tensor,              // Small: [1, dim] on device
    weights: QuantizedStorage,  // Large: mmap'd Q4_K blocks
    output: Tensor              // Small: [1, out_dim] on device
) {
    for (block in 0 until weights.blockCount) {
        // Dequantize one block (32-256 values) at a time
        val dequantized = scratchBuffer.slice(0, blockSize)
        weights.dequantizeBlock(block, dequantized)

        // Backend-specific optimized partial matmul
        accumulateMatmul(input, dequantized, output, block)
    }
}
```

**Result**: Run 70B models in 16GB RAM with GPU acceleration.

---

## Phase 1: Performance Foundation

**Goal**: Match jlama performance, enable 7B+ models

### 1.1 Memory-Mapped GGUF Loading ✅ COMPLETED

Replace heap-based loading with memory-mapped files for efficient large model handling.

**Implementation**: `MmapLlamaLoader`, `MmapFloatTensorData`, `MmapTensorSource` (JVM)

```kotlin
// Current (problematic for large models)
val bytes = source.readByteArray(tensorSize) // OOM risk

// Target
class MappedGGUFReader(path: Path) {
    private val channel: FileChannel
    private val mappedBuffer: MappedByteBuffer

    fun getTensorSlice(offset: Long, size: Long): FloatBuffer
}
```

**Platform implementations**:
- JVM: `FileChannel.map()` with `MappedByteBuffer`
- Native: `mmap()` system call
- iOS: `mmap()` via Darwin
- Android: `MemoryFile` or `mmap()`
- JS/Wasm: `fetch()` with streaming + chunked processing

**Impact**: Enable 7B, 13B, 70B models without OOM

### 1.2 BitNet / Ternary Quantization Support ✅ IMPLEMENTED

Native support for Microsoft's BitNet 1.58-bit models with ternary weights {-1, 0, +1}.

**Why BitNet First?**
- Simplest quantization format (no scales, no block structure)
- Fastest inference (addition only, no FP multiply)
- Unique differentiator (most frameworks don't have native ternary kernels)
- We already have `Ternary` DType and `DenseTernaryTensorArray`

**What's Implemented:**
- ✅ TQ1_0 dequantization (base-3 packed ternary format, ~1.69 bpw)
- ✅ TQ2_0 dequantization (2-bit packed ternary format, ~2.06 bpw)
- ✅ `Ternary2BitTensorData` - compact storage with TQ format encoding
- ✅ `TernaryMatmul.matmul()` - addition-only kernel (no FP multiply)
- ✅ `matmulAutoDispatch()` - automatic ternary detection and dispatch
- ✅ Comprehensive unit tests for all components

**Architecture Integration:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Module.forward() calls ctx.ops.matmul(input, weights)          │
│                              │                                   │
│                              ▼                                   │
│  TensorOps.matmul() dispatches based on weights.data type:      │
│    ├─ FloatArrayTensorData  → matmulFP32()                      │
│    ├─ TernaryTensorData     → matmulTernary()  ← BitNet!        │
│    └─ Q4KTensorData         → matmulQ4K()      ← Quantized      │
│                              │                                   │
│  Higher layers unchanged - transparent dispatch!                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Tasks:**

1. **TQ1_0 Loader** - Parse GGUF ternary format into `TernaryTensorData`
```kotlin
fun loadTQ1_0(bytes: ByteArray, shape: Shape): TernaryTensorData {
    // TQ1_0: 5 ternary values packed per byte using base-3 encoding
    // Decode and store in DenseTernaryTensorArray (2-bit per value)
}
```

2. **TernaryTensorData** - New TensorData implementation
```kotlin
class TernaryTensorData<T : DType>(
    override val shape: Shape,
    val packed: DenseTernaryTensorArray  // 2-bit packed storage
) : TensorData<T, Byte> {
    // Implements get/set for ternary values
}
```

3. **Ternary Matmul Kernel** - The killer feature!
```kotlin
// In DefaultCpuOps
fun matmulTernary(input: Tensor<FP32>, weights: TernaryTensorData): Tensor<FP32> {
    val output = FloatArray(outDim)
    for (i in 0 until outDim) {
        var sum = 0f
        for (j in 0 until inDim) {
            when (weights[i, j]) {
                +1 -> sum += input[j]   // Just ADD
                -1 -> sum -= input[j]   // Just SUBTRACT
                 0 -> { }                // Skip (sparse!)
            }
        }
        output[i] = sum
    }
    return output
}
```

4. **SIMD Ternary Kernel** (JVM Vector API)
```kotlin
// Pack ternary as masks, use vector blend operations
fun matmulTernarySIMD(input: FloatArray, weights: TernaryTensorData): FloatArray {
    val posMask = weights.getPositiveMask()  // Where weight = +1
    val negMask = weights.getNegativeMask()  // Where weight = -1
    // Vector masked add/subtract
}
```

**Implementation Status:**
| Component | Status | Location |
|-----------|--------|----------|
| `Ternary` DType | ✅ | `skainet-lang-core/.../types/Ternary.kt` |
| `DenseTernaryTensorArray` | ✅ | `skainet-lang-core/.../data/dense/` |
| GGUF TQ1_0/TQ2_0 enum | ✅ | `GGMLQuantizationType` |
| Type promotion | ✅ | Ternary → Int8 → FP32 |
| `dequantTQ1_0()` | ✅ | `LlamaWeightLoader.kt` |
| `dequantTQ2_0()` | ✅ | `LlamaWeightLoader.kt` |
| `Ternary2BitTensorData` | ✅ | `skainet-lang-core/.../data/TernaryTensorData.kt` |
| `TernaryMatmul` | ✅ | `skainet-lang-core/.../ops/TernaryMatmul.kt` |
| Unit tests | ✅ | `LlamaQuantDequantTest`, `TernaryTensorDataTest`, `TernaryMatmulTest` |

**Impact**:
- **Speed**: 5-10x faster than FP32 (no FP multiply, integer add only)
- **Memory**: 16x smaller than FP32 (2-bit vs 32-bit)
- **Unique**: First pure-Kotlin BitNet inference engine

---

### 1.3 Standard Quantized Inference Kernels ✅ COMPLETED

Native quantized matmul without dequantizing entire tensors.

**Implementation:**
- `Q8_0TensorData` / `Q8_0BlockTensorData` - Block-wise Q8_0 storage (34 bytes/block)
- `Q4_KTensorData` / `Q4_KBlockTensorData` - Block-wise Q4_K storage (144 bytes/block)
- `QuantizedMatmul.matmulQ8_0()` - Scalar Q8_0 matmul kernel
- `QuantizedMatmul.matmulQ4_K()` - Scalar Q4_K matmul kernel
- `QuantizedTensorFactory` - Convert raw GGUF bytes to quantized tensor data

**Key Files:**
- `skainet-lang-core/.../tensor/data/Q8_0TensorData.kt`
- `skainet-lang-core/.../tensor/data/Q4_KTensorData.kt`
- `skainet-lang-core/.../tensor/ops/QuantizedMatmul.kt`
- `skainet-io-gguf/.../llama/QuantizedTensorFactory.kt`

**Tests:** `Q8_0TensorDataTest`, `Q4_KTensorDataTest`, `QuantizedMatmulTest`

**Impact**: Direct quantized inference without full FP32 dequantization

### 1.4 JVM Vector API Integration ✅ COMPLETED

Leverages `jdk.incubator.vector` for SIMD operations on quantized data.

**Implementation:**
- `JvmQuantizedVectorKernels` - SIMD kernels for Q8_0 and Q4_K matmul
- `dotQ8_0Block()` - Vectorized dot product for Q8_0 blocks
- `dotQ4_KSubBlock()` - Vectorized dot product for Q4_K sub-blocks
- `matmulQ8_0Vec()` / `matmulQ4_KVec()` - Full vectorized matmul
- Integration with `DefaultCpuOpsJvm.chooseQuantizedMatmul()`

**Key Files:**
- `skainet-backend-cpu/src/jvmMain/.../ops/JvmQuantizedVectorKernels.kt`
- `skainet-backend-cpu/src/jvmMain/.../ops/DefaultCpuOpsJvm.kt`

**Impact**: SIMD-accelerated quantized inference on JVM

### 1.5 KV-Cache Optimization ✅ COMPLETED

Off-heap KV-cache for reduced GC pressure and better memory management.

**Implementation:**
- `KvCache` interface - Platform-independent cache abstraction
- `HeapKvCache` - FloatArray-based implementation (all platforms)
- `OffheapKvCache` - Direct ByteBuffer implementation (JVM)
- `createOptimalKvCache()` - Platform-specific factory (expect/actual)
- `LlamaRuntime` integration with pluggable KvCache

**Key Files:**
- `skainet-kllama/src/commonMain/.../KvCache.kt` - Interface + HeapKvCache
- `skainet-kllama/src/jvmMain/.../OffheapKvCache.kt` - Off-heap implementation
- `skainet-kllama/src/jvmMain/.../KvCacheJvm.kt` - JVM factory
- `skainet-kllama/src/*/KvCache*.kt` - Platform implementations (Native, Wasm, JS, Android)

**Tests:** `LlamaRuntimeQuantizedTest` - OffheapKvCache integration, heap vs offheap parity

**Impact**: Reduced GC pressure, longer context support, platform-optimized caching

---

### Phase 1 Summary ✅

All Phase 1 "Performance Foundation" items are complete:

| Component | Status | Key Files |
|-----------|--------|-----------|
| Memory-mapped GGUF | ✅ | `MmapLlamaLoader`, `MmapTensorSource` |
| BitNet/Ternary | ✅ | `TernaryTensorData`, `TernaryMatmul` |
| Q8_0 Inference | ✅ | `Q8_0TensorData`, `QuantizedMatmul` |
| Q4_K Inference | ✅ | `Q4_KTensorData`, `QuantizedMatmul` |
| JVM SIMD | ✅ | `JvmQuantizedVectorKernels` |
| Off-heap KV Cache | ✅ | `KvCache`, `OffheapKvCache` |

**Test Coverage:** Unit tests + integration tests (`LlamaRuntimeQuantizedTest`)

---

## Phase 2: Developer Experience

**Goal**: Make KLlama the easiest LLM library to use

### 2.1 High-Level Chat API

```kotlin
// Simple usage
val model = KLlama.load("llama3-8b-q4.gguf")

val response = model.generate("What is Kotlin?")
println(response)

// Chat with history
val chat = model.chat {
    system("You are a helpful coding assistant")
    user("How do I read a file in Kotlin?")
}
println(chat.response)

// Continue conversation
chat.user("Now show me how to write to it")
println(chat.response)
```

### 2.2 Streaming API with Kotlin Flow

```kotlin
// Streaming generation
model.generateFlow("Tell me a story")
    .collect { token -> print(token) }

// Chat streaming
model.chatFlow {
    system("You are a storyteller")
    user("Tell me about a brave knight")
}.collect { chunk ->
    print(chunk)
}

// With cancellation
val job = scope.launch {
    model.generateFlow(prompt).collect { ... }
}
job.cancel() // Stops generation
```

### 2.3 Chat Templates

Parse and apply chat templates from GGUF metadata.

```kotlin
enum class ChatTemplate {
    LLAMA2,      // [INST] {user} [/INST]
    LLAMA3,      // <|start_header_id|>user<|end_header_id|>
    CHATML,      // <|im_start|>user\n
    MISTRAL,     // [INST] {user} [/INST]
    PHI3,        // <|user|>\n
    GEMMA,       // <start_of_turn>user\n
    ZEPHYR,      // <|user|>\n
}

class ChatFormatter(template: ChatTemplate) {
    fun format(messages: List<ChatMessage>): String
    fun parseStopTokens(): List<String>
}

// Auto-detect from GGUF
val formatter = ChatFormatter.fromGGUF(model.metadata)
```

### 2.4 Structured Output

Grammar-based constrained generation for JSON output.

```kotlin
// JSON mode
val response = model.generate(
    prompt = "Extract: John is 30 years old",
    responseFormat = ResponseFormat.json(
        schema = """{"name": "string", "age": "number"}"""
    )
)
// Returns: {"name": "John", "age": 30}

// Enum constraint
val sentiment = model.generate(
    prompt = "Sentiment of: I love this!",
    responseFormat = ResponseFormat.enum("positive", "negative", "neutral")
)
// Returns: "positive"
```

### 2.5 Tool/Function Calling

```kotlin
val tools = listOf(
    Tool("get_weather") {
        description = "Get current weather"
        parameter("location", "string", "City name")
        parameter("unit", "string", "celsius or fahrenheit", optional = true)
    },
    Tool("search") {
        description = "Search the web"
        parameter("query", "string", "Search query")
    }
)

val result = model.chat {
    system("You have access to tools. Use them when needed.")
    user("What's the weather in Tokyo?")
    tools(tools)
}

when (result) {
    is ToolCall -> {
        val weather = getWeather(result.arguments["location"]!!)
        result.respond(weather)
    }
    is TextResponse -> println(result.text)
}
```

---

## Phase 3: Model Architecture Support

**Goal**: Support all popular model families

### 3.1 Architecture Registry

```kotlin
interface ModelArchitecture {
    val name: String
    fun createRuntime(weights: RuntimeWeights): LLMRuntime
    fun validateMetadata(metadata: ModelMetadata): Boolean
}

object ArchitectureRegistry {
    private val architectures = mutableMapOf<String, ModelArchitecture>()

    init {
        register(LlamaArchitecture())
        register(MistralArchitecture())
        register(Phi3Architecture())
        register(GemmaArchitecture())
        register(QwenArchitecture())
    }

    fun detect(metadata: GGUFMetadata): ModelArchitecture
}
```

### 3.2 Supported Architectures

| Architecture | Models | Key Differences |
|--------------|--------|-----------------|
| **LLaMA** | LLaMA 1/2/3, CodeLlama, Vicuna | Baseline |
| **Mistral** | Mistral 7B, Mixtral 8x7B | Sliding window attention, MoE |
| **Phi** | Phi-2, Phi-3 | Partial rotation, different FFN |
| **Gemma** | Gemma 1/2 | GeGLU activation, RMSNorm variations |
| **Qwen** | Qwen 1.5/2 | Different RoPE, bias in attention |
| **StarCoder** | StarCoder 1/2 | MQA, different positional encoding |

### 3.3 Mixture of Experts (MoE)

For Mixtral and similar models:

```kotlin
class MoELayer(
    val numExperts: Int,
    val numActiveExperts: Int,  // typically 2
    val router: Router,
    val experts: List<FFNLayer>
) {
    fun forward(x: Tensor): Tensor {
        val routerLogits = router(x)
        val topK = routerLogits.topK(numActiveExperts)
        val weights = softmax(topK.values)

        var output = zeros(x.shape)
        for ((idx, weight) in topK.indices.zip(weights)) {
            output += weight * experts[idx](x)
        }
        return output
    }
}
```

---

## Phase 4: Platform Acceleration

**Goal**: GPU and hardware acceleration across platforms

### 4.1 Metal/MPS (macOS/iOS)

Critical for Apple ecosystem adoption.

```kotlin
// Expect/actual pattern
expect class MetalAccelerator {
    fun matmul(a: Tensor, b: Tensor): Tensor
    fun attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor?): Tensor
}

// iOS/macOS implementation using Metal Performance Shaders
actual class MetalAccelerator {
    private val device: MTLDevice
    private val commandQueue: MTLCommandQueue

    actual fun matmul(a: Tensor, b: Tensor): Tensor {
        // Use MPSMatrixMultiplication
    }
}
```

### 4.2 WebGPU (Browser)

Essential for JS/Wasm deployment.

```kotlin
// Wasm/JS implementation
actual class WebGPUAccelerator {
    private val device: GPUDevice
    private val matmulPipeline: GPUComputePipeline

    actual fun matmul(a: Tensor, b: Tensor): Tensor {
        // WebGPU compute shader
    }
}
```

### 4.3 Android NNAPI

Hardware acceleration on Android devices.

```kotlin
// Android implementation
actual class NNAPIAccelerator {
    private val model: ANeuralNetworksModel
    private val compilation: ANeuralNetworksCompilation

    fun compileModel(weights: RuntimeWeights)
    fun execute(input: Tensor): Tensor
}
```

### 4.4 Acceleration Matrix

| Platform | CPU SIMD | GPU |
|----------|----------|-----|
| JVM | Vector API | - |
| macOS | NEON | Metal |
| iOS | NEON | Metal |
| Linux x64 | AVX2/AVX-512 | Vulkan (future) |
| Linux ARM | NEON | - |
| Android | NEON | NNAPI |
| Browser | Wasm SIMD | WebGPU |

---

## Phase 5: Distribution & Ecosystem

**Goal**: Easy adoption for all developers

### 5.1 Maven Central

```xml
<!-- JVM -->
<dependency>
    <groupId>ai.skainet</groupId>
    <artifactId>kllama-jvm</artifactId>
    <version>1.0.0</version>
</dependency>
```

```kotlin
// Kotlin Multiplatform
kotlin {
    sourceSets {
        commonMain {
            dependencies {
                implementation("ai.skainet:kllama:1.0.0")
            }
        }
    }
}
```

### 5.2 Platform Package Managers

```ruby
# CocoaPods (iOS/macOS)
pod 'KLlama', '~> 1.0'
```

```swift
// Swift Package Manager
.package(url: "https://github.com/skainet/kllama-swift", from: "1.0.0")
```

```bash
# npm (JS/Browser)
npm install @skainet/kllama
```

### 5.3 Documentation

- **Getting Started Guide** - 5-minute quickstart
- **API Reference** - Full KDoc documentation
- **Tutorials** - Common use cases
- **Architecture Guide** - For contributors
- **Benchmark Suite** - Performance comparisons
- **Model Zoo** - Tested model configurations

### 5.4 Examples Repository

```
kllama-examples/
├── jvm-console/           # Simple JVM CLI
├── android-chat/          # Android chat app
├── ios-assistant/         # iOS app with SwiftUI
├── compose-multiplatform/ # Desktop + Android
├── react-browser/         # React + Wasm
├── spring-boot-api/       # REST API server
├── ktor-streaming/        # Streaming API server
└── kotlin-notebook/       # Jupyter notebooks
```

---

## Implementation Timeline

### Q1: Performance Foundation ✅ COMPLETED
- [x] Memory-mapped GGUF loading (JVM)
- [x] BitNet/Ternary TQ1_0/TQ2_0 support with native ternary matmul
- [x] Q8_0 quantized inference kernel
- [x] Q4_K quantized inference kernel
- [x] JVM Vector API SIMD kernels for quantized matmul
- [x] Off-heap KV-cache with KvCache abstraction

### Q2: Developer Experience
- [ ] High-level Chat API
- [ ] Streaming Flow API
- [ ] Chat template support (Llama3, ChatML)
- [ ] Basic structured output (JSON)

### Q3: Model Support
- [ ] Mistral architecture
- [ ] Phi-3 architecture
- [ ] Gemma 2 architecture
- [ ] Q4_K quantized inference

### Q4: Platform & Distribution
- [ ] Metal acceleration (macOS/iOS)
- [ ] Maven Central publishing
- [ ] Documentation site
- [ ] Example applications

### Future
- [ ] WebGPU acceleration
- [ ] MoE support (Mixtral)
- [ ] Vision models (LLaVA)
- [ ] Fine-tuning/LoRA support
- [ ] Speculative decoding

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Inference speed (tok/s) | ≥80% of llama.cpp |
| Memory efficiency | Run 7B Q4 in 8GB RAM |
| API simplicity | 3 lines to generate text |
| Platform coverage | JVM, iOS, Android, Browser |
| Model compatibility | Top 20 HuggingFace models |
| Time to first token | <500ms for 7B models |

---

## Competitive Advantages

1. **True Multiplatform**: Single codebase for JVM, iOS, Android, Browser
2. **Kotlin-First**: Coroutines, Flow, null-safety, DSL builders
3. **No Native Dependencies**: Pure Kotlin/Java, no JNI/NDK complexity
4. **Incremental Adoption**: Use high-level API or drop to low-level
5. **Type Safety**: Compile-time model configuration validation

---

## Conclusion

KLlama has a unique position: the only pure Kotlin multiplatform LLM runtime. By focusing on performance parity with jlama, excellent developer experience, and leveraging Kotlin's multiplatform capabilities, KLlama can become the default choice for:

- **Android developers** wanting on-device AI
- **iOS developers** (via Kotlin Native) wanting shared AI logic
- **Backend developers** wanting simple JVM deployment
- **Full-stack developers** wanting browser-based inference

The path forward is clear: Performance first, then DX, then ecosystem.
