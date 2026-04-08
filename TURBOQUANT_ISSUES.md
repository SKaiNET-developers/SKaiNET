# TurboQuant Implementation Tracker

> Auto-generated from `prd.md` analysis on 2026-04-08.
> Branch: `feature/turboquant`

## Legend

| Symbol | Meaning |
|--------|---------|
| DONE | Implemented and tested |
| IN PROGRESS | Partially implemented |
| TODO | Not started |

---

## Step 1: SKaiNET Core Preparation (PRD sections 1-6)

### Completed

- [x] **Storage & placement abstractions** — `TensorStorage`, `TensorEncoding`, `BufferHandle`, `Placement`, `LogicalDType`
- [x] **Zero-copy & ownership semantics** — Owned, Borrowed, Aliased, FileBacked, DeviceResident
- [x] **Packed quant unification** — `PackedBlockStorage` contract with Q4_K, Q8_0, Ternary
- [x] **Streaming GGUF loader** — `StreamingGGUFReader` + `StreamingGgufParametersLoader`
- [x] **Memory planning & tracking** — `MemoryPlanner`, `MemoryTracker`, `ActiveMemoryTracker`
- [x] **Transfer & materialization APIs** — `copyMaterialize()`, `copyToHost()`, `copyToDevice()`
- [x] **DSL annotations** — `@Place`, `@Weights`
- [x] **Benchmarks** — `StorageBenchmarks.kt` (Q4_K, Q8_0, Ternary dequant throughput)
- [x] **Acceptance criteria tests** — `AcceptanceCriteriaTest.kt`

- [x] **KV-cache subsystem** — `KvCacheStore` interface, `DefaultKvCacheStore`, `KvCacheConfig`, `KvCacheMemoryReport`
- [x] **SDPA compressed K/V bridge** — `CompressedKvAttention` with dequant-on-read and raw storage paths
- [x] **Quants.kt port complete** — `byteShapeToQuantShape`, `quantByteSize`, `isBlockQuantized`, `validateQuantizedBytes`
- [x] **SafeTensors zero-copy loading** — `StorageAwareSafeTensorsLoader` with file-backed and borrowed modes

### Remaining — None (Step 1 complete)

---

### TQ-001: KV-Cache Subsystem

| Field | Value |
|---|---|
| **Status** | DONE |
| **PRD section** | Step 1, Requirement 4 |
| **Priority** | High — blocks all Step 2 work |
| **Dependencies** | None (Step 1 foundations complete) |

**Description:**
Create a `KvCacheStore` abstraction that supports append-by-token writes, layer/head addressing, compressed K/V block storage, backend-specific read/dequant flows, and asymmetric K/V policies.

**Acceptance criteria:**
- [ ] `KvCacheStore` interface defined with append, read, and eviction APIs
- [ ] Layer and head indexing supported
- [ ] Storage accepts any `TensorEncoding` (including future TurboQuant)
- [ ] Backend-specific dequant dispatch is extensible
- [ ] Asymmetric K/V encoding policies configurable per layer
- [ ] Unit tests for append, read, eviction, and multi-head addressing

**Key files to create/modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/storage/KvCacheStore.kt` (new)
- Tests in `skainet-lang/skainet-lang-core/src/commonTest/kotlin/sk/ainet/lang/tensor/storage/`

---

### TQ-002: SDPA Integration for Compressed K/V

| Field | Value |
|---|---|
| **Status** | DONE |
| **PRD section** | Step 1, Requirement 5 |
| **Priority** | High — blocks TurboQuant SDPA path |
| **Dependencies** | TQ-001 |

**Description:**
Extend `scaledDotProductAttention()` in `TensorOps.kt` to detect compressed K/V from `KvCacheStore`, decompress only the needed tiles on read, and provide a seam for fused dequant+attention.

**Acceptance criteria:**
- [ ] SDPA detects `TensorEncoding` of K/V inputs
- [ ] Compressed K/V triggers dequant-on-read path
- [ ] Only required tiles/blocks are decompressed (not full cache)
- [ ] Extension point exists for backend-fused kernels
- [ ] Tests with Q4_K and Q8_0 encoded K/V (as proxies before TurboQuant)

**Key files to modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/TensorOps.kt`

---

### TQ-003: Complete Quants.kt Port

| Field | Value |
|---|---|
| **Status** | DONE |
| **PRD section** | Step 1, Requirement 6 |
| **Priority** | Medium |
| **Dependencies** | None |

**Description:**
Complete the Python-to-Kotlin port of `Quants.kt` and `Constants.kt`. Added `byteShapeToQuantShape`, `quantElementCount`, `quantByteSize`, `isBlockQuantized`, `quantBlockSize`, `quantTypeSize`, `validateQuantizedBytes`. Removed stale TODO from `Constants.kt`.

**Acceptance criteria:**
- [ ] All quantization types from llama.cpp `quants.py` are ported
- [ ] Multi-dimension shape utilities work correctly
- [ ] `Constants.kt` port complete
- [ ] Unit tests for each ported quant type

**Key files to modify:**
- `skainet-io/skainet-io-gguf/src/commonMain/kotlin/sk/ainet/io/gguf/Quants.kt`
- `skainet-io/skainet-io-gguf/src/commonMain/kotlin/sk/ainet/io/gguf/Constants.kt`

---

### TQ-004: SafeTensors Zero-Copy / Mapped Loading

| Field | Value |
|---|---|
| **Status** | DONE |
| **PRD section** | Step 1, Requirement 6 |
| **Priority** | Medium |
| **Dependencies** | None |

**Description:**
Allow SafeTensors loaders to wrap or map buffers instead of always converting to dense arrays. Should produce `TensorStorage` with `FileBacked` or `Borrowed` buffer handles where possible.

**Acceptance criteria:**
- [ ] SafeTensors loader can produce `TensorStorage` with `FileBacked` handles
- [ ] No unnecessary heap copy for read-only weight access
- [ ] Falls back to `Owned` copy when mutation is required
- [ ] Integration test with real `.safetensors` file

**Key files to modify:**
- `skainet-io/skainet-io-safetensors/` (loader implementation)

---

## Step 2: TurboQuant Introduction (PRD sections 1-5)

---

### TQ-010: TurboQuant Encoding Types

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Product definition |
| **Priority** | High — blocks all TurboQuant kernels |
| **Dependencies** | None |

**Description:**
Add TurboQuant variants to the sealed `TensorEncoding` hierarchy: `TurboQuantPolar` (PolarOnly) and `TurboQuantPolarQjl` (PolarPlusQjl), with configurable bit budgets and block sizes.

**Acceptance criteria:**
- [ ] `TurboQuantPolar` encoding added to `TensorEncoding`
- [ ] `TurboQuantPolarQjl` encoding added to `TensorEncoding`
- [ ] Configurable: bits per element, block size, codebook variant
- [ ] `bytesPerBlock` / `elementsPerBlock` computed correctly
- [ ] Exhaustive `when` coverage in existing code updated

**Key files to modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/storage/TensorEncoding.kt`

---

### TQ-011: Random Rotation Kernel

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 1 |
| **Priority** | High |
| **Dependencies** | TQ-010 |

**Description:**
Implement random rotation generation in common Kotlin. This is the first stage of the TurboQuant pipeline — rotating input vectors before quantization.

**Acceptance criteria:**
- [ ] Deterministic random rotation matrix generation (seeded)
- [ ] Correct orthogonality properties verified
- [ ] Works for arbitrary head dimensions
- [ ] Common Kotlin (no platform-specific code)
- [ ] Unit tests verifying rotation properties

**Key files to create:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/turboquant/` (new package)

---

### TQ-012: Scalar Quantization / Codebook Lookup

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 1 |
| **Priority** | High |
| **Dependencies** | TQ-011 |

**Description:**
Implement scalar quantization with codebook lookup for the rotated vectors. Supports configurable bit widths (2, 3, 4, 8).

**Acceptance criteria:**
- [ ] Quantize rotated vector to N-bit codes
- [ ] Codebook lookup for dequantization
- [ ] Supports 2-bit, 3-bit, 4-bit, and 8-bit configurations
- [ ] Round-trip error within expected bounds per bit width
- [ ] Unit tests with known reference vectors

**Key files to create:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/turboquant/ScalarQuantizer.kt` (new)

---

### TQ-013: QJL Residual Stage

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 1 |
| **Priority** | Medium — only needed for PolarPlusQjl variant |
| **Dependencies** | TQ-012 |

**Description:**
Implement the QJL (Quantized Johnson-Lindenstrauss) residual stage for the PolarPlusQjl variant. This preserves inner-product accuracy by capturing quantization residuals.

**Acceptance criteria:**
- [ ] QJL projection of quantization residual
- [ ] Inner-product error reduction verified vs PolarOnly
- [ ] Configurable residual bit budget
- [ ] Can be disabled (for PolarOnly path)
- [ ] Unit tests comparing IP accuracy with/without QJL

**Key files to create:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/turboquant/QjlResidual.kt` (new)

---

### TQ-014: Bit-Packing Kernel

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 1 |
| **Priority** | High |
| **Dependencies** | TQ-012 |

**Description:**
Implement bit-packing/unpacking for TurboQuant codes into compact byte arrays. Must support 2, 3, 4, and 8-bit packing.

**Acceptance criteria:**
- [ ] Pack N-bit codes into byte arrays
- [ ] Unpack byte arrays back to codes
- [ ] Round-trip correctness for all supported bit widths
- [ ] Append-friendly (can pack incrementally per token)
- [ ] Unit tests for boundary conditions and all bit widths

**Key files to create:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/turboquant/BitPacker.kt` (new)

---

### TQ-015: KV Block Append/Read APIs

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 1 |
| **Priority** | High |
| **Dependencies** | TQ-001, TQ-014 |

**Description:**
Implement append and read APIs that connect TurboQuant encoding/decoding to the `KvCacheStore`. New tokens are compressed on write; stored blocks are decompressed on read.

**Acceptance criteria:**
- [ ] Append single token's K/V as TurboQuant-compressed block
- [ ] Read and decompress arbitrary range of cached tokens
- [ ] Supports both PolarOnly and PolarPlusQjl paths
- [ ] Memory-efficient (no full cache decompression)
- [ ] Integration test: append N tokens, read back, verify accuracy

**Key files to create/modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/turboquant/TurboQuantKvCodec.kt` (new)
- Integrates with `KvCacheStore` from TQ-001

---

### TQ-016: PolarOnly Variant Implementation

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Supported variants |
| **Priority** | High — primary production variant |
| **Dependencies** | TQ-011, TQ-012, TQ-014, TQ-015 |

**Description:**
Wire together rotation + scalar quantization + bit-packing into the complete PolarOnly end-to-end path. This is the backend-friendly variant without QJL.

**Acceptance criteria:**
- [ ] End-to-end: float vector in -> compressed bytes -> float vector out
- [ ] Configurable bit budget (2, 3, 4 bits)
- [ ] Accuracy within expected bounds for each bit budget
- [ ] Works through KV append/read APIs
- [ ] Benchmark: compression ratio and throughput

**Key files to modify:**
- Orchestration in `TurboQuantKvCodec.kt`

---

### TQ-017: SDPA TurboQuant Write Path

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 2 |
| **Priority** | High |
| **Dependencies** | TQ-002, TQ-016 |

**Description:**
Integrate TurboQuant compression into the SDPA write path so K/V are automatically compressed when stored to the KV cache.

**Acceptance criteria:**
- [ ] SDPA stores K/V through TurboQuant compression when configured
- [ ] Compression is transparent to callers of `scaledDotProductAttention`
- [ ] Configurable per-layer (some layers can skip compression)
- [ ] No hidden densification

**Key files to modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/TensorOps.kt`

---

### TQ-018: SDPA TurboQuant Read Path

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 2 |
| **Priority** | High |
| **Dependencies** | TQ-002, TQ-016 |

**Description:**
Integrate TurboQuant decompression into the SDPA read path so attention is computed against decompressed K/V tiles.

**Acceptance criteria:**
- [ ] SDPA reads and decompresses only required K/V tiles
- [ ] Tile-level decompression (not full cache)
- [ ] Correct attention scores compared to uncompressed baseline
- [ ] Extension point for fused backend kernels

**Key files to modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/TensorOps.kt`

---

### TQ-019: Role-Aware K/V Policies

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 3 |
| **Priority** | Medium |
| **Dependencies** | TQ-001, TQ-016 |

**Description:**
Support independent compression policies for keys and values — different bit budgets, block sizes, and even different variants (e.g., Q8_0 for K + TurboQuant-4 for V).

**Acceptance criteria:**
- [ ] K and V policies configurable independently
- [ ] Different bit budgets for K vs V
- [ ] Mixed encoding (e.g., Q8_0-K + TurboQuant-V) supported
- [ ] Per-layer policy override
- [ ] Configuration validated at init time

**Key files to modify:**
- `KvCacheStore` from TQ-001
- `TurboQuantKvCodec.kt` from TQ-015

---

### TQ-020: Presets

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Presets |
| **Priority** | Medium |
| **Dependencies** | TQ-019 |

**Description:**
Implement named preset configurations:
- **safe-lowbit**: Q8_0-K + TurboQuant-4-V
- **balanced**: TurboQuant-4 / TurboQuant-4
- **experimental-max**: TurboQuant-3 / TurboQuant-3

**Acceptance criteria:**
- [ ] Three named presets available
- [ ] Presets resolve to concrete K/V policy configurations
- [ ] Presets selectable via API and DSL
- [ ] Documentation of expected quality/compression trade-offs

**Key files to create:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/ops/turboquant/TurboQuantPresets.kt` (new)

---

### TQ-021: DSL / Annotation Support

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Recommended implementation order item 7 |
| **Priority** | Low |
| **Dependencies** | TQ-020 |

**Description:**
Extend SKaiNET DSL/annotations (`@Place`, `@Weights`) to support TurboQuant KV cache configuration declaratively.

**Acceptance criteria:**
- [ ] Annotation-based TurboQuant configuration for KV cache
- [ ] Preset selection via annotation
- [ ] Per-layer override via annotation
- [ ] Integrated with existing `PlacementAnnotations.kt`

**Key files to modify:**
- `skainet-lang/skainet-lang-core/src/commonMain/kotlin/sk/ainet/lang/tensor/storage/PlacementAnnotations.kt`

---

### TQ-022: CPU SIMD Optimization

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 5 |
| **Priority** | Medium |
| **Dependencies** | TQ-016 |

**Description:**
Optimize TurboQuant kernels (rotation, quantization, bit-packing, dequant) with CPU SIMD using the same pattern as `JvmQuantizedVectorKernels.kt`.

**Acceptance criteria:**
- [ ] SIMD-optimized rotation kernel
- [ ] SIMD-optimized quant/dequant kernels
- [ ] Benchmark showing speedup over common Kotlin reference
- [ ] Correctness matches reference implementation

**Key files to create/modify:**
- `skainet-backends/skainet-backend-cpu/src/jvmMain/kotlin/sk/ainet/exec/tensor/ops/` (new kernels)

---

### TQ-023: Metal / Apple Silicon Backend

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 5 |
| **Priority** | Medium |
| **Dependencies** | TQ-016 |

**Description:**
Implement Metal compute shaders for TurboQuant kernels targeting Apple Silicon unified memory.

**Acceptance criteria:**
- [ ] Metal shader for TurboQuant encode/decode
- [ ] Unified memory path (no CPU-GPU copy for KV cache)
- [ ] Correctness matches CPU reference
- [ ] Benchmark on Apple Silicon

**Key files to create:**
- Metal backend (new shaders)

---

### TQ-024: Fused Dequant + Attention Kernels

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Functional requirement 5 |
| **Priority** | Low — optimization after correctness |
| **Dependencies** | TQ-018, TQ-022 or TQ-023 |

**Description:**
Fuse TurboQuant decompression with attention score computation to avoid materializing decompressed K/V.

**Acceptance criteria:**
- [ ] Fused kernel avoids intermediate K/V buffer
- [ ] Correctness matches unfused path
- [ ] Benchmark showing memory and latency improvement
- [ ] At least one backend (CPU SIMD or Metal)

**Key files to create:**
- Backend-specific fused kernel implementations

---

### TQ-025: TurboQuant Benchmarks

| Field | Value |
|---|---|
| **Status** | TODO |
| **PRD section** | Step 2, Acceptance criteria |
| **Priority** | High — validates the whole effort |
| **Dependencies** | TQ-016 |

**Description:**
Add JMH benchmarks for TurboQuant KV compression: encode throughput, decode throughput, compression ratio, attention accuracy degradation.

**Acceptance criteria:**
- [ ] Encode throughput benchmark (tokens/sec)
- [ ] Decode throughput benchmark (tokens/sec)
- [ ] Compression ratio measurement for each preset
- [ ] Accuracy comparison vs uncompressed KV cache
- [ ] Results documented

**Key files to create:**
- `skainet-lang/skainet-lang-core/src/jvmMain/kotlin/sk/ainet/lang/tensor/TurboQuantBenchmarks.kt` (new)

---

## Dependency Graph

```
Step 1 remaining:
  TQ-003 (Quants.kt)          — independent
  TQ-004 (SafeTensors)         — independent
  TQ-001 (KV-cache)            — independent
  TQ-002 (SDPA compressed K/V) — depends on TQ-001

Step 2:
  TQ-010 (Encoding types)      — independent
  TQ-011 (Rotation)            — depends on TQ-010
  TQ-012 (Scalar quant)        — depends on TQ-011
  TQ-013 (QJL residual)        — depends on TQ-012
  TQ-014 (Bit-packing)         — depends on TQ-012
  TQ-015 (KV append/read)      — depends on TQ-001, TQ-014
  TQ-016 (PolarOnly e2e)       — depends on TQ-011, TQ-012, TQ-014, TQ-015
  TQ-017 (SDPA write)          — depends on TQ-002, TQ-016
  TQ-018 (SDPA read)           — depends on TQ-002, TQ-016
  TQ-019 (K/V policies)        — depends on TQ-001, TQ-016
  TQ-020 (Presets)             — depends on TQ-019
  TQ-021 (DSL)                 — depends on TQ-020
  TQ-022 (CPU SIMD)            — depends on TQ-016
  TQ-023 (Metal)               — depends on TQ-016
  TQ-024 (Fused kernels)       — depends on TQ-018, TQ-022 or TQ-023
  TQ-025 (Benchmarks)          — depends on TQ-016
```

## Recommended Implementation Order

1. **TQ-001** + **TQ-003** + **TQ-004** + **TQ-010** (parallel — no dependencies between them)
2. **TQ-002** + **TQ-011** (after TQ-001 and TQ-010)
3. **TQ-012** (after TQ-011)
4. **TQ-013** + **TQ-014** (parallel, after TQ-012)
5. **TQ-015** (after TQ-001 + TQ-014)
6. **TQ-016** + **TQ-025** (PolarOnly e2e + benchmarks)
7. **TQ-017** + **TQ-018** + **TQ-019** (SDPA integration + policies)
8. **TQ-020** + **TQ-022** + **TQ-023** (presets + backend optimization)
9. **TQ-021** + **TQ-024** (DSL + fused kernels — last)
