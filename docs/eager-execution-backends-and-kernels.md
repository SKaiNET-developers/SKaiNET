# Eager execution: backends & kernels

A map of SKaiNET's **eager** compute path — the `TensorOps` backend and its pluggable
matmul **kernel providers** — showing what exists today (✅), what's in progress (🚧), and
what's missing (❌). The eager path is `DirectCpuExecutionContext → DefaultCpuOps* →
KernelRegistry → KernelProvider`, distinct from the StableHLO/IREE export path.

Legend: ✅ available · 🚧 partial / works via a legacy path · ❌ missing.

```mermaid
mindmap
  root((SKaiNET eager execution))
    CPU backend
      Scalar floor ✅
        commonMain — all KMP targets
        FP32 ✅
        BF16 ✅
        Q8_0 ✅
        Q4_0 ✅
        Q4_K ✅
        Q6_K ✅
        Q5_1 ✅
        Q5_0 ✅
      Panama Vector ✅
        JVM SIMD only — jdk.incubator.vector, NOT available on Android/ART
        FP32 BF16 Q8_0 Q4_0 ✅
        Q4_K Q6_K ✅
        Q5_1 Q5_0 ✅
      Native FFM ✅
        JVM only — C kernels via CMake, ART has no java.lang.foreign
        FP32 BF16 Q8_0 Q4_0 Q4_K ✅
        Q4_K MemSeg zero-copy ✅
        Q5_1 Q5_0 ✅ new
        Q6_K ❌
      Native JNI ✅ new
        Android only — same C kernels as Native FFM, reached via JNI
        two .so tiers, cpuinfo-gated armv8-a vs armv8.2+dotprod
        Q8_0 Q4_0 Q4_K Q5_K Q6_K Q5_1 Q5_0 ✅
        dense FP32 ❌ issue 920
      Apple Accelerate ✅
        Native macOS iOS — cinterop
        dense FP32 matmul ✅
        elementwise reductions ✅
        packed quant via scalar
      Native cinterop ✅ new
        Kotlin/Native — linux + Apple, static archive in the klib
        Q8_0 Q4_0 Q4_K Q5_K Q6_K Q5_1 Q5_0 ✅
        Apple: runtime FEAT_DotProd dispatch, one archive A12 through M-series
        manual registration — installNativeKernels(), no ServiceLoader on K/N
    Platforms
      JVM ✅ scalar + Panama + Native-FFM
      Android ✅ scalar + Native-JNI — NOT Panama, NOT FFM
      Native linux ✅ scalar + Native-cinterop
      Native apple ✅ scalar + Accelerate + Native-cinterop
      JS and WASM ✅ scalar only
    Gaps and roadmap
      Native FFM Q6_K ❌
      Native JNI dense FP32 ❌ issue 920
      Q5_K Q2_K Q3_K IQ4 packed on non-K/N targets ❌ dequant only
      GPU backends IREE Metal ❌ future
```

## Kernel × provider (matmul, FP32 activations)

| Weight format | Scalar (all targets) | Panama Vector (JVM only) | Native FFM (JVM) | Native JNI (Android) | Native cinterop (K/N) |
|---|:--:|:--:|:--:|:--:|:--:|
| FP32 | ✅ | ✅ | ✅ | ❌ (#920) | ❌ |
| BF16 | ✅ | ✅ | ✅ | ❌ (#920) | ❌ |
| Q8_0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q4_0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q4_K | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q6_K | ✅ | ✅ | ❌ | ✅ | ✅ |
| Q5_K | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q5_1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q5_0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q2_K / Q3_K / Q8_K / IQ4 | ❌ (dequant-to-FP32 only) | ❌ | ❌ | ❌ | ❌ |

Resolution is by priority: **Native (100, whichever of FFM/JNI/cinterop applies to the
target) → Panama (50, JVM only) → Scalar (0)** — the best *available* provider that
carries the kernel wins; otherwise it cascades down. At most one native tier is ever
compiled into a given target, so "Native" columns are mutually exclusive per platform,
not stacked.

## Platform × what runs

| Target | Providers available | Notes |
|---|---|---|
| **JVM** | Scalar + Panama + Native-FFM | full SIMD/native acceleration |
| **Android** | Scalar + Native-JNI | Panama (`jdk.incubator.vector`) and FFM (`java.lang.foreign`) are both JDK-only — ART has neither, so Android's native tier is JNI, not a degraded JVM |
| **Kotlin/Native — linux x64/arm64** | Scalar + Native-cinterop | static archive embedded in the klib; manual `installNativeKernels()`, no ServiceLoader |
| **Kotlin/Native — macOS/iOS** | Scalar + Apple Accelerate + Native-cinterop | Accelerate accelerates *dense* FP32/reductions; Native-cinterop covers packed quant, with runtime FEAT_DotProd dispatch (one archive serves A12 through M-series) |
| **JS / WASM (Js, Wasi)** | Scalar | no SIMD |

**Packed-quant matmul now works on every target** (Q4_K/Q6_K/Q5_1/Q5_0 gained a commonMain
scalar kernel, and `DefaultCpuOpsBase` dispatches packed weights via the registry). Before,
those formats were JVM-only and broke on Native.

## In progress / missing (with trackers)

- ❌ **Native FFM Q6_K** — the only packed format the FFM C kernel set doesn't cover (FP32/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q5_1/Q5_0 all ship). Q5_1/Q5_0 shipped in 0.39.1, closing the former **SKaiNET#708**.
- ❌ **Native JNI dense FP32/BF16** — the Android JNI provider has no GEMM shim yet; dense ops fall through to scalar on Android regardless of which `.so` tier loaded. Tracked by **SKaiNET#920**.
- ✅ **Native packed-quant kernels on Kotlin/Native** — `NativeKnKernelProvider` (priority 100, `skainet-backend-native-cpu`) serves Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/Q5_0/Q5_1 from the C kernels statically embedded in the klib, on linuxX64/linuxArm64 and (since #959) iosArm64/iosSimulatorArm64/macosArm64. Apple archives use runtime FEAT_DotProd dispatch (#958) so one device archive serves A12 through M-series. Registration is **manual** — call `installNativeKernels()` once at startup (no ServiceLoader on K/N).
- ❌ **Dense FP32/BF16 SIMD on Kotlin/Native linux** — the dense floats still run the scalar floor there (Apple has Accelerate). Tracked by **SKaiNET#722** / **#910**.
- ❌ **Other GGML quant formats** (Q2_K, Q3_K, Q8_K, IQ4_NL/XS) — loadable via dequant-to-FP32, but no packed matmul kernel on any provider.
- ❌ **Non-CPU eager backends** (IREE, Metal, GPU) — the `KernelProvider` SPI anticipates them, but none are implemented for the eager path today. The *compiled* path (DSL → StableHLO → IREE) does reach GPU on Android via Vulkan — see `SKaiNET-transformers`' `llm-runtime/iree-android` — but that's a separate pipeline from this eager-execution mindmap.

> This mindmap is a hand-authored overview. Its companion
> [kernel × platform support matrix](modules/ROOT/pages/reference/kernel-support-matrix.adoc) is
> **machine-generated** from the registered `KernelProvider`s (`KernelSupportMatrixTest` →
> `kernel-support.json` → `generateKernelMatrix`, the kernel-side analogue of the
> `operators.json` → `ops-status-matrix.adoc` pipeline) and gated against scalar-floor drift,
> so the per-platform coverage stays in sync with the code.
