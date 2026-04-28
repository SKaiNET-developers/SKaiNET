# PRD — Native (FFM) Kernel Provider

**Status:** Deferred (post-0.21.0). Captured here so the design doesn't drift.
**Owner:** unassigned.
**Milestone:** M5 (CPU backend dispatch) — final piece. The roadmap's M5 success metric `native ≥2.5× for Q4_K` requires this provider; the JVM Vector half (PRs #554, #557, #560, #562, #563, #564) closed the Panama story but not the native one.

## Context

The kernel SPI shipped in PR #554 (`KernelProvider`, `Fp32MatmulKernel`, `KernelRegistry`) was designed to host **three** providers, ordered by priority:

| priority | provider | status |
|---------:|---|---|
|   0 | `ScalarKernelProvider` | shipped (#554) |
|  50 | `PanamaVectorKernelProvider` | shipped (#557, plus tile-blocking #560, ServiceLoader #559) |
| 100 | `NativeKernelProvider` (FFM) | **this PRD** |

The Panama provider runs the FP32 matmul at ~73 GFLOPS for square 4096² shapes on Apple Silicon (per #558's JMH bench), the Q4_K SIMD kernel runs in the same throughput regime as Panama FP32 (#562 numbers). That's already in the ggml NEON ballpark — but ggml's hand-tuned NEON / AVX2 still outruns Panama on dense per-cycle FLOPs and on Q4_K specifically, where 4-bit nibble unpacking maps cleanly to dedicated SIMD shuffles that the Vector API can't always emit.

A native provider closes that gap and unlocks two follow-ons:
- **M4 ↔ M5 synergy.** Mmap'd Q4_K weights stay as `MemorySegment` views; a native kernel reads the same pages with zero copy via FFI. No staging buffer, no `ByteArray` round-trip.
- **Hardware-specific lanes.** AVX-512 VNNI fused INT8 dot products, NEON `bf16`/`fp16` SDOT instructions, future SVE — the Vector API exposes none of these portably today.

## Goals

1. **A `NativeKernelProvider` registered at priority 100** that on JDK 21+ wins `KernelRegistry.bestAvailable()` over Panama whenever the native lib is loaded successfully.
2. **A first concrete kernel: native Q4_K matmul.** It must:
   - take a `MemorySegment` for both input (FP32) and packed Q4_K weights (canonical ggml layout — same as `Q4_KBlockTensorData` and the existing `matmulF32Q4_KMemSeg`);
   - produce numerically equivalent output to `PanamaVectorQ4KMatmulKernel` within `1e-4` relative tolerance (same parity bar `PanamaVectorQ4KMatmulKernelTest` uses);
   - clear **≥2.5× over the prior Q4_K scalar dequant baseline** on the bench shapes from `QuantizedMatmulBench` (1024², 4096×1024, 4096²).
3. **Optional follow-on kernels** — Q6_K, Q8_0, FP32 matmul — share the build system but each ship as a separate small PR.
4. **One supported architecture for the first PR** (likely Apple Silicon NEON, since that's the development hardware in use), with a clear extension path for `linuxX64` AVX2 / `linuxArm64` NEON.

## Non-goals

- **JNI.** The roadmap explicitly says "FFM not JNI". JNI's per-call overhead and the global JNI lock are wrong for hot per-token kernels; FFM (Java 22 stable, Java 21 preview) gives near-zero-overhead native calls and direct `MemorySegment` ABI.
- **Cross-compilation matrix on day one.** The first PR can ship just one (host-arch) variant; CI cross-arch builds come later.
- **Replacing Panama.** Panama remains the priority-50 fallback for environments that can't load native libs (sandboxes, Wasm, Native targets, JDK without `jdk.incubator.vector`).
- **Distribution via Maven Central pre-built native artifacts.** Out of scope for the first PR — local build only. A separate "publish native classifier JARs" PRD comes later.

## Architecture

### Module layout

```
skainet-backends/
  skainet-backend-native-cpu/                  # NEW
    src/
      jvmMain/kotlin/sk/ainet/exec/kernel/     # Kotlin side
        NativeKernelProvider.kt                # priority=100, isAvailable() = libLoaded
        NativeQ4KMatmulKernel.kt               # implements Q4KMatmulKernel, calls FFM
        NativeLibraryLoader.kt                 # loadLibrary, locate, check API version
      jvmMain/resources/META-INF/services/
        sk.ainet.backend.api.kernel.KernelProvider  # appends NativeKernelProviderFactory
      jvmTest/kotlin/sk/ainet/exec/kernel/
        NativeQ4KMatmulKernelTest.kt           # parity vs PanamaVectorQ4KMatmulKernel
      native/                                  # native source tree
        c/
          q4k_matmul.c                         # ggml-style hand-tuned kernel
          q4k_matmul.h
        CMakeLists.txt                         # or Bazel BUILD
        build.gradle.kts                       # Gradle wrapper that invokes CMake
```

The native library compiles to a shared object (`libskainet_kernels.dylib` on macOS, `.so` on Linux, `.dll` on Windows) and is packaged into the module's resources for `System.loadLibrary` discovery.

### FFM binding pattern

Single C entry point per kernel:

```c
// q4k_matmul.h
void skainet_q4k_matmul(
    const float* input,        // FP32 input vector, length input_dim
    const uint8_t* weight,     // packed Q4_K bytes (canonical ggml layout)
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,             // FP32 output, length output_dim
    int32_t output_offset
);
```

Kotlin side:

```kotlin
internal object NativeQ4KMatmulKernel : Q4KMatmulKernel {
    private val handle: MethodHandle = run {
        val arena = Arena.ofAuto()
        val symbol = NativeLibraryLoader.lib.find("skainet_q4k_matmul").orElseThrow()
        Linker.nativeLinker().downcallHandle(
            symbol,
            FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ),
        )
    }

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        // Heap arrays: pass via temporary off-heap MemorySegment + bulk copy,
        // OR (preferred) overload with a MemorySegment-input variant for
        // mmap'd weights to avoid the copy.
        ...
    }
}
```

The cleaner path is to introduce a sibling **`Q4KMemSegMatmulKernel`** SPI (mentioned as out-of-scope in #563) that takes `MemorySegment` directly, and have the native provider implement *that* — no heap copy. The `Q4KMatmulKernel` (ByteArray) variant can wrap the MemSeg one with a temporary `Arena.ofConfined()` copy if needed for legacy callers.

### Build system

**Gradle + CMake** is the path of least resistance:
- A new Gradle plugin (or hand-rolled `Exec` tasks) invokes CMake for the native module's `build` task.
- Native artifacts land in `build/native/<arch>/` and are copied into `src/jvmMain/resources/native/<os>-<arch>/` so `System.loadLibrary` finds them.
- The Kotlin compile depends on the native artifact being built first.

The `xnnpack` backend already in the repo (`skainet-backends/skainet-backend-xnnpack/`) demonstrates a similar pattern — Gradle invokes CMake to build a native lib via cinterop. **Reuse that template** rather than reinventing.

**Architecture detection**: at native module build time, query host arch and only build for it (first PR scope). CI cross-arch matrix follows.

### Provider class

```kotlin
public object NativeKernelProvider : KernelProvider {
    override val name: String = "native-ffm"
    override val priority: Int = 100

    private val available: Boolean by lazy {
        runCatching { NativeLibraryLoader.load() }.isSuccess
    }

    override fun isAvailable(): Boolean = available

    override fun matmulFp32(): Fp32MatmulKernel? = null  // future PR
    override fun matmulQ4K(): Q4KMatmulKernel? =
        if (isAvailable()) NativeQ4KMatmulKernel else null
}
```

Registered via the existing ServiceLoader mechanism (`META-INF/services/sk.ainet.backend.api.kernel.KernelProvider`, factory wrapper because `KernelProvider by NativeKernelProvider`). When unavailable, the cascade falls through to Panama (priority 50), preserving the M5 metric on environments without native code.

## Staged delivery

PRs in order, each independently mergeable:

1. **`skainet-backend-native-cpu` module scaffolding** — Gradle module, build.gradle.kts wired to invoke CMake, a *trivial* C kernel (e.g. just multiplies its first input by 2.0 and writes to output) to prove the FFM pipeline end-to-end. NativeKernelProvider that's `isAvailable() = false` until the real kernel lands. Sets up CI artifact path on host arch.
2. **First real native kernel: Q4_K matmul (Apple Silicon NEON)** — hand-tuned kernel, parity tests vs PanamaVectorQ4KMatmulKernel, JMH bench variant added to `QuantizedMatmulBench`.
3. **`Q4KMemSegMatmulKernel` SPI sibling + native variant** — closes the M4↔M5 zero-copy story for mmap'd weights.
4. **linuxX64 AVX2 variant + cross-arch CI build** — the cross-compilation matrix story.
5. **Optional: native FP32 matmul, native Q6_K, native Q8_0** — same shape as PRs 2–3, one per format.

The first PR (1) is the largest in *scaffolding* terms (~500–800 LoC of build glue + 1 trivial kernel), but every subsequent PR is small and template-able.

## Success metrics

- **PR 2 sign-off**: native Q4_K matmul on Apple Silicon clears **≥2.5×** over the scalar Q4_K dequant-then-matmul baseline at 4096² (the M5 milestone target). For reference: Panama Q4_K SIMD already exceeds this metric (see #562 PR body, ~73 GFLOPS), so the bar is "beats Panama by a meaningful margin", probably ≥1.5× over Panama.
- **PR 3 sign-off**: Q4_K MemSeg native path is faster than the Panama Q4_K MemSeg path from #563, with no heap copy in the timed region.
- **No regression on JVM-only environments** — when the native lib fails to load (sandbox, missing arch, etc.), `KernelRegistry.bestAvailable()` cleanly falls through to Panama, and existing tests / benches show the same numbers as today.

## Risks & open questions

1. **JDK 21 preview vs JDK 22 stable.** FFM left preview in Java 22. The repo currently builds on JDK 21 with `--enable-preview --add-modules jdk.incubator.vector`. We need to decide: stay on JDK 21 preview FFM (smaller blast radius, matches Vector API status), or bump to JDK 22+ for stable FFM. **Recommendation**: stay on 21 preview; flip to 22 in a separate toolchain-bump PR.
2. **`MethodHandle` invocation overhead.** Even with FFM, each native call has a small fixed cost (microseconds-ish). For the smallest matmul shapes (e.g. 256² FP32) this could swamp the FLOPs win. Mitigation: route small inputs to Panama and large inputs to native at the registry/provider level, OR accept that the win is sized for production-relevant shapes (4096²+).
3. **Native code quality and maintenance.** Hand-tuned NEON / AVX2 in C is harder to audit than Kotlin Vector API code. Mitigation: keep kernels small (<300 LoC each), parity-test exhaustively, prefer porting from ggml's reference (which is BSD-licensed and well-vetted) over writing from scratch.
4. **Distribution.** Native artifacts complicate Maven Central publication (need `<classifier>` per OS/arch). For the first internal-use PR this isn't a blocker, but a separate "publish native classifier JARs" PRD will be needed before community use.
5. **Cross-arch CI cost.** Building NEON natively on Apple Silicon CI plus AVX2 on linuxX64 plus Android NDK doubles or triples build time. The xnnpack backend's existing CI matrix is a precedent — reuse the same approach.
6. **Native `MemorySegment` lifetime.** The Kotlin caller owns the `Arena` for arrays it copies in. The native kernel must NOT retain pointers past the FFM call return. Document this contract in `NativeQ4KMatmulKernel.matmul` kdoc.

## When to start

Trigger conditions (any one):
- Real workload demands the native ≥2.5× target (Panama Q4_K stops being fast enough on a customer machine).
- A community contributor offers a hand-tuned NEON / AVX2 Q4_K kernel that's measurably faster than Panama.
- A second M5 metric (e.g. SDPA throughput, training-loop throughput) needs hand-tuned native code.

Until then: **pause**. The Panama provider is doing the milestone-equivalent work in absolute terms, and adding a native build system is a meaningful complexity tax to take on speculatively.
