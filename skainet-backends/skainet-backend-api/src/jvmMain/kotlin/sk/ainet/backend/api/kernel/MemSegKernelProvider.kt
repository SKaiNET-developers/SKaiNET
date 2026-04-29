package sk.ainet.backend.api.kernel

/**
 * JVM-only sibling of [KernelProvider] for kernels whose interface
 * surface depends on `java.lang.foreign.MemorySegment`. Kept separate
 * because [KernelProvider] lives in `commonMain` — adding
 * `MemorySegment` accessors there would break Kotlin/Native, JS, and
 * Wasm targets.
 *
 * Providers that ship MemSeg-input kernels declare both interfaces:
 *
 * ```kotlin
 * public object MyProvider : KernelProvider, MemSegKernelProvider { ... }
 * ```
 *
 * Lookup pattern at the call site:
 *
 * ```kotlin
 * val kernel = (KernelRegistry.bestAvailable() as? MemSegKernelProvider)
 *     ?.matmulQ4KMemSeg()
 *     ?: fallbackHeapPath()
 * ```
 *
 * No automatic registry lookup helper for now — the smart-cast is
 * sufficient and avoids a second registry. If a third MemSeg surface
 * lands (FP32 matmul-MemSeg, Q6_K matmul-MemSeg, ...) it joins this
 * interface as another `null`-defaulting accessor.
 */
public interface MemSegKernelProvider {
    /**
     * F32 × Q4_K matmul-MemSeg kernel exposed by this provider, or
     * `null` if this provider does not specialize the MemSeg path.
     * Default returns `null` so providers that pre-date the MemSeg SPI
     * keep compiling.
     */
    public fun matmulQ4KMemSeg(): Q4KMemSegMatmulKernel? = null
}
