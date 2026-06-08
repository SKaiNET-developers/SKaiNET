package sk.ainet.backend.api.kernel

/**
 * Provider for a family of related numeric kernels (matmul, SDPA, ...).
 *
 * A backend (Panama Vector, native FFM, IREE, Metal, ...) bundles its
 * kernels behind a single provider so callers only need to know the
 * top-level provider name. Providers self-report whether they are
 * available on the current platform / runtime — e.g. a Panama Vector
 * provider returns `false` from [isAvailable] on a JDK that doesn't
 * have the incubator module loaded.
 *
 * Lookup rules:
 * - Higher [priority] wins when multiple providers report
 *   [isAvailable] = `true`. Providers should rank themselves by
 *   expected performance: scalar ≈ 0, Panama Vector ≈ 50, hand-tuned
 *   native ≈ 100.
 * - Each per-kernel accessor returns `null` when the provider does not
 *   carry that kernel, so callers can fall through to a lower-priority
 *   provider.
 */
public interface KernelProvider {
    /** Stable, human-readable identifier. */
    public val name: String

    /**
     * Relative ranking versus other providers. Higher = preferred when
     * available. The scalar reference uses `0`; SIMD-accelerated
     * providers should use a larger value.
     */
    public val priority: Int

    /**
     * Reports whether this provider's kernels can run in the current
     * process. Expensive checks (probing CPU features, loading native
     * libraries) should be done once and cached.
     */
    public fun isAvailable(): Boolean

    /**
     * FP32 matmul kernel exposed by this provider, or `null` if this
     * provider does not specialize matmul.
     */
    public fun matmulFp32(): Fp32MatmulKernel?

    /**
     * F32 × Q4_K matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize Q4_K. Default returns `null`
     * so providers that pre-date this accessor (e.g. older custom
     * providers and the scalar reference) keep compiling without
     * change — callers cascade to a lower-priority provider that does
     * carry the kernel.
     */
    public fun matmulQ4K(): Q4KMatmulKernel? = null

    /**
     * F32 × BF16 matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize BF16. Same fall-through pattern
     * as [matmulQ4K] — older providers keep compiling; callers cascade
     * to the next provider when this one returns `null`.
     */
    public fun matmulBf16(): Bf16MatmulKernel? = null

    /**
     * F32 × Q8_0 matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize Q8_0. Same fall-through pattern.
     */
    public fun matmulQ8_0(): Q8_0MatmulKernel? = null

    /**
     * F32 × Q4_0 matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize Q4_0. Same fall-through pattern.
     */
    public fun matmulQ4_0(): Q4_0MatmulKernel? = null

    /**
     * F32 × Q6_K matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize Q6_K. Same fall-through pattern.
     */
    public fun matmulQ6K(): Q6KMatmulKernel? = null

    /**
     * F32 × Q5_1 matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize Q5_1. Same fall-through pattern.
     */
    public fun matmulQ5_1(): Q5_1MatmulKernel? = null

    /**
     * F32 × Q5_0 matmul kernel exposed by this provider, or `null` if
     * this provider does not specialize Q5_0. Same fall-through pattern.
     */
    public fun matmulQ5_0(): Q5_0MatmulKernel? = null

    /**
     * Capability query: does this provider carry a kernel for
     * [opName] with the given [dtypeKeys]?
     *
     * Returns `true` iff the corresponding per-kernel accessor on
     * this interface returns non-null. Callers (constraint
     * resolution, fail-fast dispatch, capability tables) use this
     * to ask "do you support this combination?" without actually
     * fetching the kernel.
     *
     * Convention:
     * - For matmul, [dtypeKeys] is `[inputDtypeName, weightDtypeName]`
     *   using the same string names as [sk.ainet.lang.types.DType.name]
     *   for floats / ints (`"Float32"`, `"BFloat16"`, …) and the
     *   short canonical block-format names for quantized weights
     *   (`"Q4_K"`, `"Q8_0"`).
     * - For ops that aren't matmul (future: SDPA, gather, RMSNorm…),
     *   providers can override this method to declare those kernels.
     *
     * The default body covers the four matmul accessors that exist
     * on this interface today. Providers that ship additional
     * kernels override and chain through `super.supports(...)` for
     * the matmul base cases.
     */
    public fun supports(opName: String, dtypeKeys: List<String>): Boolean {
        if (opName != "matmul" || dtypeKeys.size != 2) return false
        val (input, weight) = dtypeKeys
        if (input != "Float32") return false
        return when (weight) {
            "Float32" -> matmulFp32() != null
            "BFloat16" -> matmulBf16() != null
            "Q4_K" -> matmulQ4K() != null
            "Q8_0" -> matmulQ8_0() != null
            "Q4_0" -> matmulQ4_0() != null
            "Q6_K" -> matmulQ6K() != null
            "Q5_1" -> matmulQ5_1() != null
            "Q5_0" -> matmulQ5_0() != null
            else -> false
        }
    }
}
