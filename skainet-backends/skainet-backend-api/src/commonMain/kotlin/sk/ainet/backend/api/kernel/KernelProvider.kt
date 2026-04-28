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
}
