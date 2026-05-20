package sk.ainet.backend.api.kernel

/**
 * Process-global fail-fast policy for kernel resolution.
 *
 * The RFC's "fail before execution" principle says that if a graph
 * can't find a registered kernel for an op/dtype combination, the
 * problem should surface at the load / compile / dispatch boundary
 * rather than as a silent perf regression at forward time (e.g. a
 * Q4_K matmul that falls back to a scalar dequant + FP32 matmul).
 *
 * This object provides the smallest possible affordance: a system
 * property (`-Dskainet.strict.kernels=true`) flips the runtime into
 * fail-fast mode. Dispatch sites call [failIfStrict] right before
 * they would otherwise silently fall back; the call is a no-op when
 * strict mode is off, preserving the existing adaptive behaviour.
 *
 * Per-context strict mode (e.g. `DirectCpuExecutionContext.create(strict = true)`)
 * is a follow-up that requires plumbing the flag through every
 * platform's `platformDefaultCpuOpsFactory`. The system-property
 * affordance is sufficient for tests, CI, and debugging — the
 * primary use cases — and ships with zero cross-platform plumbing.
 */
public object KernelStrictness {

    /** System-property name that flips fail-fast mode on. */
    public const val SYSTEM_PROPERTY: String = "skainet.strict.kernels"

    /**
     * Returns `true` when the runtime should fail fast instead of
     * falling back. Reads the system property on every call so a
     * test can toggle it via `System.setProperty(...)` between
     * cases without restarting the JVM.
     */
    public fun isEnabled(): Boolean =
        System.getProperty(SYSTEM_PROPERTY) == "true"

    /**
     * Throws [NoSuchKernelException] with the supplied message
     * builder when strict mode is on; otherwise returns. The
     * message builder is only invoked when the exception is going
     * to be thrown, so callers can include expensive details
     * (provider list, dtype tuples) without paying the cost in the
     * default-adaptive path.
     */
    public inline fun failIfStrict(message: () -> String) {
        if (isEnabled()) throw NoSuchKernelException(message())
    }
}

/**
 * Raised by dispatch sites when [KernelStrictness] is enabled and
 * no registered kernel matches the requested op/dtype combination.
 * The message includes the op name, the failing dtype tuple, and
 * the list of currently-registered providers so the operator can
 * see exactly which capability is missing.
 */
public class NoSuchKernelException(message: String) : RuntimeException(message)
