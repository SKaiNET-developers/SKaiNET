package sk.ainet.lang.types

/**
 * Execution-side dtype constraint.
 *
 * A [DTypePolicy] describes what an op, layer, tensor binding, or
 * backend *requires* of an input tensor's dtype — it does NOT
 * describe what dtype the source file already contains. The loader
 * (or the constraint-resolution pass on a compiled graph) is
 * responsible for satisfying the policy before forward execution
 * begins: by passing the tensor through unchanged, by casting it,
 * or by failing fast if the requirement can't be met.
 *
 * Maps directly onto the RFC's "policy categories" section
 * (`rfc.md`, "DType Constraints as Policies"). The four arms
 * cover the full spectrum of strictness:
 *
 * - [Any]: no constraint — keep the source dtype, whatever it is.
 *   The adaptive default; this is what every existing call site
 *   gets implicitly today.
 * - [Require]: hard requirement — fail fast at load / compile if
 *   the tensor can't be made available in the required dtype.
 * - [Prefer]: soft requirement — use the preferred dtype if it's
 *   already available or cheap to produce, otherwise warn and fall
 *   through.
 * - [OneOf]: restricted set — accept any dtype from a small list,
 *   convert from outside the set if a conversion exists.
 *
 * Prior art in the codebase: `Bf16LoadPolicy` in
 * `skainet-io-safetensors` (the `DEQUANT_TO_FP32 | KEEP_NATIVE`
 * enum) is exactly this pattern, scoped to one dtype. [DTypePolicy]
 * generalises it so the same shape applies to every dtype the
 * engine supports.
 */
public sealed interface DTypePolicy {
    /**
     * Returns `true` if a tensor that currently has dtype [candidate]
     * already satisfies this policy without conversion. Resolution
     * code uses this as the fast-path check: if it returns `true`,
     * no cast is needed; otherwise the resolver decides whether to
     * cast, warn, or fail per the policy arm.
     */
    public fun isSatisfiedBy(candidate: DType): Boolean

    /** Adaptive: no dtype constraint. */
    public data object Any : DTypePolicy {
        override fun isSatisfiedBy(candidate: DType): Boolean = true
    }

    /**
     * Hard requirement: the tensor MUST be available in [target].
     * If the source dtype doesn't match and no cast kernel is
     * registered to bridge `source → target`, the loader / pass
     * raises an error before forward execution can start.
     */
    public data class Require(val target: DType) : DTypePolicy {
        override fun isSatisfiedBy(candidate: DType): Boolean =
            candidate == target
    }

    /**
     * Soft preference: use [target] if already available or cheap
     * to produce, otherwise fall through to the source dtype with
     * a warning.
     */
    public data class Prefer(val target: DType) : DTypePolicy {
        override fun isSatisfiedBy(candidate: DType): Boolean =
            candidate == target
    }

    /**
     * Restricted set: any dtype in [allowed] passes verbatim;
     * anything outside the set is a candidate for conversion.
     */
    public data class OneOf(val allowed: Set<DType>) : DTypePolicy {
        init {
            require(allowed.isNotEmpty()) {
                "DTypePolicy.OneOf requires a non-empty allowed set"
            }
        }

        override fun isSatisfiedBy(candidate: DType): Boolean =
            candidate in allowed
    }

    public companion object {
        /** Java-friendly factory for [Any]. */
        @kotlin.jvm.JvmStatic public fun any(): DTypePolicy = Any

        /** Java-friendly factory for [Require]. */
        @kotlin.jvm.JvmStatic public fun require(target: DType): DTypePolicy = Require(target)

        /** Java-friendly factory for [Prefer]. */
        @kotlin.jvm.JvmStatic public fun prefer(target: DType): DTypePolicy = Prefer(target)

        /** Java-friendly factory for [OneOf]. */
        @kotlin.jvm.JvmStatic public fun oneOf(vararg allowed: DType): DTypePolicy =
            OneOf(allowed.toSet())
    }
}
