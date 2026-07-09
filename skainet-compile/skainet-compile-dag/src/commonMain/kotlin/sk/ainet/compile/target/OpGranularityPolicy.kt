package sk.ainet.compile.target

/**
 * Per-target policy deciding, for a **fused** graph op (by op name, e.g. "layernorm",
 * "scaleddotproductattention"), whether an emitter should KEEP it as a single target op
 * (`stablehlo.composite` / native kernel-call) or DECOMPOSE it into primitives.
 *
 * This is the seam for **target legalization** — matching the op *granularity* a fragile
 * vendor compiler expects, without leaking any hardware knowledge into the agnostic core.
 * The core never calls this directly; the emitters ([toStableHlo] / the C codegen) consult
 * the policy their caller resolved for the selected target. A `null` policy means "decompose
 * everything", which is the portable default (llvm-cpu et al.).
 *
 * Lives in the shared DAG module so both the StableHLO emitter (`skainet-compile-hlo`) and
 * the `TargetOptimizer` registry (`skainet-compile-opt`) can reference it without either
 * gaining a new dependency edge.
 */
public interface OpGranularityPolicy {
    /** Backend/device id this policy applies to (matches the target string / iree device). */
    public val target: String

    /** `true` = keep [opName] fused (emit composite / kernel-call); `false` = decompose. */
    public fun keepFused(opName: String): Boolean
}

/**
 * Straightforward allow-list policy: an op is kept fused iff its (lower-cased) name is in
 * [fused]. Everything else decomposes.
 */
public class FusedOpAllowList(
    override val target: String,
    private val fused: Set<String>,
) : OpGranularityPolicy {
    override fun keepFused(opName: String): Boolean = opName.lowercase() in fused
}
