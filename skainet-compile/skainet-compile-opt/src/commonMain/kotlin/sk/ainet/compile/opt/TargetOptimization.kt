package sk.ainet.compile.opt

/**
 * Phases of the compile pipeline where target-specific optimizations can plug in.
 * The IR flows TAPE → DAG → STABLE_HLO; each phase has its own pass shape
 * ([GraphOptimizationPass] for [DAG]; tape/StableHLO pass types are their own).
 */
public enum class CompilePhase { TAPE, DAG, STABLE_HLO }

/**
 * A **pluggable, target-specific optimizer**. This is the seam that keeps
 * hardware knowledge OUT of the agnostic compiler core: a backend (e.g. `"torq"`,
 * `"llvm-cpu"`) registers its own passes here from OUTSIDE core, and the core
 * pipeline merely asks the registry which passes to run for the selected target
 * at each phase. Nothing target-specific ever lands in the shared IR emitter or in
 * a model definition — the StableHLO stays portable.
 *
 * Contribute passes per phase by overriding the phase you need; the rest default
 * to empty. [dagPasses] (graph rewrites) and [stableHloPasses] (rewrites of the
 * emitted module) are wired; a tape phase hook follows the same shape when needed.
 */
public interface TargetOptimizer {
    /** Backend/device id this optimizer contributes to (matches the target name). */
    public val target: String

    /** DAG-phase graph rewrites (produce standard, still-portable ops). */
    public fun dagPasses(): List<GraphOptimizationPass> = emptyList()

    /**
     * Op-granularity policy for the emitters (fused vs decomposed). `null` = decompose
     * everything (the portable default). A target that wants a fused op kept as a single
     * `stablehlo.composite` / kernel-call returns a policy here. See
     * [sk.ainet.compile.target.OpGranularityPolicy].
     */
    public fun granularity(): sk.ainet.compile.target.OpGranularityPolicy? = null

    /**
     * StableHLO-phase rewrites of the emitted module, in order. They must keep the module
     * valid, portable StableHLO: a target uses them to shape the module for the vendor
     * toolchain that consumes it (e.g. `sk.ainet.compile.hlo.tosa.TosaPreparation` before a
     * StableHLO → TOSA legalization). Callers resolve them with
     * [TargetOptimizers.stableHloPassesFor] and hand them to `toStableHlo(...)`.
     */
    public fun stableHloPasses(): List<StableHloPass> = emptyList()

    // Future phases keep the same shape, e.g.:
    //   public fun tapePasses(): List<TapePass> = emptyList()
}

/**
 * A StableHLO-phase pass: rewrites an emitted module's MLIR text into equivalent MLIR text.
 * Text-level (like the emitter's own output), so it carries no dependency on the emitter
 * module; it must leave the module valid StableHLO.
 */
public interface StableHloPass {
    /** Short name for logs and diagnostics. */
    public val name: String

    /** The rewritten module. Return [mlir] unchanged when nothing applies. */
    public fun apply(mlir: String): String
}

/**
 * Pluggable registry of [TargetOptimizer]s. Backends register themselves; core
 * queries by target name. Being a registry (not a hard-coded list) is precisely
 * what keeps target-specific optimization out of the agnostic core.
 */
public object TargetOptimizers {
    private val registry = mutableMapOf<String, MutableList<TargetOptimizer>>()

    /** Plug an optimizer in. Several may register for one target; all of them apply. */
    public fun register(optimizer: TargetOptimizer) {
        registry.getOrPut(optimizer.target) { mutableListOf() }.add(optimizer)
    }

    /**
     * Callback-style registration — plug in a target's DAG passes without declaring
     * a class: `registerDagPasses("torq") { listOf(TorqAttentionTilingPass()) }`.
     */
    public fun registerDagPasses(target: String, passes: () -> List<GraphOptimizationPass>) {
        register(object : TargetOptimizer {
            override val target: String = target
            override fun dagPasses(): List<GraphOptimizationPass> = passes()
        })
    }

    /**
     * Callback-style registration of StableHLO-phase passes:
     * `registerStableHloPasses("torq") { TosaPreparation.passes() }`.
     */
    public fun registerStableHloPasses(target: String, passes: () -> List<StableHloPass>) {
        register(object : TargetOptimizer {
            override val target: String = target
            override fun stableHloPasses(): List<StableHloPass> = passes()
        })
    }

    /** Optimizers registered for [target] (empty if none / null). */
    public fun forTarget(target: String?): List<TargetOptimizer> =
        target?.let { registry[it]?.toList() } ?: emptyList()

    /**
     * The op-granularity policy for [target] — the first non-null [TargetOptimizer.granularity]
     * among the registered optimizers, or `null` (decompose everything) if none provide one.
     * Callers (the model-build tool) resolve this and pass it into `toStableHlo(...)`, so the
     * emitter stays decoupled from this registry.
     */
    public fun granularityFor(target: String?): sk.ainet.compile.target.OpGranularityPolicy? =
        forTarget(target).firstNotNullOfOrNull { it.granularity() }

    /** StableHLO-phase passes for [target], in registration order (empty if none / null). */
    public fun stableHloPassesFor(target: String?): List<StableHloPass> =
        forTarget(target).flatMap { it.stableHloPasses() }

    /** Test/reset hook. */
    public fun clear(): Unit = registry.clear()
}

/**
 * DAG-phase pipeline for a target: HW-agnostic [corePasses] first, then whatever the
 * registry has plugged in for [target]. Callers stay agnostic — they never name a
 * backend's passes, only the target string (which is the iree device name anyway).
 */
public fun dagPipelineFor(
    target: String?,
    corePasses: List<GraphOptimizationPass> = emptyList(),
    maxIterations: Int = 1,
): GraphOptimizationPipeline {
    val targetPasses = TargetOptimizers.forTarget(target).flatMap { it.dagPasses() }
    return GraphOptimizationPipeline(corePasses + targetPasses, maxIterations)
}
