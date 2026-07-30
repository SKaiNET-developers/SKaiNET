package sk.ainet.lang.tensor

/**
 * Vocabulary for a single tensor-dimension extent, and the canonical home of the [DYNAMIC] marker.
 *
 * A [Shape] stores extents as plain `Int`s (kept for storage/backend compatibility). An extent is either:
 *  - a concrete, materializable size (`>= 0`), or
 *  - [DYNAMIC] — a size unknown at compile time (e.g. a growing KV-cache sequence length). It renders as
 *    `?` in MLIR, and a single compiled program then serves every concrete size at that axis.
 *
 * [DYNAMIC] is a RESERVED sentinel ([Int.MIN_VALUE]), deliberately distinct from `-1`, which the reshape
 * DSL uses for "infer this dimension from the total volume". Keeping the two values distinct removes the
 * historical overloading where a dynamic axis and a reshape-infer slot were both `-1` and could collide.
 *
 * All dynamic-aware shape arithmetic lives here so ops don't scatter ad-hoc `extent < 0` guards.
 */
public object Dim {
    /** Reserved sentinel extent meaning "dynamic / unknown size". Distinct from reshape's `-1` = infer. */
    public const val DYNAMIC: Int = Int.MIN_VALUE

    /** True iff [extent] is the [DYNAMIC] sentinel. */
    public fun isDynamic(extent: Int): Boolean = extent == DYNAMIC

    /** True iff [extent] is a concrete, materializable size (`>= 0`). */
    public fun isStatic(extent: Int): Boolean = extent >= 0

    /**
     * Extent of concatenating many tensors along one axis: [DYNAMIC] if ANY input is dynamic there, else
     * the sum. (A growing cache `? ++ n` must stay `?`; numerically summing it would corrupt the shape,
     * e.g. `? + 1` collapsing to a bogus static `0`.)
     */
    public fun concat(extents: List<Int>): Int =
        if (extents.any { isDynamic(it) }) DYNAMIC else extents.sum()

    /**
     * Are two extents compatible for an elementwise / broadcast op — equal, or either side dynamic?
     * A dynamic axis is compatible with any concrete size (the concrete one wins as the known extent).
     */
    public fun compatible(a: Int, b: Int): Boolean = isDynamic(a) || isDynamic(b) || a == b

    /** Render an extent for MLIR: [DYNAMIC] (or any non-concrete value) as `?`, else the decimal size. */
    public fun render(extent: Int): String = if (isStatic(extent)) extent.toString() else "?"
}

/** True if any extent in this dimension list is [Dim.DYNAMIC]. Lets both the tracer and the emitter pick
 *  dynamic-shape-safe forms only when needed, leaving fully-static shapes on their original code paths. */
public fun List<Int>.hasDynamic(): Boolean = any { Dim.isDynamic(it) }

/** True if any extent is [Dim.DYNAMIC]. */
public fun IntArray.hasDynamic(): Boolean = any { Dim.isDynamic(it) }
