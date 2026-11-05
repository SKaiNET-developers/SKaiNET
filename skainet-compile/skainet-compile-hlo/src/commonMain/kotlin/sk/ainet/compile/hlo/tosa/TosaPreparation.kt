package sk.ainet.compile.hlo.tosa

import sk.ainet.compile.opt.StableHloPass

/**
 * StableHLO-phase passes that shape an emitted module for StableHLO's own TOSA legalization
 * (`stablehlo-opt --stablehlo-prepare-for-tosa --stablehlo-legalize-to-tosa`, [LEGALIZE_ARGS]).
 *
 * The output stays valid, portable StableHLO with the same numerics: it only moves work to where a
 * TOSA consumer handles it well. A backend that compiles TOSA (e.g. an NPU whose TOSA input pipeline
 * makes a whole function one dispatch) registers them for its target:
 *
 * ```
 * TargetOptimizers.registerStableHloPasses("torq") { TosaPreparation.passes() }
 * ```
 *
 * The legalization leaves a few StableHLO ops it has no pattern for; [TosaFinish] translates those.
 */
public object TosaPreparation {
    /** Arguments of `stablehlo-opt` for the legalization that follows these passes. */
    public val LEGALIZE_ARGS: List<String> = listOf("--stablehlo-prepare-for-tosa", "--stablehlo-legalize-to-tosa")

    /** All preparation passes, in order. */
    public fun passes(): List<StableHloPass> = listOf(DivideByBroadcastPass, RankTwoWeightMatmulPass)
}

/**
 * `x / broadcast(sqrt(z))` → `x * broadcast(rsqrt(z))` and `x / broadcast(y)` → `x * broadcast(1 / y)`.
 *
 * Legalized as is, the reciprocal runs on the broadcast tile (a layer norm's `[T, D]`, a softmax's
 * `[H, T, T]`) instead of the row, and `sqrt` becomes `tosa.pow`. Generated values use the `%tq_` prefix.
 */
public object DivideByBroadcastPass : StableHloPass {
    override val name: String = "tosa-divide-by-broadcast"

    override fun apply(mlir: String): String = rewrite(mlir).first

    /** The rewritten module and the number of divides rewritten. */
    public fun rewrite(mlir: String): Pair<String, Int> {
        val lines = mlir.split('\n')
        // value -> (op, rest of the line), as defined so far: a value is defined before it is read, and names are
        // scoped per function, so the latest definition is the one in scope
        val defs = HashMap<String, Pair<String, String>>()
        val out = ArrayList<String>(lines.size + 64)
        var n = 0
        for (l in lines) {
            val m = if (l.length <= LONG) SHLO_DEF.matchEntire(l) else null
            if (m != null) defs[m.groupValues[2]] = m.groupValues[3] to m.groupValues[4]
            if (m != null && m.groupValues[3] == "stablehlo.divide") {
                val (ind, v, _, rest) = m.destructured
                val (a, b) = NAME.findAll(rest).map { it.value }.take(2).toList()
                val bcast = defs[b]
                if (bcast?.first == "stablehlo.broadcast_in_dim") {
                    val bc = bcast.second
                    val y = NAME.find(bc)!!.value
                    val dims = BC_DIMS.find(bc)!!.value
                    val (ty, tb) = BC_TYPES.find(bc)!!.destructured
                    n++
                    val r = "%tq_r$n"
                    val src = defs[y]
                    if (src?.first == "stablehlo.sqrt") {
                        out += "$ind$r = stablehlo.rsqrt ${NAME.find(src.second)!!.value} : $ty"
                    } else {
                        out += "$ind%tq_one$n = stablehlo.constant dense<1.000000e+00> : $ty"
                        out += "$ind$r = stablehlo.divide %tq_one$n, $y : $ty"
                    }
                    out += "$ind%tq_b$n = stablehlo.broadcast_in_dim $r, $dims : ($ty) -> $tb"
                    out += "$ind$v = stablehlo.multiply $a, %tq_b$n : $tb"
                    continue
                }
            }
            out += l
        }
        return out.joinToString("\n") to n
    }

    private val BC_DIMS = Regex("""dims = \[[^\]]*\]""")
    private val BC_TYPES = Regex(""": \((tensor<[^>]+>)\) -> (tensor<[^>]+>)""")
}

/**
 * A weight matmul `dot_general` of a `[1, M, K]` activation with a `[K, N]` matrix → reshape to
 * `[M, K]`, a rank-2 `dot_general`, reshape back to `[1, M, N]`.
 *
 * The legalization converts only "simple" dots (both operands of rank ≤ 2, default dimension
 * numbers); in this form it emits `tosa.matmul` itself. Batched dots (attention) are left as they
 * are, for [TosaFinish].
 */
public object RankTwoWeightMatmulPass : StableHloPass {
    override val name: String = "tosa-rank-two-weight-matmul"

    override fun apply(mlir: String): String = rewrite(mlir).first

    /** The rewritten module and the number of matmuls rewritten. */
    public fun rewrite(mlir: String): Pair<String, Int> {
        val lines = mlir.split('\n')
        val out = ArrayList<String>(lines.size + 128)
        var n = 0
        for (l in lines) {
            val m = if (l.length <= LONG) WEIGHT_DOT.matchEntire(l) else null
            if (m == null) { out += l; continue }
            val (ind, r, a, w, at, rows, k, et, wt, cols) = m.destructured
            n++
            val p = "%tq_mm${n}_"
            out += "$ind${p}a = stablehlo.reshape $a : ($at) -> tensor<${rows}x${k}x$et>"
            out += "$ind${p}d = stablehlo.dot_general ${p}a, $w, contracting_dims = [1] x [0] : (tensor<${rows}x${k}x$et>, $wt) -> tensor<${rows}x${cols}x$et>"
            out += "$ind$r = stablehlo.reshape ${p}d : (tensor<${rows}x${cols}x$et>) -> tensor<1x${rows}x${cols}x$et>"
        }
        return out.joinToString("\n") to n
    }

    private val WEIGHT_DOT = Regex(
        """^(\s*)(%[\w]+) = stablehlo\.dot_general (%[\w]+), (%[\w]+), contracting_dims = \[2\] x \[0\] : """ +
            """\((tensor<1x(\d+)x(\d+)x([a-z]+\d+)>), (tensor<\d+x\d+x\8>)\) -> tensor<1x\d+x(\d+)x\8>$""",
    )
}

internal const val LONG = 4096
internal val NAME = Regex("""%[\w]+""")
private val SHLO_DEF = Regex("""^(\s*)(%[\w]+) = (stablehlo\.[a-z_]+)(.*)$""")
