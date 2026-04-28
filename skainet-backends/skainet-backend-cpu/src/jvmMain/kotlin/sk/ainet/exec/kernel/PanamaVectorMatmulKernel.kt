package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Fp32MatmulKernel

/**
 * SIMD reference [Fp32MatmulKernel] implemented on the JDK Vector API
 * (JEP 338+, `jdk.incubator.vector`). Produces results that match
 * [ScalarMatmulKernel] within FP-rounding tolerance.
 *
 * Strategy:
 * - Pack `B` into a transposed buffer `bt` of shape `(n, k)` so the
 *   inner reduction streams contiguously over `k` for both operands —
 *   `a[i, kk]` walks one row of `A` and `bt[j, kk]` walks one row of
 *   the packed transpose.
 * - Inner loop is a vector-width FMA accumulator (`v.fma(w, acc)`),
 *   reduced once per `(i, j)` pair via `reduceLanes(ADD)`.
 * - Tail elements that don't fill a vector lane are handled in scalar.
 *
 * The B-pack is `O(n * k)` floats per call; that's cheap relative to
 * the `O(m * n * k)` FLOPs but still allocates each invocation. A
 * scratch-pool integration is out of scope for this kernel and lives
 * one layer up (see `ScratchPool` SPI in `skainet-lang-core`).
 *
 * Caller contract is identical to [Fp32MatmulKernel]: strides are in
 * floats, `out` is fully overwritten in the `m × n` block, and `k == 0`
 * zeros the output block.
 */
public object PanamaVectorMatmulKernel : Fp32MatmulKernel {
    private val species: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "PanamaVectorMatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return
        if (k == 0) {
            for (i in 0 until m) {
                val rowOff = outOffset + i * outStride
                for (j in 0 until n) out[rowOff + j] = 0f
            }
            return
        }

        // Pack B^T: bt[j, kk] = b[kk, j].
        val bt = FloatArray(n * k)
        for (kk in 0 until k) {
            val src = bOffset + kk * bStride
            for (j in 0 until n) {
                bt[j * k + kk] = b[src + j]
            }
        }

        val step = species.length()
        val loopBound = species.loopBound(k)

        for (i in 0 until m) {
            val aRow = aOffset + i * aStride
            val outRow = outOffset + i * outStride
            for (j in 0 until n) {
                val btRow = j * k
                var acc = FloatVector.zero(species)
                var idx = 0
                while (idx < loopBound) {
                    val va = FloatVector.fromArray(species, a, aRow + idx)
                    val vb = FloatVector.fromArray(species, bt, btRow + idx)
                    acc = va.fma(vb, acc)
                    idx += step
                }
                var sum = acc.reduceLanes(VectorOperators.ADD)
                while (idx < k) {
                    sum += a[aRow + idx] * bt[btRow + idx]
                    idx++
                }
                out[outRow + j] = sum
            }
        }
    }
}
