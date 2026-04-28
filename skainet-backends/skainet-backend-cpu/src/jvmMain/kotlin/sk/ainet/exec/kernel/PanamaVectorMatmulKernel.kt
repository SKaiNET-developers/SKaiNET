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
 *   inner reduction streams contiguously over `k` for both operands.
 * - Cache-block the `(m, n, k)` iteration space with tiles
 *   ([TILE_M], [TILE_N], [TILE_K]). Default 8×8×128 keeps a working
 *   set well under L1 — eight A rows × 128 floats + eight Bᵀ rows ×
 *   128 floats ≈ 8 KB, within typical 32 KB L1.
 * - Inner reduction is a vector-width FMA accumulator
 *   (`v.fma(w, acc)`), reduced via `reduceLanes(ADD)` once per
 *   `(i, j)` cell per K-tile. Tail elements that don't fill a vector
 *   lane are handled in scalar.
 * - Output is zeroed once up front; per-tile work accumulates via `+=`
 *   so the K-loop can split across multiple tiles cleanly.
 *
 * The B-pack is `O(n * k)` floats per call; cheap relative to the
 * `O(m * n * k)` FLOPs but still allocates each invocation. A
 * scratch-pool integration is out of scope for this kernel and lives
 * one layer up (see `ScratchPool` SPI in `skainet-lang-core`).
 *
 * Caller contract is identical to [Fp32MatmulKernel]: strides are in
 * floats, `out` is fully overwritten in the `m × n` block, and `k == 0`
 * zeros the output block.
 */
public object PanamaVectorMatmulKernel : Fp32MatmulKernel {
    private val species: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    private const val TILE_M = 8
    private const val TILE_N = 8
    private const val TILE_K = 128

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
        // Zero the m×n output block once. The K-tile loop accumulates
        // via `+=`, so the contract "fully overwrite the output block"
        // is preserved even when k == 0 (no tile loop runs).
        for (i in 0 until m) {
            val rowOff = outOffset + i * outStride
            for (j in 0 until n) out[rowOff + j] = 0f
        }
        if (k == 0) return

        // Pack B^T: bt[j, kk] = b[kk, j].
        val bt = FloatArray(n * k)
        for (kk in 0 until k) {
            val src = bOffset + kk * bStride
            for (j in 0 until n) {
                bt[j * k + kk] = b[src + j]
            }
        }

        val step = species.length()

        var mTile = 0
        while (mTile < m) {
            val mEnd = minOf(mTile + TILE_M, m)
            var nTile = 0
            while (nTile < n) {
                val nEnd = minOf(nTile + TILE_N, n)
                var kTile = 0
                while (kTile < k) {
                    val kEnd = minOf(kTile + TILE_K, k)
                    val kLen = kEnd - kTile
                    val loopBound = species.loopBound(kLen)
                    for (i in mTile until mEnd) {
                        val aRowBase = aOffset + i * aStride + kTile
                        val outRowBase = outOffset + i * outStride
                        for (j in nTile until nEnd) {
                            val btRowBase = j * k + kTile
                            var acc = FloatVector.zero(species)
                            var idx = 0
                            while (idx < loopBound) {
                                val va = FloatVector.fromArray(species, a, aRowBase + idx)
                                val vb = FloatVector.fromArray(species, bt, btRowBase + idx)
                                acc = va.fma(vb, acc)
                                idx += step
                            }
                            var sum = acc.reduceLanes(VectorOperators.ADD)
                            while (idx < kLen) {
                                sum += a[aRowBase + idx] * bt[btRowBase + idx]
                                idx++
                            }
                            out[outRowBase + j] += sum
                        }
                    }
                    kTile = kEnd
                }
                nTile = nEnd
            }
            mTile = mEnd
        }
    }
}
