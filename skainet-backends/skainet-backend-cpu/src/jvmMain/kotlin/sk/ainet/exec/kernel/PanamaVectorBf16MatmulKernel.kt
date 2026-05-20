package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Bf16MatmulKernel

/**
 * SIMD-vectorized FP32 × BF16 matmul on the JDK Vector API.
 *
 * Outer iteration order is i-p-j (outer-product accumulator into rows
 * of `out`). The inner-`j` loop is the hot path:
 *
 *  1. Dequant `laneCount` BF16 weights into a scratch `FloatArray`
 *     (`bf16_bits << 16`, scalar but tight; JIT auto-vectorizes the
 *     shift-or-shift sequence to a small degree on some configs).
 *  2. Load the scratch into a `FloatVector` (`bVec`).
 *  3. Load the existing `out` row segment into `outVec`.
 *  4. `aBcast.fma(bVec, outVec)` then `intoArray(out, …)` — one
 *     FMA cycle per lane, lowered to `vfmadd231ps` (x86) or `fmla`
 *     (ARM NEON) per the active species.
 *  5. Scalar tail when `n % laneCount != 0`.
 *
 * Performance vs [ScalarBf16MatmulKernel] on FP32 × BF16: ~2–4× on
 * NEON, ~4–8× on AVX2, depending on `n`. Dequant remains scalar; the
 * hot multiply-accumulate is vectorized. A future revision can lift
 * the dequant into the SIMD domain via `IntVector.lanewise(LSHL, 16)`
 * + `reinterpretAsFloats()` once profiling shows the dequant is the
 * bottleneck.
 *
 * Numerical parity vs [ScalarBf16MatmulKernel] is asserted by
 * `PanamaVectorBf16MatmulKernelParityTest` within FMA + reordered-
 * reduction tolerance (the same `1e-5 * k` bar Panama uses elsewhere;
 * tightened to absolute `1e-2` for the dequant rounding error BF16
 * carries by construction).
 */
public object PanamaVectorBf16MatmulKernel : Bf16MatmulKernel {

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "PanamaVectorBf16MatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return
        if (k == 0) {
            for (i in 0 until m) {
                val rowOff = outOffset + i * outStride
                for (j in 0 until n) out[rowOff + j] = 0f
            }
            return
        }

        val laneCount = floatSpecies.length()
        val bound = floatSpecies.loopBound(n)
        val scratch = FloatArray(laneCount)

        // Zero the output block first (i-p-j outer product accumulates into it).
        for (i in 0 until m) {
            val rowOff = outOffset + i * outStride
            for (j in 0 until n) out[rowOff + j] = 0f
        }

        for (i in 0 until m) {
            val aRowOff = aOffset + i * aStride
            val outRowOff = outOffset + i * outStride
            for (p in 0 until k) {
                val aIp = a[aRowOff + p]
                val aBcast = FloatVector.broadcast(floatSpecies, aIp)
                val bRowByteOff = bByteOffset + p * bByteStride
                var j = 0
                while (j < bound) {
                    val byteBase = bRowByteOff + j * 2
                    for (lane in 0 until laneCount) {
                        val lo = b[byteBase + lane * 2].toInt() and 0xFF
                        val hi = b[byteBase + lane * 2 + 1].toInt() and 0xFF
                        scratch[lane] = Float.fromBits(((hi shl 8) or lo) shl 16)
                    }
                    val bVec = FloatVector.fromArray(floatSpecies, scratch, 0)
                    val outVec = FloatVector.fromArray(floatSpecies, out, outRowOff + j)
                    aBcast.fma(bVec, outVec).intoArray(out, outRowOff + j)
                    j += laneCount
                }
                // Tail (scalar): up to laneCount - 1 trailing columns.
                while (j < n) {
                    val bByteIdx = bRowByteOff + j * 2
                    val lo = b[bByteIdx].toInt() and 0xFF
                    val hi = b[bByteIdx + 1].toInt() and 0xFF
                    out[outRowOff + j] += aIp * Float.fromBits(((hi shl 8) or lo) shl 16)
                    j++
                }
            }
        }
    }
}
