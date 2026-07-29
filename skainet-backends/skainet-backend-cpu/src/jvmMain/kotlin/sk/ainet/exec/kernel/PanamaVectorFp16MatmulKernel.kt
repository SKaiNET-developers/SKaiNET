package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Fp16MatmulKernel
import sk.ainet.lang.types.Fp16Codec

/**
 * Panama Vector API implementation of [Fp16MatmulKernel] — the FP16 counterpart to
 * [PanamaVectorBf16MatmulKernel], using the same i-p-j outer-product shape and FP32 FMA
 * accumulation.
 *
 * **Why the decode stays scalar here.** BF16 can be widened lane-wise with an integer shift
 * (`IntVector.lanewise(LSHL, 16).reinterpretAsFloats()`), which is why the BF16 kernel vectorizes
 * its dequant. Binary16 needs exponent rebiasing and gradual-underflow handling, and mainstream
 * JDKs expose no FP16 vector species, so each element is decoded scalar into a lane-width scratch
 * buffer before the vectorized multiply-accumulate. The FMA over `n` is still fully vectorized —
 * only the widening is not.
 *
 * **Why not [Fp16Codec] (#887).** The codec is portable integer bit math, and calling it once per
 * weight element made this kernel run at a flat ~0.5 GFLOP/s — 2-18x *slower* than the FP32 SGEMM
 * it replaces, while the structurally identical BF16 kernel was 1.5-2.1x faster. `float16ToFloat`
 * is a HotSpot intrinsic (JDK 20+) that lowers to a single `vcvtph2ps` on F16C hardware and to a
 * compact branch-free sequence elsewhere, so the scratch fill stops dominating the inner loop.
 * The kernel is JVM-only and this provider already gates on JDK 21+, so the intrinsic is always
 * available where this code runs.
 *
 * Results are unchanged: the JDK conversion and [Fp16Codec.decode] agree bit-for-bit on all 65536
 * inputs, which `Fp16CodecIntrinsicParityTest` asserts exhaustively. Numerical parity vs
 * [ScalarFp16MatmulKernel] — which still goes through the codec — is asserted by
 * `PanamaVectorFp16MatmulKernelParityTest`.
 */
public object PanamaVectorFp16MatmulKernel : Fp16MatmulKernel {

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "PanamaVectorFp16MatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
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

        // Zero the output block first — the i-p-j outer product accumulates into it.
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
                        scratch[lane] = halfToFloat(lo, hi)
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
                    out[outRowOff + j] += aIp * halfToFloat(lo, hi)
                    j++
                }
            }
        }
    }

    /**
     * Widen one little-endian binary16 element to FP32.
     *
     * `toShort()` keeps the low 16 bits, which is exactly the packed element; the sign extension
     * that produces is what `float16ToFloat` expects.
     */
    private fun halfToFloat(lo: Int, hi: Int): Float =
        java.lang.Float.float16ToFloat((((hi shl 8) or lo).toShort()))
}
