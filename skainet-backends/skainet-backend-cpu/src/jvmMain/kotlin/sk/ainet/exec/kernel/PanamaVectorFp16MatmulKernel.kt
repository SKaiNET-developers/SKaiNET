package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.IntVector
import jdk.incubator.vector.VectorOperators
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
 * The substitution is exact: the JDK conversion and [Fp16Codec.decode] agree bit-for-bit on all
 * 65536 inputs, which `Fp16CodecIntrinsicParityTest` asserts exhaustively. Reaching that agreement
 * is why the codec now quiets NaN — the hardware conversion does, so the codec was aligned with it
 * in the same change rather than the kernel being held back. Numerical parity vs
 * [ScalarFp16MatmulKernel] — which still goes through the codec — is asserted by
 * `PanamaVectorFp16MatmulKernelParityTest`.
 */
public object PanamaVectorFp16MatmulKernel : Fp16MatmulKernel {

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    /**
     * Derived from [floatSpecies]' shape rather than taken as `SPECIES_PREFERRED`, so the two
     * always have the same lane count — [widen] reinterprets between them lanewise.
     */
    private val intSpecies: VectorSpecies<Int> = IntVector.SPECIES_PREFERRED.withShape(
        floatSpecies.vectorShape(),
    ) as VectorSpecies<Int>

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
        // Raw 16-bit patterns, not decoded floats: the widening happens in the vector domain.
        val scratch = IntArray(laneCount)

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
                        scratch[lane] = (hi shl 8) or lo
                    }
                    val bVec = widen(IntVector.fromArray(intSpecies, scratch, 0))
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
     * Widen one little-endian binary16 element to FP32, for the scalar tail.
     *
     * `toShort()` keeps the low 16 bits, which is exactly the packed element; the sign extension
     * that produces is what `float16ToFloat` expects. The tail runs at most `laneCount - 1` times
     * per row, so the intrinsic is enough here and the vector path is reserved for [widen].
     */
    private fun halfToFloat(lo: Int, hi: Int): Float =
        java.lang.Float.float16ToFloat((((hi shl 8) or lo).toShort()))

    /**
     * Widen a whole vector of binary16 patterns to FP32, branch-free.
     *
     * The classic shift-and-rebias conversion, done lanewise. Shifting the sign-free pattern left
     * by 13 lands binary16's exponent and mantissa in FP32's positions; adding `(127 - 15) << 23`
     * rebiases the exponent. Two cases need a correction on top, and both are applied under a mask
     * rather than a branch:
     *
     *  - **Inf/NaN** (exponent all ones) needs a second `(128 - 16) << 23`, which saturates the
     *    FP32 exponent to all ones.
     *  - **Zero and subnormals** (exponent zero) are renormalized by the FPU instead of by a loop:
     *    bump the exponent by one and subtract `2⁻¹⁴`. For a subnormal `m * 2⁻²⁴` the bumped value
     *    is `2⁻¹⁴ * (1 + m * 2⁻¹⁰)`, so the subtraction leaves exactly `m * 2⁻²⁴`; for zero it
     *    leaves `+0`, and the sign is reapplied afterwards either way.
     *
     * A signaling NaN stays signaling here, where [Fp16Codec.decode] would quiet it. That is not
     * observable: every lane feeds the FMA below, and the FMA quiets it. The exhaustive kernel
     * sweep in `PanamaVectorFp16MatmulKernelParityTest` asserts bit equality with the codec on
     * every non-NaN pattern and NaN-ness on the rest, which is exactly this contract.
     */
    private fun widen(h: IntVector): FloatVector {
        val shifted = h.and(0x7FFF).lanewise(VectorOperators.LSHL, 13)
        val expField = shifted.and(EXP_FIELD)

        var biased = shifted.add(EXP_REBIAS)
        biased = biased.add(EXP_REBIAS, expField.compare(VectorOperators.EQ, EXP_FIELD))

        val renormalized = biased.add(SUBNORMAL_BUMP).reinterpretAsFloats().sub(SUBNORMAL_MAGIC)
        val subnormal = expField.compare(VectorOperators.EQ, 0).cast(floatSpecies)

        val magnitude = biased.reinterpretAsFloats().blend(renormalized, subnormal)
        val sign = h.and(0x8000).lanewise(VectorOperators.LSHL, 16)
        return magnitude.reinterpretAsInts().or(sign).reinterpretAsFloats()
    }

    /** binary16's exponent field once shifted into FP32 position: `0x7C00 shl 13`. */
    private const val EXP_FIELD = 0x0F80_0000

    /** `(127 - 15) shl 23` — the FP32/binary16 exponent bias difference. */
    private const val EXP_REBIAS = 0x3800_0000

    /** `1 shl 23` — one exponent step, to lift a subnormal into the magic constant's binade. */
    private const val SUBNORMAL_BUMP = 0x0080_0000

    /** `2⁻¹⁴` (bits `113 shl 23`) — binary16's smallest normal, subtracted to renormalize. */
    private val SUBNORMAL_MAGIC: Float = Float.fromBits(0x3880_0000)
}
