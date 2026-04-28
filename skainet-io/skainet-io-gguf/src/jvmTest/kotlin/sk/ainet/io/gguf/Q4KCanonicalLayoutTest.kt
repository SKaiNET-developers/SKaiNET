package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.toFloatArray

/**
 * Demonstrates that `Q4_KBlockTensorData` (consumed by the JVM matmul kernel)
 * disagrees with `DequantOps.dequantQ4KFromBytes` (port of ggml-quants.c
 * `dequantize_row_q4_K`) on the *same* Q4_K bytes — i.e. on real GGUF data.
 *
 * Two independent layout disagreements:
 *
 *   1. **Code-byte (qs) layout.** ggml is strided: in each 32-byte group of qs,
 *      byte `i` lo nibble decodes to element `i` of sub-block 2j, and byte `i`
 *      hi nibble decodes to element `i` of sub-block 2j+1. The block-tensor's
 *      `getCode(b, e)` instead pairs element `2i` and `2i+1` into byte `i`
 *      (interleaved per-byte).
 *
 *   2. **Scale/min packing.** ggml uses `get_scale_min_k4` — a bit-mixing
 *      layout where sub-blocks 0..3 take 6 bits from `scales[j]` / `scales[j+4]`
 *      and sub-blocks 4..7 reuse top-2-bits of earlier bytes. The block-tensor
 *      and the matmul kernel both use a flat "12 bits per sub-block, sequential
 *      across the 12 scale bytes" packing.
 *
 * The fixture builds a single 144-byte Q4_K block in the **canonical ggml
 * encoding** (so the test's source of truth is what real GGUF Q4_K_M files
 * actually contain), runs both decode paths, and asserts that the scalar
 * dequant matches an analytically-computed expected output. The block-tensor
 * path diverges — that divergence is the bug.
 *
 * Once the kernel + block-tensor are fixed, both paths must equal `expected`.
 */
class Q4KCanonicalLayoutTest {

    /** IEEE-754 binary16 → binary32, matches `Q4_KBlockTensorData.halfToFloat`. */
    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits and 0x8000) shl 16
        val exp = (hbits and 0x7C00) shr 10
        val mant = hbits and 0x03FF
        return when (exp) {
            0 -> if (mant == 0) Float.fromBits(sign) else {
                var m = mant; var e = -14
                while ((m and 0x400) == 0) { m = m shl 1; e-- }
                m = m and 0x3FF
                Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
            }
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }

    private fun floatToHalf(value: Float): Int {
        val bits = value.toRawBits()
        val sign = (bits shr 16) and 0x8000
        val exponent = ((bits shr 23) and 0xFF) - 127
        val mantissa = bits and 0x7FFFFF
        return when {
            exponent >= 16 -> sign or 0x7C00
            exponent >= -14 -> sign or ((exponent + 15) shl 10) or (mantissa shr 13)
            else -> sign
        }
    }

    /**
     * Build a single 144-byte Q4_K block in canonical ggml layout.
     *
     * @param d           super-block scale (FP16-encodable)
     * @param dMin        super-block min  (FP16-encodable)
     * @param scaleIdx    8 6-bit scale indices (sub-blocks 0..7)
     * @param minIdx      8 6-bit min indices   (sub-blocks 0..7)
     * @param subBlockCodes  shape [8 sub-blocks][32 codes]; each code in 0..15
     *
     * Layout written:
     *   - bytes [0..1]: f16 d (LE)
     *   - bytes [2..3]: f16 dMin (LE)
     *   - bytes [4..15]: 12 packed scale/min bytes (ggml `get_scale_min_k4`)
     *   - bytes [16..143]: 128 qs bytes laid out as 4 groups of 32:
     *       group j (j=0..3) covers sub-blocks (2j, 2j+1).
     *       byte (16 + j*32 + i) holds:
     *           lo nibble = subBlockCodes[2j][i]
     *           hi nibble = subBlockCodes[2j+1][i]
     */
    private fun buildCanonicalQ4KBlock(
        d: Float,
        dMin: Float,
        scaleIdx: IntArray,
        minIdx: IntArray,
        subBlockCodes: Array<IntArray>,
    ): ByteArray {
        require(scaleIdx.size == 8 && minIdx.size == 8)
        require(subBlockCodes.size == 8 && subBlockCodes.all { it.size == 32 })

        val block = ByteArray(144)

        val dBits = floatToHalf(d)
        block[0] = (dBits and 0xFF).toByte()
        block[1] = ((dBits shr 8) and 0xFF).toByte()
        val dMinBits = floatToHalf(dMin)
        block[2] = (dMinBits and 0xFF).toByte()
        block[3] = ((dMinBits shr 8) and 0xFF).toByte()

        // Inverse of `get_scale_min_k4` in ggml-quants.c:
        //   if (j < 4): q[j]   = scaleIdx[j] & 0x3F          (low 6 bits of byte j)
        //               q[j+4] = minIdx[j]   & 0x3F          (low 6 bits of byte j+4)
        //   if (j >= 4): low 4 bits of q[j+4]  = scaleIdx[j] & 0x0F
        //                top 4 bits of q[j+4]  = minIdx[j]   & 0x0F
        //                top 2 bits of q[j-4]  = (scaleIdx[j] >> 4) & 0x03
        //                top 2 bits of q[j]    = (minIdx[j]   >> 4) & 0x03
        val scaleBytes = IntArray(12)
        for (j in 0 until 4) {
            scaleBytes[j]     = scaleBytes[j]     or (scaleIdx[j] and 0x3F)
            scaleBytes[j + 4] = scaleBytes[j + 4] or (minIdx[j]   and 0x3F)
        }
        for (j in 4 until 8) {
            val sLow4 = scaleIdx[j] and 0x0F
            val sHi2  = (scaleIdx[j] shr 4) and 0x03
            val mLow4 = minIdx[j]   and 0x0F
            val mHi2  = (minIdx[j]   shr 4) and 0x03
            scaleBytes[j + 4] = scaleBytes[j + 4] or sLow4 or (mLow4 shl 4)
            scaleBytes[j - 4] = scaleBytes[j - 4] or (sHi2 shl 6)
            scaleBytes[j]     = scaleBytes[j]     or (mHi2 shl 6)
        }
        for (i in 0 until 12) block[4 + i] = (scaleBytes[i] and 0xFF).toByte()

        // qs: 4 groups of 32 bytes; group j carries sub-blocks (2j, 2j+1)
        for (j in 0 until 4) {
            val lo = subBlockCodes[2 * j]
            val hi = subBlockCodes[2 * j + 1]
            for (i in 0 until 32) {
                val byteVal = (lo[i] and 0x0F) or ((hi[i] and 0x0F) shl 4)
                block[16 + j * 32 + i] = byteVal.toByte()
            }
        }
        return block
    }

    /**
     * Compute the analytic dequant per ggml's formula for a single block:
     *   sub-block s scale  = d    * scaleIdx[s]   (no /63 normalisation — ggml's `d1 = d * sc`)
     *   sub-block s offset = dMin * minIdx[s]
     *   out[s*32 + j] = subBlockCodes[s][j] * scale - offset
     *
     * The matmul kernel and `Q4_KBlockTensorData` get all three of these
     * pieces wrong: they divide indices by 63 and add (instead of subtract)
     * the offset. Combined with the wrong code/scale-index unpacking, every
     * Q4_K matmul on real GGUF bytes produces garbage.
     */
    private fun analyticDequant(
        d: Float,
        dMin: Float,
        scaleIdx: IntArray,
        minIdx: IntArray,
        subBlockCodes: Array<IntArray>,
    ): FloatArray {
        val out = FloatArray(256)
        for (s in 0 until 8) {
            val scale = d * scaleIdx[s]
            val offset = dMin * minIdx[s]
            for (j in 0 until 32) {
                out[s * 32 + j] = subBlockCodes[s][j] * scale - offset
            }
        }
        return out
    }

    @Test
    fun `scalar dequant matches analytic on canonical-layout block`() {
        // Distinct, non-trivial scale & min indices per sub-block; codes vary by
        // both sub-block index and position so any layout swap shows up.
        val d = 0.125f
        val dMin = 0.0625f
        val scaleIdx = intArrayOf(63, 50, 40, 30, 25, 18, 12, 5)
        val minIdx   = intArrayOf( 0,  3,  9, 15, 22, 31, 47, 60)
        val subBlockCodes = Array(8) { s -> IntArray(32) { j -> ((s * 7 + j * 3) and 0x0F) } }

        val block = buildCanonicalQ4KBlock(d, dMin, scaleIdx, minIdx, subBlockCodes)
        val expected = analyticDequant(d, dMin, scaleIdx, minIdx, subBlockCodes)
        val gotScalar = DequantOps.dequantFromBytes(block, GGMLQuantizationType.Q4_K, 256)

        var maxAbs = 0f
        var firstDiffIdx = -1
        for (i in 0 until 256) {
            val ad = kotlin.math.abs(expected[i] - gotScalar[i])
            if (ad > maxAbs) maxAbs = ad
            if (firstDiffIdx == -1 && ad > 1e-3f) firstDiffIdx = i
        }
        if (firstDiffIdx >= 0) {
            println("[analytic vs scalar] maxAbs=$maxAbs firstDiffIdx=$firstDiffIdx")
            for (i in 0 until 8) println("  i=$i  analytic=${expected[i]}  scalar=${gotScalar[i]}")
            for (i in 32 until 40) println("  i=$i  analytic=${expected[i]}  scalar=${gotScalar[i]}")
        }
        // d * sc * code - dMin * m, all values exactly representable; tolerance
        // covers the f16 round-trip on d / dMin only.
        assertTrue(maxAbs < 1e-3f, "scalar dequant disagrees with analytic by maxAbs=$maxAbs")
    }

    @Test
    fun `block-tensor dequant matches scalar on canonical-layout block`() {
        val d = 0.125f
        val dMin = 0.0625f
        val scaleIdx = intArrayOf(63, 50, 40, 30, 25, 18, 12, 5)
        val minIdx   = intArrayOf( 0,  3,  9, 15, 22, 31, 47, 60)
        val subBlockCodes = Array(8) { s -> IntArray(32) { j -> ((s * 7 + j * 3) and 0x0F) } }

        val block = buildCanonicalQ4KBlock(d, dMin, scaleIdx, minIdx, subBlockCodes)
        val gotScalar = DequantOps.dequantFromBytes(block, GGMLQuantizationType.Q4_K, 256)
        val gotPacked = Q4_KBlockTensorData.fromRawBytes(Shape(256), block).toFloatArray()

        var maxAbs = 0f
        for (i in 0 until 256) {
            val ad = kotlin.math.abs(gotScalar[i] - gotPacked[i])
            if (ad > maxAbs) maxAbs = ad
        }
        // Both implementations are reading the same canonical ggml bytes;
        // numerical divergence can only come from f16 rehydration of d/dMin,
        // which both paths perform identically. Tolerance is generous.
        assertTrue(
            maxAbs < 1e-3f,
            "block-tensor toFloatArray() disagrees with scalar DequantOps " +
                "by maxAbs=$maxAbs — Q4_KBlockTensorData has regressed away " +
                "from canonical ggml layout."
        )
    }
}
