package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.io.gguf.dequant.DequantOps

/**
 * Canonical-layout regression guard for Q5_K, mirroring [Q4KCanonicalLayoutTest].
 *
 * Q5_K block format (176 bytes per 256-element block, per ggml-quants.c
 * `block_q5_K`):
 *   - bytes [  0..  1]: f16 d (super-block scale)
 *   - bytes [  2..  3]: f16 dMin (super-block min-scale)
 *   - bytes [  4.. 15]: 12-byte packed (scaleIdx, minIdx) via `get_scale_min_k4`
 *   - bytes [ 16.. 47]: 32 qh bytes (high 1 bit of each 5-bit code)
 *   - bytes [ 48..175]: 128 qs bytes (low 4 bits of each 5-bit code, strided
 *                       per 32-byte group like Q4_K)
 *
 * Per-element decode (from `dequantize_row_q5_K`):
 *   for j = 0..3 (4 outer 64-element groups):
 *     u = 1 << (2*j); u1=u; u2=u<<1
 *     for l = 0..31:
 *       low nibble:  qLow=qs[l]&0xF, qHigh = (qh[l] & u1) ? 1 : 0;  q=qLow|(qHigh<<4)
 *                    out[2j*32 + l] = (d * scaleIdx[2j])   * q - (dMin * minIdx[2j])
 *       high nibble: qLow=qs[l]>>4,  qHigh = (qh[l] & u2) ? 1 : 0;  q=qLow|(qHigh<<4)
 *                    out[(2j+1)*32 + l] = (d * scaleIdx[2j+1]) * q - (dMin * minIdx[2j+1])
 *     advance qs += 32 (next 32-byte group)
 *
 * `qh[l]` is indexed by `l = 0..31` in EVERY outer group — different bits of
 * the same 32 bytes encode the high-bit for different elements across
 * groups. Earlier `dequantQ5KFromBytes` indexed `qh[idx / 8]` with `idx` a
 * sequential output position — that maps qh's 32 bytes by *output index* not
 * *element-within-group*, which is wrong and corrupts the high bits on
 * every element except by accident. Real GGUF tensors store qh in the
 * canonical `qh[l]` layout, so the bug surfaces on real Gemma 4 E2B's
 * `per_layer_token_embd` (Q5_K).
 */
class Q5KCanonicalLayoutTest {

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
     * Build a single 176-byte Q5_K block in canonical ggml layout.
     *
     * @param subBlockCodes shape [8 sub-blocks][32 codes]; each code in 0..31 (5 bits)
     */
    private fun buildCanonicalQ5KBlock(
        d: Float,
        dMin: Float,
        scaleIdx: IntArray,
        minIdx: IntArray,
        subBlockCodes: Array<IntArray>,
    ): ByteArray {
        require(scaleIdx.size == 8 && minIdx.size == 8)
        require(subBlockCodes.size == 8 && subBlockCodes.all { it.size == 32 })
        require(subBlockCodes.all { sub -> sub.all { it in 0..31 } }) {
            "Q5_K codes are 5-bit (0..31)"
        }

        val block = ByteArray(176)

        val dBits = floatToHalf(d)
        block[0] = (dBits and 0xFF).toByte()
        block[1] = ((dBits shr 8) and 0xFF).toByte()
        val dMinBits = floatToHalf(dMin)
        block[2] = (dMinBits and 0xFF).toByte()
        block[3] = ((dMinBits shr 8) and 0xFF).toByte()

        // Same get_scale_min_k4 packing as Q4_K (sub-blocks 4..7 reuse top
        // 2 bits of bytes 0..3).
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

        // qh: 32 bytes, qh[l] holds 8 bits — for outer iter j and nibble (low/hi):
        //   bit 2j   of qh[l] = high-bit of subBlockCodes[2j  ][l]
        //   bit 2j+1 of qh[l] = high-bit of subBlockCodes[2j+1][l]
        val qhBytes = IntArray(32)
        for (j in 0 until 4) {
            for (l in 0 until 32) {
                val highLo = (subBlockCodes[2 * j    ][l] ushr 4) and 0x01  // 5th bit, low-nibble half
                val highHi = (subBlockCodes[2 * j + 1][l] ushr 4) and 0x01  // 5th bit, hi-nibble half
                qhBytes[l] = qhBytes[l] or (highLo shl (2 * j))
                qhBytes[l] = qhBytes[l] or (highHi shl (2 * j + 1))
            }
        }
        for (i in 0 until 32) block[16 + i] = (qhBytes[i] and 0xFF).toByte()

        // qs: 128 bytes, strided in 4 groups of 32 like Q4_K; byte (16 + j*32 + l)
        //   wait — for Q5_K the qs starts at offset 48 (after 4 + 12 + 32 = 48).
        //   byte (48 + j*32 + l) lo nibble = code[2j  ][l] & 0x0F
        //   byte (48 + j*32 + l) hi nibble = code[2j+1][l] & 0x0F
        for (j in 0 until 4) {
            for (l in 0 until 32) {
                val lo = subBlockCodes[2 * j    ][l] and 0x0F
                val hi = subBlockCodes[2 * j + 1][l] and 0x0F
                block[48 + j * 32 + l] = ((hi shl 4) or lo).toByte()
            }
        }
        return block
    }

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
    fun `scalar Q5_K dequant matches analytic on canonical-layout block`() {
        val d = 0.125f
        val dMin = 0.0625f
        val scaleIdx = intArrayOf(63, 50, 40, 30, 25, 18, 12, 5)
        val minIdx   = intArrayOf( 0,  3,  9, 15, 22, 31, 47, 60)
        // Codes vary in BOTH the low-4-bits AND the 5th bit so the qh layout
        // bug we're flagging will produce wildly different results.
        val subBlockCodes = Array(8) { s ->
            IntArray(32) { j -> ((s * 11 + j * 5) and 0x1F) }  // 0..31 (5-bit)
        }

        val block = buildCanonicalQ5KBlock(d, dMin, scaleIdx, minIdx, subBlockCodes)
        val expected = analyticDequant(d, dMin, scaleIdx, minIdx, subBlockCodes)
        val gotScalar = DequantOps.dequantFromBytes(block, GGMLQuantizationType.Q5_K, 256)

        var maxAbs = 0f
        var firstDiffIdx = -1
        for (i in 0 until 256) {
            val ad = kotlin.math.abs(expected[i] - gotScalar[i])
            if (ad > maxAbs) maxAbs = ad
            if (firstDiffIdx == -1 && ad > 1e-3f) firstDiffIdx = i
        }
        if (firstDiffIdx >= 0) {
            println("[analytic vs scalar Q5_K] maxAbs=$maxAbs firstDiffIdx=$firstDiffIdx")
            for (i in 0 until 8) println("  i=$i  analytic=${expected[i]}  scalar=${gotScalar[i]}")
            for (i in 32 until 40) println("  i=$i  analytic=${expected[i]}  scalar=${gotScalar[i]}")
            for (i in 64 until 72) println("  i=$i  analytic=${expected[i]}  scalar=${gotScalar[i]}")
        }
        assertTrue(
            maxAbs < 1e-3f,
            "scalar Q5_K dequant disagrees with analytic by maxAbs=$maxAbs — qh-byte indexing bug suspected"
        )
    }
}
