package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.toFloatArray

/**
 * Ties the packed [Q5_KBlockTensorData] (used by the eager CPU matmul path)
 * to the proven golden [DequantOps.dequantQ5KFromBytes] across a *multi-block*
 * buffer. Multiple blocks are the case that exposed the historical `qh[idx/8]`
 * indexing bug — a single block can pass by accident, so this builds 3 blocks
 * with distinct codes and asserts bit-exact agreement.
 */
class Q5KBlockTensorDataParityTest {

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

    /** Build a single 176-byte canonical Q5_K block (see [Q5KCanonicalLayoutTest]). */
    private fun buildBlock(
        d: Float,
        dMin: Float,
        scaleIdx: IntArray,
        minIdx: IntArray,
        subBlockCodes: Array<IntArray>,
    ): ByteArray {
        val block = ByteArray(176)
        val dBits = floatToHalf(d)
        block[0] = (dBits and 0xFF).toByte()
        block[1] = ((dBits shr 8) and 0xFF).toByte()
        val dMinBits = floatToHalf(dMin)
        block[2] = (dMinBits and 0xFF).toByte()
        block[3] = ((dMinBits shr 8) and 0xFF).toByte()

        val scaleBytes = IntArray(12)
        for (j in 0 until 4) {
            scaleBytes[j] = scaleBytes[j] or (scaleIdx[j] and 0x3F)
            scaleBytes[j + 4] = scaleBytes[j + 4] or (minIdx[j] and 0x3F)
        }
        for (j in 4 until 8) {
            val sLow4 = scaleIdx[j] and 0x0F
            val sHi2 = (scaleIdx[j] shr 4) and 0x03
            val mLow4 = minIdx[j] and 0x0F
            val mHi2 = (minIdx[j] shr 4) and 0x03
            scaleBytes[j + 4] = scaleBytes[j + 4] or sLow4 or (mLow4 shl 4)
            scaleBytes[j - 4] = scaleBytes[j - 4] or (sHi2 shl 6)
            scaleBytes[j] = scaleBytes[j] or (mHi2 shl 6)
        }
        for (i in 0 until 12) block[4 + i] = (scaleBytes[i] and 0xFF).toByte()

        val qhBytes = IntArray(32)
        for (j in 0 until 4) {
            for (l in 0 until 32) {
                val highLo = (subBlockCodes[2 * j][l] ushr 4) and 0x01
                val highHi = (subBlockCodes[2 * j + 1][l] ushr 4) and 0x01
                qhBytes[l] = qhBytes[l] or (highLo shl (2 * j))
                qhBytes[l] = qhBytes[l] or (highHi shl (2 * j + 1))
            }
        }
        for (i in 0 until 32) block[16 + i] = (qhBytes[i] and 0xFF).toByte()

        for (j in 0 until 4) {
            for (l in 0 until 32) {
                val lo = subBlockCodes[2 * j][l] and 0x0F
                val hi = subBlockCodes[2 * j + 1][l] and 0x0F
                block[48 + j * 32 + l] = ((hi shl 4) or lo).toByte()
            }
        }
        return block
    }

    @Test
    fun `Q5_KBlockTensorData toFloatArray matches DequantOps golden across blocks`() {
        val nBlocks = 3
        val buf = ByteArray(nBlocks * 176)
        for (b in 0 until nBlocks) {
            val d = 0.125f + 0.01f * b
            val dMin = 0.0625f + 0.005f * b
            val scaleIdx = IntArray(8) { (63 - (it * 7 + b * 3)) and 0x3F }
            val minIdx = IntArray(8) { (it * 8 + b * 5) and 0x3F }
            val codes = Array(8) { s -> IntArray(32) { j -> ((s * 11 + j * 5 + b * 13) and 0x1F) } }
            val block = buildBlock(d, dMin, scaleIdx, minIdx, codes)
            block.copyInto(buf, b * 176)
        }

        val golden = DequantOps.dequantFromBytes(buf, GGMLQuantizationType.Q5_K, nBlocks * 256)
        val packed = Q5_KBlockTensorData(Shape(nBlocks * 256), buf).toFloatArray()

        assertEquals(golden.size, packed.size)
        var maxAbs = 0f
        var firstDiff = -1
        for (i in golden.indices) {
            val ad = kotlin.math.abs(golden[i] - packed[i])
            if (ad > maxAbs) maxAbs = ad
            if (firstDiff == -1 && ad > 0f) firstDiff = i
        }
        assertTrue(
            maxAbs == 0f,
            "Q5_KBlockTensorData disagrees with DequantOps golden: maxAbs=$maxAbs firstDiff=$firstDiff",
        )
    }
}
