package sk.ainet.exec.tensor.ops

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * Validates the packed Q5_1 / Q5_0 matmul kernels + lazy transpose: feeding a packed
 * weight through `ops.matmul(x, ops.transpose(W))` must match feeding the FP32-dequantized
 * weight through the same path. The FP32 reference is dequantized inline (independent of the
 * `Q5_*BlockTensorData.dequantizeBlock` code under test), matching ggml / `DequantOps`.
 */
class Q5MatmulDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    private fun f16(v: Float): Int {
        // float -> IEEE half bits (round-to-nearest-even, good enough for test weights)
        val bits = v.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var expo = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        if (expo <= 0) return sign // flush tiny to signed zero
        if (expo >= 31) return sign or 0x7C00 // inf
        return sign or (expo shl 10) or (mant ushr 13)
    }

    private fun halfToFloat(h: Int): Float {
        val sign = (h and 0x8000) shl 16
        val exp = (h and 0x7C00) shr 10
        val mant = h and 0x03FF
        return when (exp) {
            0 -> Float.fromBits(sign) // (subnormals flushed by f16() above)
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }

    // --- Q5_1: 24 bytes/block (d, m, qh[4], qs[16]) ---------------------------------------

    private fun randomQ5_1Block(rng: Random, out: ByteArray, off: Int) {
        val d = f16(0.02f + rng.nextFloat() * 0.05f)
        val m = f16(-0.3f + rng.nextFloat() * 0.6f)
        out[off] = (d and 0xFF).toByte(); out[off + 1] = ((d ushr 8) and 0xFF).toByte()
        out[off + 2] = (m and 0xFF).toByte(); out[off + 3] = ((m ushr 8) and 0xFF).toByte()
        for (k in 0 until 4) out[off + 4 + k] = rng.nextInt(256).toByte()      // qh
        for (k in 0 until 16) out[off + 8 + k] = rng.nextInt(256).toByte()     // qs
    }

    private fun dequantQ5_1Block(b: ByteArray, off: Int, dst: FloatArray, dstOff: Int) {
        val d = halfToFloat(((b[off + 1].toInt() and 0xFF) shl 8) or (b[off].toInt() and 0xFF))
        val m = halfToFloat(((b[off + 3].toInt() and 0xFF) shl 8) or (b[off + 2].toInt() and 0xFF))
        val qh = intArrayOf(b[off + 4].toInt() and 0xFF, b[off + 5].toInt() and 0xFF, b[off + 6].toInt() and 0xFF, b[off + 7].toInt() and 0xFF)
        for (j in 0 until 16) {
            val q = b[off + 8 + j].toInt() and 0xFF
            val lo = q and 0x0F; val hi = q ushr 4
            val bitLo = (qh[j / 8] ushr (j % 8)) and 0x01
            val bitHi = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 0x01
            dst[dstOff + j] = d * (lo + (bitLo shl 4)) + m
            dst[dstOff + 16 + j] = d * (hi + (bitHi shl 4)) + m
        }
    }

    // --- Q5_0: 22 bytes/block (d, qh[4], qs[16]), symmetric -16 --------------------------

    private fun randomQ5_0Block(rng: Random, out: ByteArray, off: Int) {
        val d = f16(0.02f + rng.nextFloat() * 0.05f)
        out[off] = (d and 0xFF).toByte(); out[off + 1] = ((d ushr 8) and 0xFF).toByte()
        for (k in 0 until 4) out[off + 2 + k] = rng.nextInt(256).toByte()
        for (k in 0 until 16) out[off + 6 + k] = rng.nextInt(256).toByte()
    }

    private fun dequantQ5_0Block(b: ByteArray, off: Int, dst: FloatArray, dstOff: Int) {
        val d = halfToFloat(((b[off + 1].toInt() and 0xFF) shl 8) or (b[off].toInt() and 0xFF))
        val qh = intArrayOf(b[off + 2].toInt() and 0xFF, b[off + 3].toInt() and 0xFF, b[off + 4].toInt() and 0xFF, b[off + 5].toInt() and 0xFF)
        for (j in 0 until 16) {
            val q = b[off + 6 + j].toInt() and 0xFF
            val lo = q and 0x0F; val hi = q ushr 4
            val bitLo = (qh[j / 8] ushr (j % 8)) and 0x01
            val bitHi = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 0x01
            dst[dstOff + j] = d * (lo + (bitLo shl 4) - 16)
            dst[dstOff + 16 + j] = d * (hi + (bitHi shl 4) - 16)
        }
    }

    private fun assertPackedMatchesFp32(
        encoding: String, inputDim: Int, outputDim: Int, batchSize: Int, seed: Int,
    ) {
        val rng = Random(seed)
        val blocksPerRow = inputDim / 32
        val bytesPerBlock = if (encoding == "Q5_1") 24 else 22
        val bytes = ByteArray(outputDim * blocksPerRow * bytesPerBlock)
        val wf = FloatArray(outputDim * inputDim) // row-major [out, in]
        for (o in 0 until outputDim) {
            for (blk in 0 until blocksPerRow) {
                val off = (o * blocksPerRow + blk) * bytesPerBlock
                val dstOff = o * inputDim + blk * 32
                if (encoding == "Q5_1") { randomQ5_1Block(rng, bytes, off); dequantQ5_1Block(bytes, off, wf, dstOff) }
                else { randomQ5_0Block(rng, bytes, off); dequantQ5_0Block(bytes, off, wf, dstOff) }
            }
        }

        val packed: Tensor<FP32, Float> = if (encoding == "Q5_1")
            ctx.fromData(Q5_1BlockTensorData(Shape(outputDim, inputDim), bytes) as TensorData<FP32, Float>, FP32::class)
        else
            ctx.fromData(Q5_0BlockTensorData(Shape(outputDim, inputDim), bytes) as TensorData<FP32, Float>, FP32::class)
        val fp32 = ctx.fromFloatArray<FP32, Float>(Shape(outputDim, inputDim), FP32::class, wf)

        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(batchSize, inputDim), FP32::class, FloatArray(batchSize * inputDim) { (rng.nextFloat() - 0.5f) },
        )
        val outPacked = ctx.ops.matmul(input, ctx.ops.transpose(packed)).data.copyToFloatArray()
        val outFp32 = ctx.ops.matmul(input, ctx.ops.transpose(fp32)).data.copyToFloatArray()

        assertEquals(outFp32.size, outPacked.size, "$encoding output size")
        var maxErr = 0f
        for (i in outFp32.indices) maxErr = maxOf(maxErr, kotlin.math.abs(outFp32[i] - outPacked[i]))
        assertTrue(maxErr < 1e-3f, "$encoding packed matmul deviates from FP32 dequant: maxErr=$maxErr")
    }

    @Test fun q5_1_matmul_matches_fp32_dequant_single_batch() =
        assertPackedMatchesFp32("Q5_1", inputDim = 128, outputDim = 64, batchSize = 1, seed = 1)

    @Test fun q5_1_matmul_matches_fp32_dequant_multi_batch() =
        assertPackedMatchesFp32("Q5_1", inputDim = 256, outputDim = 96, batchSize = 3, seed = 2)

    @Test fun q5_0_matmul_matches_fp32_dequant_single_batch() =
        assertPackedMatchesFp32("Q5_0", inputDim = 128, outputDim = 64, batchSize = 1, seed = 3)

    @Test fun q5_0_matmul_matches_fp32_dequant_multi_batch() =
        assertPackedMatchesFp32("Q5_0", inputDim = 192, outputDim = 48, batchSize = 2, seed = 4)
}
