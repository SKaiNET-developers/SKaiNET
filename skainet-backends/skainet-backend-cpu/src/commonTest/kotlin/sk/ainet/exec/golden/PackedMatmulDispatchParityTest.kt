package sk.ainet.exec.golden

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * SKEEP-003 golden gate, dispatch half (runs on every target): for all seven GGML packed
 * encodings, `ops.matmul(x, ops.transpose(w))` on a canonical row-major packed weight must agree
 * with an FP32 reference computed from the decoded weight. Tolerance-based because the JVM may
 * pick SIMD / native / Q8-activation kernel tiers (#944), which are not bit-identical to the
 * scalar reference; the bit-identical guarantees live in the goldenTest source set.
 */
class PackedMatmulDispatchParityTest {

    private enum class Fmt(val blockSize: Int, val bytesPerBlock: Int) {
        Q4_0(32, 18), Q5_0(32, 22), Q5_1(32, 24), Q8_0(32, 34), Q4_K(256, 144), Q5_K(256, 176), Q6_K(256, 210)
    }

    private class Rng(seed: Long) {
        private var s: Long = seed
        fun nextLong(): Long { var x = s; x = x xor (x ushr 12); x = x xor (x shl 25); x = x xor (x ushr 27); s = x; return x * 0x2545F4914F6CDD1DuL.toLong() }
        fun nextByte(): Byte = (nextLong() ushr 56).toByte()
        fun nextFloat(): Float = ((nextLong() ushr 40).toInt() and 0xFFFFFF) / 16777216.0f
    }

    private fun half(v: Float): Int {
        val bits = v.toRawBits(); val sign = (bits ushr 16) and 0x8000
        val expo = ((bits ushr 23) and 0xFF) - 127 + 15; val mant = bits and 0x7FFFFF
        if (expo <= 0) return sign; if (expo >= 31) return sign or 0x7C00
        return sign or (expo shl 10) or (mant ushr 13)
    }

    private fun le16(b: ByteArray, off: Int, h: Int) { b[off] = (h and 0xFF).toByte(); b[off + 1] = ((h ushr 8) and 0xFF).toByte() }

    /** Valid random block: random payload, sane FP16 scales (no NaN/Inf). */
    private fun block(f: Fmt, rng: Rng): ByteArray {
        val b = ByteArray(f.bytesPerBlock) { rng.nextByte() }
        val d = rng.nextFloat() * 0.045f + 0.005f; val dMin = rng.nextFloat() * 0.02f + 0.005f; val m = rng.nextFloat() - 0.5f
        when (f) {
            Fmt.Q4_0, Fmt.Q5_0, Fmt.Q8_0 -> le16(b, 0, half(d))
            Fmt.Q5_1 -> { le16(b, 0, half(d)); le16(b, 2, half(m)) }
            Fmt.Q4_K, Fmt.Q5_K -> { le16(b, 0, half(d)); le16(b, 2, half(dMin)) }
            Fmt.Q6_K -> le16(b, 208, half(d))
        }
        return b
    }

    private val ctx = DirectCpuExecutionContext()

    @Suppress("UNCHECKED_CAST")
    private fun parity(f: Fmt, build: (Shape, ByteArray) -> PackedBlockStorage) {
        val outDim = 4; val blocksPerRow = 3; val inDim = blocksPerRow * f.blockSize; val batch = 3
        val rng = Rng(0x5EED_0004L + f.ordinal)
        // canonical row-major bytes: row o's blocks contiguous
        val bytes = ByteArray(outDim * blocksPerRow * f.bytesPerBlock)
        for (o in 0 until outDim) for (bI in 0 until blocksPerRow) {
            block(f, rng).copyInto(bytes, (o * blocksPerRow + bI) * f.bytesPerBlock)
        }
        val storage = build(Shape(outDim, inDim), bytes)
        val w = ctx.fromData(storage as TensorData<FP32, Float>, FP32::class)
        val xf = FloatArray(batch * inDim) { rng.nextFloat() * 2f - 1f }
        val x = ctx.fromFloatArray<FP32, Float>(Shape(batch, inDim), FP32::class, xf)

        val actual = ctx.ops.matmul(x, ctx.ops.transpose(w)).data.copyToFloatArray()

        // FP32 reference from the decoded weight
        val wf = FloatArray(outDim * inDim); val tmp = FloatArray(f.blockSize)
        for (b in 0 until outDim * blocksPerRow) { storage.dequantizeBlock(b, tmp, 0); tmp.copyInto(wf, b * f.blockSize) }
        val expected = FloatArray(batch * outDim)
        var maxRef = 0f
        for (r in 0 until batch) for (o in 0 until outDim) {
            var acc = 0f
            for (i in 0 until inDim) acc += xf[r * inDim + i] * wf[o * inDim + i]
            expected[r * outDim + o] = acc; maxRef = maxOf(maxRef, abs(acc))
        }
        val tol = 5e-3f * maxOf(maxRef, 1f) + 1e-4f
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= tol,
                "${f.name}: dispatch output[$i]=${actual[i]} vs reference ${expected[i]} (tol $tol)",
            )
        }
    }

    @Test fun q4_0() = parity(Fmt.Q4_0) { s, b -> Q4_0BlockTensorData(s, b) }
    @Test fun q5_0() = parity(Fmt.Q5_0) { s, b -> Q5_0BlockTensorData(s, b) }
    @Test fun q5_1() = parity(Fmt.Q5_1) { s, b -> Q5_1BlockTensorData(s, b) }
    @Test fun q8_0() = parity(Fmt.Q8_0) { s, b -> Q8_0BlockTensorData(s, b) }
    @Test fun q4_K() = parity(Fmt.Q4_K) { s, b -> Q4_KBlockTensorData(s, b) }
    @Test fun q5_K() = parity(Fmt.Q5_K) { s, b -> Q5_KBlockTensorData(s, b) }
    @Test fun q6_K() = parity(Fmt.Q6_K) { s, b -> Q6_KBlockTensorData(s, b) }
}
