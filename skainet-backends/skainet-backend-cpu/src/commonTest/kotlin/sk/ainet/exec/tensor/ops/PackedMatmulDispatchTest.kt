package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * End-to-end proof that packed-quant weights flow through `ctx.ops.matmul(x, ops.transpose(W))`
 * on EVERY platform — exercising the lazy-transpose shape-swap + `chooseQuantizedMatmulHeap` in
 * DefaultCpuOpsBase, resolving the registered kernel (scalar on Native/JS/WASM, Panama/FFM on JVM).
 * Runs on jvmTest AND linuxX64Test; a green linuxX64 run is the headline "Native packed matmul works".
 */
class PackedMatmulDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    private fun half(v: Float): Int {
        val b = v.toRawBits(); val s = (b ushr 16) and 0x8000
        val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF
        if (e <= 0) return s; if (e >= 31) return s or 0x7C00
        return s or (e shl 10) or (m ushr 13)
    }
    private fun le16(b: ByteArray, o: Int, h: Int) { b[o] = (h and 0xFF).toByte(); b[o + 1] = ((h ushr 8) and 0xFF).toByte() }

    /** Random block-major Q5_1 bytes for [out,in] + the FP32 weight they dequantize to (row-major). */
    private fun q5_1(inDim: Int, outDim: Int, rng: Random): Pair<ByteArray, FloatArray> {
        val blocks = inDim / 32; val bytes = ByteArray(outDim * blocks * 24); val wf = FloatArray(outDim * inDim)
        for (o in 0 until outDim) for (bI in 0 until blocks) {
            val off = (bI * outDim + o) * 24; val dst = o * inDim + bI * 32
            val d = rng.nextFloat() * 0.05f + 0.01f; val m = rng.nextFloat() - 0.5f
            le16(bytes, off, half(d)); le16(bytes, off + 2, half(m))
            val qh = IntArray(4) { rng.nextInt(256) }; for (k in 0 until 4) bytes[off + 4 + k] = qh[k].toByte()
            for (k in 0 until 16) bytes[off + 8 + k] = rng.nextInt(256).toByte()
            for (j in 0 until 16) {
                val q = bytes[off + 8 + j].toInt() and 0xFF
                val bl = (qh[j / 8] ushr (j % 8)) and 1; val bh = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 1
                wf[dst + j] = d * ((q and 0xF) + (bl shl 4)) + m; wf[dst + 16 + j] = d * ((q ushr 4) + (bh shl 4)) + m
            }
        }
        return bytes to wf
    }

    /** Random block-major Q4_K bytes for [out,in] + the FP32 weight. */
    private fun q4_k(inDim: Int, outDim: Int, rng: Random): Pair<ByteArray, FloatArray> {
        val blocks = inDim / 256; val bytes = ByteArray(outDim * blocks * 144); val wf = FloatArray(outDim * inDim)
        for (o in 0 until outDim) for (bI in 0 until blocks) {
            val off = (bI * outDim + o) * 144; val dst = o * inDim + bI * 256
            val d = rng.nextFloat() * 0.02f + 0.005f; val dMin = rng.nextFloat() * 0.02f + 0.005f
            le16(bytes, off, half(d)); le16(bytes, off + 2, half(dMin))
            for (k in 0 until 140) bytes[off + 4 + k] = rng.nextInt(256).toByte()
            val sc = off + 4; val si = IntArray(8); val mi = IntArray(8)
            for (s in 0 until 4) { si[s] = bytes[sc + s].toInt() and 0x3F; mi[s] = bytes[sc + s + 4].toInt() and 0x3F }
            for (s in 4 until 8) {
                si[s] = (bytes[sc + s + 4].toInt() and 0x0F) or (((bytes[sc + s - 4].toInt() and 0xFF) ushr 6) shl 4)
                mi[s] = ((bytes[sc + s + 4].toInt() and 0xFF) ushr 4) or (((bytes[sc + s].toInt() and 0xFF) ushr 6) shl 4)
            }
            val codes = off + 16
            for (g in 0 until 4) for (h in 0 until 2) {
                val s = 2 * g + h
                for (i in 0 until 32) {
                    val by = bytes[codes + g * 32 + i].toInt() and 0xFF
                    val code = if (h == 0) (by and 0x0F) else (by ushr 4)
                    wf[dst + s * 32 + i] = code * (d * si[s]) - dMin * mi[s]
                }
            }
        }
        return bytes to wf
    }

    /** Random block-major Q6_K bytes for [out,in] + the FP32 weight. */
    private fun q6_k(inDim: Int, outDim: Int, rng: Random): Pair<ByteArray, FloatArray> {
        val blocks = inDim / 256; val bytes = ByteArray(outDim * blocks * 210); val wf = FloatArray(outDim * inDim)
        for (o in 0 until outDim) for (bI in 0 until blocks) {
            val off = (bI * outDim + o) * 210; val dst = o * inDim + bI * 256
            for (k in 0 until 208) bytes[off + k] = rng.nextInt(256).toByte()
            val d = rng.nextFloat() * 0.01f + 0.002f; le16(bytes, off + 208, half(d))
            for (h in 0..1) {
                val qlB = off + h * 64; val qhB = off + 128 + h * 32; val scB = off + 192 + h * 8; val ob = h * 128
                for (isIdx in 0..1) {
                    val sc1 = d * bytes[scB + isIdx].toInt(); val sc2 = d * bytes[scB + isIdx + 2].toInt()
                    val sc3 = d * bytes[scB + isIdx + 4].toInt(); val sc4 = d * bytes[scB + isIdx + 6].toInt()
                    for (l in isIdx * 16 until isIdx * 16 + 16) {
                        val ql0 = bytes[qlB + l].toInt() and 0xFF; val ql32 = bytes[qlB + l + 32].toInt() and 0xFF
                        val qhL = bytes[qhB + l].toInt() and 0xFF
                        wf[dst + ob + l + 0] = sc1 * (((ql0 and 0xF) or ((qhL and 3) shl 4)) - 32)
                        wf[dst + ob + l + 32] = sc2 * (((ql32 and 0xF) or (((qhL ushr 2) and 3) shl 4)) - 32)
                        wf[dst + ob + l + 64] = sc3 * (((ql0 ushr 4) or (((qhL ushr 4) and 3) shl 4)) - 32)
                        wf[dst + ob + l + 96] = sc4 * (((ql32 ushr 4) or (((qhL ushr 6) and 3) shl 4)) - 32)
                    }
                }
            }
        }
        return bytes to wf
    }

    private fun run(fmt: String, inDim: Int, outDim: Int, seed: Int) {
        val rng = Random(seed)
        val (bytes, wf) = when (fmt) {
            "Q5_1" -> q5_1(inDim, outDim, rng)
            "Q6_K" -> q6_k(inDim, outDim, rng)
            else -> q4_k(inDim, outDim, rng)
        }
        @Suppress("UNCHECKED_CAST")
        val w = ctx.fromData(
            (when (fmt) {
                "Q5_1" -> Q5_1BlockTensorData(Shape(outDim, inDim), bytes)
                "Q6_K" -> Q6_KBlockTensorData(Shape(outDim, inDim), bytes)
                else -> Q4_KBlockTensorData(Shape(outDim, inDim), bytes)
            }) as TensorData<FP32, Float>,
            FP32::class,
        )
        val xf = FloatArray(inDim) { rng.nextFloat() - 0.5f }
        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, inDim), FP32::class, xf)
        val out = ctx.ops.matmul(x, ctx.ops.transpose(w)).data.copyToFloatArray()
        val expected = FloatArray(outDim) { o -> var s = 0f; for (i in 0 until inDim) s += xf[i] * wf[o * inDim + i]; s }
        var maxErr = 0f; var maxAbs = 1f
        for (o in 0 until outDim) { maxErr = maxOf(maxErr, abs(expected[o] - out[o])); maxAbs = maxOf(maxAbs, abs(expected[o])) }
        assertTrue(maxErr < 5e-3f * maxAbs, "$fmt e2e matmul deviates: maxErr=$maxErr (maxAbs=$maxAbs)")
    }

    @Test fun q5_1_through_ops_matmul_transpose() = run("Q5_1", inDim = 128, outDim = 16, seed = 7)
    @Test fun q4_k_through_ops_matmul_transpose() = run("Q4_K", inDim = 256, outDim = 12, seed = 8)
    @Test fun q6_k_through_ops_matmul_transpose() = run("Q6_K", inDim = 512, outDim = 8, seed = 9)
}
