package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Verifies the commonMain scalar packed-quant matmul kernels (Q5_1/Q5_0/Q4_K/Q6_K)
 * against an independent inline dequant: for block-major packed bytes, the kernel's
 * `Σ input·dequant` must match a reference that dequantizes the same bytes and does the
 * dot in FP32. Runs on every target the module is compiled for (JVM + linuxX64), proving
 * Native packed-matmul correctness, not only JVM.
 */
class ScalarPackedKernelParityTest {

    private fun half(v: Float): Int {
        val bits = v.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val expo = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        if (expo <= 0) return sign
        if (expo >= 31) return sign or 0x7C00
        return sign or (expo shl 10) or (mant ushr 13)
    }

    private fun le16(b: ByteArray, off: Int, h: Int) {
        b[off] = (h and 0xFF).toByte(); b[off + 1] = ((h ushr 8) and 0xFF).toByte()
    }

    /** Build random block-major bytes + the row-major [out,in] FP32 weight they dequantize to. */
    private fun build(
        fmt: String, inputDim: Int, outputDim: Int, rng: Random,
    ): Pair<ByteArray, FloatArray> {
        val bs = if (fmt == "Q4_K" || fmt == "Q6_K") 256 else 32
        val bpb = when (fmt) { "Q5_1" -> 24; "Q5_0" -> 22; "Q4_K" -> 144; else -> 210 }
        val blocks = inputDim / bs
        val bytes = ByteArray(outputDim * blocks * bpb)
        val wf = FloatArray(outputDim * inputDim)
        for (o in 0 until outputDim) for (bI in 0 until blocks) {
            val off = (bI * outputDim + o) * bpb
            val dst = o * inputDim + bI * bs
            when (fmt) {
                "Q5_1" -> blkQ5_1(bytes, off, wf, dst, rng)
                "Q5_0" -> blkQ5_0(bytes, off, wf, dst, rng)
                "Q4_K" -> blkQ4_K(bytes, off, wf, dst, rng)
                "Q6_K" -> blkQ6_K(bytes, off, wf, dst, rng)
            }
        }
        return bytes to wf
    }

    private fun blkQ5_1(b: ByteArray, off: Int, wf: FloatArray, dst: Int, rng: Random) {
        val d = (rng.nextFloat() * 0.05f + 0.01f); val m = (rng.nextFloat() - 0.5f)
        le16(b, off, half(d)); le16(b, off + 2, half(m))
        val qh = IntArray(4) { rng.nextInt(256) }; for (k in 0 until 4) b[off + 4 + k] = qh[k].toByte()
        for (k in 0 until 16) b[off + 8 + k] = rng.nextInt(256).toByte()
        for (j in 0 until 16) {
            val q = b[off + 8 + j].toInt() and 0xFF; val lo = q and 0xF; val hi = q ushr 4
            val bl = (qh[j / 8] ushr (j % 8)) and 1; val bh = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 1
            wf[dst + j] = d * (lo + (bl shl 4)) + m; wf[dst + 16 + j] = d * (hi + (bh shl 4)) + m
        }
    }

    private fun blkQ5_0(b: ByteArray, off: Int, wf: FloatArray, dst: Int, rng: Random) {
        val d = (rng.nextFloat() * 0.05f + 0.01f); le16(b, off, half(d))
        val qh = IntArray(4) { rng.nextInt(256) }; for (k in 0 until 4) b[off + 2 + k] = qh[k].toByte()
        for (k in 0 until 16) b[off + 6 + k] = rng.nextInt(256).toByte()
        for (j in 0 until 16) {
            val q = b[off + 6 + j].toInt() and 0xFF; val lo = q and 0xF; val hi = q ushr 4
            val bl = (qh[j / 8] ushr (j % 8)) and 1; val bh = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 1
            wf[dst + j] = d * (lo + (bl shl 4) - 16); wf[dst + 16 + j] = d * (hi + (bh shl 4) - 16)
        }
    }

    private fun blkQ4_K(b: ByteArray, off: Int, wf: FloatArray, dst: Int, rng: Random) {
        val d = rng.nextFloat() * 0.02f + 0.005f; val dMin = rng.nextFloat() * 0.02f + 0.005f
        le16(b, off, half(d)); le16(b, off + 2, half(dMin))
        for (k in 0 until 140) b[off + 4 + k] = rng.nextInt(256).toByte() // 12 scales + 128 codes
        val sc = off + 4; val scaleIdx = IntArray(8); val minIdx = IntArray(8)
        for (s in 0 until 4) { scaleIdx[s] = b[sc + s].toInt() and 0x3F; minIdx[s] = b[sc + s + 4].toInt() and 0x3F }
        for (s in 4 until 8) {
            scaleIdx[s] = (b[sc + s + 4].toInt() and 0x0F) or (((b[sc + s - 4].toInt() and 0xFF) ushr 6) shl 4)
            minIdx[s] = ((b[sc + s + 4].toInt() and 0xFF) ushr 4) or (((b[sc + s].toInt() and 0xFF) ushr 6) shl 4)
        }
        val codes = off + 16
        for (g in 0 until 4) for (half in 0 until 2) {
            val s = 2 * g + half; val scale = d * scaleIdx[s]; val offs = dMin * minIdx[s]
            for (i in 0 until 32) {
                val by = b[codes + g * 32 + i].toInt() and 0xFF
                val code = if (half == 0) (by and 0x0F) else (by ushr 4)
                wf[dst + s * 32 + i] = code * scale - offs
            }
        }
    }

    private fun blkQ6_K(b: ByteArray, off: Int, wf: FloatArray, dst: Int, rng: Random) {
        for (k in 0 until 208) b[off + k] = rng.nextInt(256).toByte() // ql+qh+scales
        val d = rng.nextFloat() * 0.01f + 0.002f; le16(b, off + 208, half(d))
        for (h in 0..1) {
            val qlB = off + h * 64; val qhB = off + 128 + h * 32; val scB = off + 192 + h * 8; val ob = h * 128
            for (isIdx in 0..1) {
                val sc1 = d * b[scB + isIdx].toInt(); val sc2 = d * b[scB + isIdx + 2].toInt()
                val sc3 = d * b[scB + isIdx + 4].toInt(); val sc4 = d * b[scB + isIdx + 6].toInt()
                for (l in isIdx * 16 until isIdx * 16 + 16) {
                    val ql0 = b[qlB + l].toInt() and 0xFF; val ql32 = b[qlB + l + 32].toInt() and 0xFF
                    val qhL = b[qhB + l].toInt() and 0xFF
                    wf[dst + ob + l + 0] = sc1 * (((ql0 and 0xF) or ((qhL and 3) shl 4)) - 32)
                    wf[dst + ob + l + 32] = sc2 * (((ql32 and 0xF) or (((qhL ushr 2) and 3) shl 4)) - 32)
                    wf[dst + ob + l + 64] = sc3 * (((ql0 ushr 4) or (((qhL ushr 4) and 3) shl 4)) - 32)
                    wf[dst + ob + l + 96] = sc4 * (((ql32 ushr 4) or (((qhL ushr 6) and 3) shl 4)) - 32)
                }
            }
        }
    }

    private fun check(fmt: String, inputDim: Int, outputDim: Int, seed: Int) {
        val rng = Random(seed)
        val (bytes, wf) = build(fmt, inputDim, outputDim, rng)
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val expected = FloatArray(outputDim) { o -> var s = 0f; for (i in 0 until inputDim) s += input[i] * wf[o * inputDim + i]; s }
        val actual = FloatArray(outputDim)
        when (fmt) {
            "Q5_1" -> ScalarQ5_1MatmulKernel.matmul(input, 0, bytes, 0, inputDim, outputDim, actual, 0)
            "Q5_0" -> ScalarQ5_0MatmulKernel.matmul(input, 0, bytes, 0, inputDim, outputDim, actual, 0)
            "Q4_K" -> ScalarQ4_KMatmulKernel.matmul(input, 0, bytes, 0, inputDim, outputDim, actual, 0)
            "Q6_K" -> ScalarQ6_KMatmulKernel.matmul(input, 0, bytes, 0, inputDim, outputDim, actual, 0)
        }
        var maxErr = 0f
        var maxAbs = 1f
        for (o in 0 until outputDim) {
            maxErr = maxOf(maxErr, abs(expected[o] - actual[o]))
            maxAbs = maxOf(maxAbs, abs(expected[o]))
        }
        // Relative tolerance: the kernel accumulates per sub-block (with a
        // codeSum*scale - inputSum*offset cancellation for Q4_K), so it differs
        // from the flat reference sum only by FP reassociation (~1e-3 rel).
        assertTrue(
            maxErr < 5e-3f * maxAbs,
            "$fmt scalar kernel deviates from inline dequant: maxErr=$maxErr (maxAbs=$maxAbs)",
        )
    }

    @Test fun q5_1() = check("Q5_1", inputDim = 128, outputDim = 16, seed = 1)
    @Test fun q5_0() = check("Q5_0", inputDim = 96, outputDim = 24, seed = 2)
    @Test fun q4_k() = check("Q4_K", inputDim = 256, outputDim = 12, seed = 3)
    @Test fun q6_k() = check("Q6_K", inputDim = 512, outputDim = 8, seed = 4)
}
