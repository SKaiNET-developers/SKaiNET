package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/** Panama SIMD Q6_K kernel must match the scalar reference within FMA tolerance. */
class PanamaVectorQ6KParityTest {

    private fun half(v: Float): Int {
        val b = v.toRawBits(); val s = (b ushr 16) and 0x8000
        val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF
        if (e <= 0) return s; if (e >= 31) return s or 0x7C00
        return s or (e shl 10) or (m ushr 13)
    }

    /** Block-major Q6_K bytes (210 B/block) with a valid finite f16 scale; random ql/qh/scales. */
    private fun bytes(inDim: Int, outDim: Int, rng: Random): ByteArray {
        val out = ByteArray(outDim * (inDim / 256) * 210)
        var off = 0
        while (off < out.size) {
            for (k in 0 until 208) out[off + k] = rng.nextInt(256).toByte() // ql + qh + scales
            val d = half(rng.nextFloat() * 0.01f + 0.002f)
            out[off + 208] = (d and 0xFF).toByte(); out[off + 209] = ((d ushr 8) and 0xFF).toByte()
            off += 210
        }
        return out
    }

    private fun check(inDim: Int, outDim: Int, seed: Int) {
        val rng = Random(seed)
        val w = bytes(inDim, outDim, rng)
        val input = FloatArray(inDim) { rng.nextFloat() - 0.5f }
        val a = FloatArray(outDim); val b = FloatArray(outDim)
        ScalarQ6_KMatmulKernel.matmul(input, 0, w, 0, inDim, outDim, a, 0)
        PanamaVectorQ6_KMatmulKernel.matmul(input, 0, w, 0, inDim, outDim, b, 0)
        var maxErr = 0f; var maxAbs = 1f
        for (o in 0 until outDim) { maxErr = maxOf(maxErr, abs(a[o] - b[o])); maxAbs = maxOf(maxAbs, abs(a[o])) }
        assertTrue(maxErr < 1e-3f * maxAbs + 1e-3f, "Q6_K Panama≠Scalar: maxErr=$maxErr (maxAbs=$maxAbs)")
    }

    @Test fun q6_k_panama_matches_scalar_single() = check(256, 32, 1)
    @Test fun q6_k_panama_matches_scalar_multi() = check(512, 16, 2)
}
