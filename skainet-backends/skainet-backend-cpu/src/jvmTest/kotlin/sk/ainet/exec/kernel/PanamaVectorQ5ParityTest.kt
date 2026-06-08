package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/** Panama SIMD Q5_1/Q5_0 kernels must match the scalar reference within FMA tolerance. */
class PanamaVectorQ5ParityTest {

    private fun half(v: Float): Int {
        val b = v.toRawBits(); val s = (b ushr 16) and 0x8000
        val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF
        if (e <= 0) return s; if (e >= 31) return s or 0x7C00
        return s or (e shl 10) or (m ushr 13)
    }

    /** Block-major packed bytes with VALID (finite) f16 scales; random qh/qs codes. */
    private fun bytes(bpb: Int, inDim: Int, outDim: Int, rng: Random): ByteArray {
        val out = ByteArray(outDim * (inDim / 32) * bpb)
        var off = 0
        while (off < out.size) {
            val d = half(rng.nextFloat() * 0.05f + 0.01f)
            out[off] = (d and 0xFF).toByte(); out[off + 1] = ((d ushr 8) and 0xFF).toByte()
            var codeStart = off + 2
            if (bpb == 24) { // Q5_1 has a per-block min `m`
                val m = half(rng.nextFloat() - 0.5f)
                out[off + 2] = (m and 0xFF).toByte(); out[off + 3] = ((m ushr 8) and 0xFF).toByte()
                codeStart = off + 4
            }
            for (k in codeStart until off + bpb) out[k] = rng.nextInt(256).toByte()
            off += bpb
        }
        return out
    }

    private fun check(q5_1: Boolean, inDim: Int, outDim: Int, seed: Int) {
        val rng = Random(seed)
        val w = bytes(if (q5_1) 24 else 22, inDim, outDim, rng)
        val input = FloatArray(inDim) { rng.nextFloat() - 0.5f }
        val a = FloatArray(outDim); val b = FloatArray(outDim)
        if (q5_1) {
            ScalarQ5_1MatmulKernel.matmul(input, 0, w, 0, inDim, outDim, a, 0)
            PanamaVectorQ5_1MatmulKernel.matmul(input, 0, w, 0, inDim, outDim, b, 0)
        } else {
            ScalarQ5_0MatmulKernel.matmul(input, 0, w, 0, inDim, outDim, a, 0)
            PanamaVectorQ5_0MatmulKernel.matmul(input, 0, w, 0, inDim, outDim, b, 0)
        }
        var maxErr = 0f; var maxAbs = 1f
        for (o in 0 until outDim) { maxErr = maxOf(maxErr, abs(a[o] - b[o])); maxAbs = maxOf(maxAbs, abs(a[o])) }
        assertTrue(maxErr < 1e-4f * maxAbs + 1e-4f, "${if (q5_1) "Q5_1" else "Q5_0"} Panama≠Scalar: maxErr=$maxErr (maxAbs=$maxAbs)")
    }

    @Test fun q5_1_panama_matches_scalar() = check(true, 256, 64, 1)
    @Test fun q5_0_panama_matches_scalar() = check(false, 256, 48, 2)
}
