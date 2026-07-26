package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Fp16MatmulKernel
import sk.ainet.lang.types.Fp16Codec

/**
 * Scalar reference implementation of [Fp16MatmulKernel] — three nested loops, decode inline, no
 * SIMD. Always available on every KMP target, and the correctness reference an accelerated FP16
 * kernel must match within FP-ordering tolerance.
 *
 * Mirrors [ScalarBf16MatmulKernel]; the only difference is the decode, which for binary16 needs
 * exponent rebiasing rather than BF16's bit shift.
 */
public object ScalarFp16MatmulKernel : Fp16MatmulKernel {
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "ScalarFp16MatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return
        // k == 0 is an explicit "zero the output block" per the SPI.
        if (k == 0) {
            for (i in 0 until m) {
                val rowOff = outOffset + i * outStride
                for (j in 0 until n) out[rowOff + j] = 0f
            }
            return
        }
        for (i in 0 until m) {
            val aRowOff = aOffset + i * aStride
            val outRowOff = outOffset + i * outStride
            for (j in 0 until n) {
                // Accumulate in FP32 — the narrow format is storage only.
                var sum = 0f
                for (p in 0 until k) {
                    val bByteIdx = bByteOffset + p * bByteStride + j * 2
                    val lo = b[bByteIdx].toInt() and 0xFF
                    val hi = b[bByteIdx + 1].toInt() and 0xFF
                    sum += a[aRowOff + p] * Fp16Codec.decode((hi shl 8) or lo)
                }
                out[outRowOff + j] = sum
            }
        }
    }
}
