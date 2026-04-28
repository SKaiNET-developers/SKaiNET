package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Fp32MatmulKernel

/**
 * Scalar reference implementation of [Fp32MatmulKernel] — three nested
 * loops, no SIMD. Always available on every KMP target. Used as:
 *
 * - The correctness reference that accelerated kernels (Panama, native)
 *   must match bit-for-bit (within FP order tolerance).
 * - A guaranteed fallback when no accelerated provider is registered or
 *   available.
 *
 * Performance is modest (no vectorization, no cache-blocking), so
 * production code should layer a Panama or native provider on top via
 * [sk.ainet.backend.api.kernel.KernelRegistry].
 */
public object ScalarMatmulKernel : Fp32MatmulKernel {
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "ScalarMatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return
        // k == 0 → C = 0; the strides may still be > 0 so we need to
        // explicitly zero the output block.
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
                var sum = 0f
                for (kk in 0 until k) {
                    sum += a[aRowOff + kk] * b[bOffset + kk * bStride + j]
                }
                out[outRowOff + j] = sum
            }
        }
    }
}
