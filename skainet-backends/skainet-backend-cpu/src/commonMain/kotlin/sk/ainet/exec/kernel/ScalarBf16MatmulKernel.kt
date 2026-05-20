package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Bf16MatmulKernel

/**
 * Scalar reference implementation of [Bf16MatmulKernel] — three nested
 * loops, dequant inline (`bf16_bits << 16` reinterpret), no SIMD. Always
 * available on every KMP target. Used as:
 *
 * - The correctness reference that accelerated kernels (Panama Vector,
 *   native FFM) must match within FP order tolerance.
 * - A guaranteed fallback when no accelerated provider is registered.
 *
 * Performance is intentionally modest; production paths should pick the
 * Panama Vector or native variant via the kernel registry. The BF16
 * conversion math is identical across all impls — see the kdoc on
 * [Bf16MatmulKernel] for the bit-shift identity.
 */
public object ScalarBf16MatmulKernel : Bf16MatmulKernel {
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "ScalarBf16MatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
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
                var sum = 0f
                for (p in 0 until k) {
                    // BF16 layout: little-endian 16-bit value; the high
                    // 16 bits of the equivalent FP32 share the BF16 bit
                    // pattern, low 16 are zero. Read two bytes, shift
                    // left 16, reinterpret as float.
                    val bByteIdx = bByteOffset + p * bByteStride + j * 2
                    val lo = b[bByteIdx].toInt() and 0xFF
                    val hi = b[bByteIdx + 1].toInt() and 0xFF
                    val bFloat = Float.fromBits(((hi shl 8) or lo) shl 16)
                    sum += a[aRowOff + p] * bFloat
                }
                out[outRowOff + j] = sum
            }
        }
    }
}
