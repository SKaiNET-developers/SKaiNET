package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q4_0MatmulKernel

/**
 * Scalar reference implementation of [Q4_0MatmulKernel] — straight
 * per-block dequant + per-element FMA, no SIMD. Always available on
 * every KMP target. Used as:
 *
 * - The correctness reference that accelerated kernels (Panama Vector,
 *   native FFM) must match within FP order tolerance.
 * - A guaranteed fallback when no accelerated provider is registered.
 *
 * Block layout (32-element block, 18 bytes):
 *   - bytes 0..1 : FP16 little-endian scale (`d`)
 *   - bytes 2..17: 16 bytes packing 32 4-bit codes (split layout)
 *
 * Dequant per element: `(code - 8) * d`. No min / offset.
 *
 * Performance is intentionally modest; production paths should pick the
 * Panama Vector or native variant via the kernel registry.
 */
public object ScalarQ4_0MatmulKernel : Q4_0MatmulKernel {

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 18

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "ScalarQ4_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) {
            if (outputDim > 0) {
                for (o in 0 until outputDim) output[outputOffset + o] = 0f
            }
            return
        }
        val blocksPerInputDim = inputDim / BLOCK_SIZE

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                // FP16 scale: two LE bytes.
                val dBits = (weight[blockBase].toInt() and 0xFF) or
                    ((weight[blockBase + 1].toInt() and 0xFF) shl 8)
                val d = halfToFloat(dBits)
                // 32 codes, blockIdx-th window of the input vector. Split
                // layout: low nibbles → elements 0..15, high → 16..31.
                val inputBase = inputOffset + blockIdx * BLOCK_SIZE
                val codesBase = blockBase + 2
                for (j in 0 until 16) {
                    val b = weight[codesBase + j].toInt() and 0xFF
                    val lo = (b and 0x0F) - 8
                    val hi = (b ushr 4) - 8
                    acc += input[inputBase + j] * lo * d
                    acc += input[inputBase + 16 + j] * hi * d
                }
            }
            output[outputOffset + o] = acc
        }
    }

    /**
     * Convert a 16-bit IEEE-754 half-precision value (low 16 bits of
     * [hbits]) to FP32. Mirrors [ScalarQ8_0MatmulKernel]'s inlined helper
     * — the skainet-lang-core dequant helper is internal to that module.
     */
    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits and 0x8000) shl 16
        val exp = (hbits and 0x7C00) shr 10
        val mant = hbits and 0x03FF
        return when (exp) {
            0 -> {
                if (mant == 0) Float.fromBits(sign)
                else {
                    var m = mant
                    var e = -14
                    while ((m and 0x400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                }
            }
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
