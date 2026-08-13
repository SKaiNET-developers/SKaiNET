package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q6KMatmulKernel

/**
 * Test-support reference for the native/FFM/Kotlin-Native/JNI Q6_K kernels'
 * ggml-style int8 activation-quantization fast path (`skainet_q6k_matmul` in
 * `q6k_matmul.c`) — deliberately NOT [ScalarQ6_KMatmulKernel]'s exact-float
 * algorithm.
 *
 * Same rationale as [Q4_KQ8ActivationReferenceKernel] (see its kdoc and
 * #944): the native side quantizes the activation to int8 first
 * (`d_in = maxabs/127`), unpacks the 6-bit weight code to a centered int8
 * `code - 32`, and does 16 int8 dot-products per block (one per scale
 * group) instead of an exact float dequant-then-dot. Comparing a native
 * kernel against this transcription instead of the exact-float scalar
 * reference isolates genuine kernel bugs from the expected, small
 * activation-quantization loss — use a tight tolerance here, keep the
 * RMS-energy gate against [ScalarQ6_KMatmulKernel].
 *
 * Not registered with any [sk.ainet.backend.api.kernel.KernelRegistry] and
 * not tuned for speed — test support only.
 */
public object Q6_KQ8ActivationReferenceKernel : Q6KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val BYTES_PER_BLOCK = 210
    private const val QL_OFFSET = 0
    private const val QH_OFFSET = 128
    private const val SCALES_OFFSET = 192
    private const val D_OFFSET = 208

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "Q6_KQ8ActivationReferenceKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }

        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val q8 = IntArray(inputDim)
        val dIn = FloatArray(blocksPerInputDim)
        for (b in 0 until blocksPerInputDim) {
            val base = inputOffset + b * BLOCK_SIZE
            var maxAbs = 0f
            for (i in 0 until BLOCK_SIZE) {
                val a = kotlin.math.abs(input[base + i])
                if (a > maxAbs) maxAbs = a
            }
            if (maxAbs == 0f) {
                dIn[b] = 0f
                for (i in 0 until BLOCK_SIZE) q8[b * BLOCK_SIZE + i] = 0
                continue
            }
            val inv = 127f / maxAbs
            dIn[b] = maxAbs / 127f
            for (i in 0 until BLOCK_SIZE) {
                var v = roundHalfAwayFromZero(input[base + i] * inv)
                if (v > 127) v = 127 else if (v < -127) v = -127
                q8[b * BLOCK_SIZE + i] = v
            }
        }

        val codes = IntArray(BLOCK_SIZE)
        for (o in 0 until outputDim) output[outputOffset + o] = 0f

        for (blockIdx in 0 until blocksPerInputDim) {
            val di = dIn[blockIdx]
            val q8Base = blockIdx * BLOCK_SIZE

            for (o in 0 until outputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                val d = decodeHalf(
                    ((weight[blockBase + D_OFFSET + 1].toInt() and 0xFF) shl 8) or
                        (weight[blockBase + D_OFFSET].toInt() and 0xFF),
                )

                unpackCodes(weight, blockBase, codes)

                // Σ_g sc[g] · Σ_{i∈g} q8[i]·codes[i], over 16 contiguous 16-element
                // scale groups — matches skainet_q6k_weighted_dot_generic exactly
                // (same group start/scale-index formula, same accumulation order).
                var wdot = 0L
                for (half in 0 until 2) {
                    for (k in 0 until 4) {
                        for (isb in 0 until 2) {
                            val start = half * 128 + 32 * k + isb * 16
                            val gs = half * 8 + isb + 2 * k
                            val sc = weight[blockBase + SCALES_OFFSET + gs].toInt() // signed int8
                            var dot = 0
                            for (j in 0 until 16) {
                                dot += q8[q8Base + start + j] * codes[start + j]
                            }
                            wdot += sc.toLong() * dot
                        }
                    }
                }

                output[outputOffset + o] += d * di * wdot.toFloat()
            }
        }
    }

    /**
     * Unpack one 256-element Q6_K block into centered int8 codes (`code - 32`,
     * range [-32,31]) in natural element order. Byte-for-byte transcription of
     * `skainet_q6k_unpack_codes`.
     */
    private fun unpackCodes(weight: ByteArray, blockBase: Int, codes: IntArray) {
        val ql0 = blockBase + QL_OFFSET
        val qh0 = blockBase + QH_OFFSET
        for (half in 0 until 2) {
            val ql = ql0 + half * 64
            val qh = qh0 + half * 32
            val out = half * 128
            for (isb in 0 until 2) {
                val lStart = isb * 16
                for (l in lStart until lStart + 16) {
                    val qL0 = weight[ql + l].toInt() and 0xFF
                    val qL32 = weight[ql + l + 32].toInt() and 0xFF
                    val qH = weight[qh + l].toInt() and 0xFF
                    codes[out + l + 0] = ((qL0 and 0x0F) or ((qH and 0x03) shl 4)) - 32
                    codes[out + l + 32] = ((qL32 and 0x0F) or (((qH ushr 2) and 0x03) shl 4)) - 32
                    codes[out + l + 64] = ((qL0 ushr 4) or (((qH ushr 4) and 0x03) shl 4)) - 32
                    codes[out + l + 96] = ((qL32 ushr 4) or (((qH ushr 6) and 0x03) shl 4)) - 32
                }
            }
        }
    }

    private fun roundHalfAwayFromZero(x: Float): Int =
        if (x >= 0f) (x + 0.5f).toInt() else -(-x + 0.5f).toInt()
}
