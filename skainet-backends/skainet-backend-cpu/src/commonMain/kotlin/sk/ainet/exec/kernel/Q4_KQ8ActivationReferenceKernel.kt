package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import kotlin.math.abs

/**
 * Test-support reference for the native/FFM/Kotlin-Native/JNI Q4_K kernels'
 * ggml-style int8 activation-quantization fast path (`skainet_q4k_matmul` in
 * `q4k_matmul.c`) — deliberately NOT [ScalarQ4_KMatmulKernel]'s exact-float
 * algorithm.
 *
 * [ScalarQ4_KMatmulKernel] and the native-family kernels compute genuinely
 * different things (#944): the native side quantizes the input activation to
 * int8 first (ggml's `block_q8_K`, `d_in = maxabs/127`, symmetric round +
 * clamp to [-127,127]) before an integer dot product against the 4-bit
 * weight codes, then applies the block's `d`/`dMin` scale/min — a small,
 * deliberate, expected source of divergence against the exact-float scalar
 * kernel that per-row or RMS-energy tolerances have to absorb. Comparing a
 * native kernel's output against *this* kernel instead — same activation
 * quantization, same integer dot, same scale/min application, transcribed
 * byte-for-byte from the C — isolates genuine kernel bugs (wrong offsets,
 * layout, scale decode, dispatch) from that expected loss: agreement should
 * be tight (float-accumulation-order / rounding-tie noise only), not
 * activation-quant noise. Use a tight tolerance against this kernel; keep
 * using the RMS-energy gate against [ScalarQ4_KMatmulKernel] for the
 * "is the intended lossy path still within its expected budget" check.
 *
 * Not registered with any [sk.ainet.backend.api.kernel.KernelRegistry] and
 * not tuned for speed (allocates per call) — test support only, never a
 * production dispatch target.
 */
public object Q4_KQ8ActivationReferenceKernel : Q4KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val SUB_BLOCK = 32
    private const val BYTES_PER_BLOCK = 144

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "Q4_KQ8ActivationReferenceKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }

        val blocksPerInputDim = inputDim / BLOCK_SIZE
        // Pre-quantize the whole input row to Q8 once, mirroring the C
        // kernel's single quantization pass reused across every output row.
        val q8 = IntArray(inputDim)
        val dIn = FloatArray(blocksPerInputDim)
        for (b in 0 until blocksPerInputDim) {
            val base = inputOffset + b * BLOCK_SIZE
            var maxAbs = 0f
            for (i in 0 until BLOCK_SIZE) {
                val a = abs(input[base + i])
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

        val scaleIdx = IntArray(8)
        val minIdx = IntArray(8)
        for (o in 0 until outputDim) output[outputOffset + o] = 0f

        for (blockIdx in 0 until blocksPerInputDim) {
            val di = dIn[blockIdx]
            val q8Base = blockIdx * BLOCK_SIZE

            for (o in 0 until outputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                val d = decodeHalf(((weight[blockBase + 1].toInt() and 0xFF) shl 8) or (weight[blockBase].toInt() and 0xFF))
                val dMin = decodeHalf(((weight[blockBase + 3].toInt() and 0xFF) shl 8) or (weight[blockBase + 2].toInt() and 0xFF))

                // ggml get_scale_min_k4 over the 12 scale bytes — identical to
                // ScalarQ4_KMatmulKernel and skainet_q4k_decode_scales.
                val sc = blockBase + 4
                for (sb in 0 until 4) {
                    scaleIdx[sb] = weight[sc + sb].toInt() and 0x3F
                    minIdx[sb] = weight[sc + sb + 4].toInt() and 0x3F
                }
                for (sb in 4 until 8) {
                    val low4S = weight[sc + sb + 4].toInt() and 0x0F
                    val high2S = (weight[sc + sb - 4].toInt() and 0xFF) ushr 6
                    scaleIdx[sb] = low4S or (high2S shl 4)
                    val low4M = (weight[sc + sb + 4].toInt() and 0xFF) ushr 4
                    val high2M = (weight[sc + sb].toInt() and 0xFF) ushr 6
                    minIdx[sb] = low4M or (high2M shl 4)
                }

                val codesOffset = blockBase + 16
                var blockScaleDot = 0L
                var blockMinSum = 0L
                for (groupJ in 0 until 4) {
                    val qsRegion = codesOffset + groupJ * 32
                    val sbLo = 2 * groupJ
                    val sbHi = sbLo + 1
                    val q8LoBase = q8Base + sbLo * SUB_BLOCK
                    val q8HiBase = q8Base + sbHi * SUB_BLOCK
                    var dotLo = 0
                    var sumLo = 0
                    var dotHi = 0
                    var sumHi = 0
                    for (i in 0 until SUB_BLOCK) {
                        val b = weight[qsRegion + i].toInt() and 0xFF
                        val codeLo = b and 0x0F
                        val codeHi = b ushr 4
                        val aLo = q8[q8LoBase + i]
                        val aHi = q8[q8HiBase + i]
                        dotLo += aLo * codeLo
                        sumLo += aLo
                        dotHi += aHi * codeHi
                        sumHi += aHi
                    }
                    blockScaleDot += scaleIdx[sbLo].toLong() * dotLo + scaleIdx[sbHi].toLong() * dotHi
                    blockMinSum += minIdx[sbLo].toLong() * sumLo + minIdx[sbHi].toLong() * sumHi
                }

                output[outputOffset + o] += di * (d * blockScaleDot.toFloat() - dMin * blockMinSum.toFloat())
            }
        }
    }

    /** Round-half-away-from-zero, matching `lrintf`'s effect for the non-tie floats real inputs produce. */
    private fun roundHalfAwayFromZero(x: Float): Int =
        if (x >= 0f) (x + 0.5f).toInt() else -(-x + 0.5f).toInt()
}
