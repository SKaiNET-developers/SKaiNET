package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q6KMatmulKernel

/**
 * Scalar reference [Q6KMatmulKernel] — commonMain, so Q6_K packed matmul works
 * on Kotlin/Native / JS / WASM, not only the JVM SIMD path.
 *
 * Q6_K super-block: 256 elements / 210 bytes, block-major `(blockIdx*outputDim+o)*210`:
 * `ql[128]` (low 4 bits) `qh[64]` (high 2 bits) `scales[16]` (int8) `d`(f16).
 * Each block is dequantized to 256 floats (matching the scalar path of
 * `JvmQuantizedVectorKernels.dequantQ6_KBlock` / `DequantOps.dequantQ6KFromBytes`)
 * and dotted with the matching input window.
 */
public object ScalarQ6_KMatmulKernel : Q6KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val BYTES_PER_BLOCK = 210

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "ScalarQ6_KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }
        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val scratch = FloatArray(BLOCK_SIZE)

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                dequantBlock(weight, blockBase, scratch)
                val inBase = inputOffset + blockIdx * BLOCK_SIZE
                for (i in 0 until BLOCK_SIZE) acc += input[inBase + i] * scratch[i]
            }
            output[outputOffset + o] = acc
        }
    }

    /** Dequant one 256-element Q6_K block into [scratch]. Mirrors the scalar path of ggml `dequantize_row_q6_K`. */
    private fun dequantBlock(w: ByteArray, blockBase: Int, scratch: FloatArray) {
        val qlBase0 = blockBase
        val qhBase0 = blockBase + 128
        val scBase0 = blockBase + 192
        val d = decodeHalf(((w[blockBase + 209].toInt() and 0xFF) shl 8) or (w[blockBase + 208].toInt() and 0xFF))

        for (half in 0..1) {
            val qlBase = qlBase0 + half * 64
            val qhBase = qhBase0 + half * 32
            val scBase = scBase0 + half * 8
            val outBase = half * 128
            for (isIdx in 0..1) {
                val sc1 = d * w[scBase + isIdx + 0].toInt()
                val sc2 = d * w[scBase + isIdx + 2].toInt()
                val sc3 = d * w[scBase + isIdx + 4].toInt()
                val sc4 = d * w[scBase + isIdx + 6].toInt()
                val lStart = isIdx * 16
                for (l in lStart until lStart + 16) {
                    val ql0 = w[qlBase + l].toInt() and 0xFF
                    val ql32 = w[qlBase + l + 32].toInt() and 0xFF
                    val qhL = w[qhBase + l].toInt() and 0xFF
                    val q1 = ((ql0 and 0x0F) or ((qhL and 0x03) shl 4)) - 32
                    val q2 = ((ql32 and 0x0F) or (((qhL ushr 2) and 0x03) shl 4)) - 32
                    val q3 = ((ql0 ushr 4) or (((qhL ushr 4) and 0x03) shl 4)) - 32
                    val q4 = ((ql32 ushr 4) or (((qhL ushr 6) and 0x03) shl 4)) - 32
                    scratch[outBase + l + 0] = sc1 * q1
                    scratch[outBase + l + 32] = sc2 * q2
                    scratch[outBase + l + 64] = sc3 * q3
                    scratch[outBase + l + 96] = sc4 * q4
                }
            }
        }
    }
}
