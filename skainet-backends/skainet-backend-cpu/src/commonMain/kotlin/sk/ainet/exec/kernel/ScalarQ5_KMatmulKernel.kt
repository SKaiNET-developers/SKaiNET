package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q5KMatmulKernel

/**
 * Scalar reference [Q5KMatmulKernel] — commonMain, so Q5_K packed matmul works
 * on Kotlin/Native / JS / WASM, not only the JVM SIMD path.
 *
 * Q5_K super-block: 256 elements / 176 bytes, block-major `(blockIdx*outputDim+o)*176`:
 * `d`(f16) `dMin`(f16) 12 scale bytes (ggml `get_scale_min_k4` packing) 32 `qh`
 * high-bit bytes 128 `qs` low-nibble bytes. Each of the 8 sub-blocks (32 elts)
 * contributes `codeSum*scale - inputSum*offset`, with `scale = d*scaleIdx`,
 * `offset = dMin*minIdx`, and the 5-bit `code = lowNibble | (fifthBit << 4)`.
 * Math mirrors `DequantOps.dequantQ5KFromBytes` and the Q4_K kernel (only the
 * 5th-bit fold differs).
 */
public object ScalarQ5_KMatmulKernel : Q5KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val SUB_BLOCK = 32
    private const val BYTES_PER_BLOCK = 176
    private const val QH_OFFSET = 16
    private const val QS_OFFSET = 48

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "ScalarQ5_KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }
        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val scaleIdx = IntArray(8)
        val minIdx = IntArray(8)

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                val d = decodeHalf(((weight[blockBase + 1].toInt() and 0xFF) shl 8) or (weight[blockBase].toInt() and 0xFF))
                val dMin = decodeHalf(((weight[blockBase + 3].toInt() and 0xFF) shl 8) or (weight[blockBase + 2].toInt() and 0xFF))

                // ggml get_scale_min_k4 over the 12 scale bytes (identical to Q4_K).
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

                val qhBase = blockBase + QH_OFFSET
                val qsBase = blockBase + QS_OFFSET
                val inBlockBase = inputOffset + blockIdx * BLOCK_SIZE
                for (groupJ in 0 until 4) {
                    val qsRegion = qsBase + groupJ * 32
                    // sub-block lo (low nibbles) then hi (high nibbles) of the same 32 bytes;
                    // the 5th bit comes from qh[i], bit (2*groupJ + half).
                    for (half in 0 until 2) {
                        val sb = 2 * groupJ + half
                        val bit = 2 * groupJ + half
                        val inStart = inBlockBase + sb * SUB_BLOCK
                        var codeSum = 0f
                        var inputSum = 0f
                        for (i in 0 until 32) {
                            val b = weight[qsRegion + i].toInt() and 0xFF
                            val low = if (half == 0) (b and 0x0F) else (b ushr 4)
                            val fifth = ((weight[qhBase + i].toInt() and 0xFF) ushr bit) and 0x01
                            val code = low or (fifth shl 4)
                            val v = input[inStart + i]
                            codeSum += v * code
                            inputSum += v
                        }
                        acc += codeSum * (d * scaleIdx[sb]) - inputSum * (dMin * minIdx[sb])
                    }
                }
            }
            output[outputOffset + o] = acc
        }
    }
}
