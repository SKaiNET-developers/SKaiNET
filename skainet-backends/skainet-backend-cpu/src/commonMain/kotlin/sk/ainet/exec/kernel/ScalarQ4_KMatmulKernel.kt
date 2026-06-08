package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q4KMatmulKernel

/**
 * Scalar reference [Q4KMatmulKernel] — commonMain, so Q4_K packed matmul works
 * on Kotlin/Native / JS / WASM, not only the JVM SIMD path.
 *
 * Q4_K super-block: 256 elements / 144 bytes, block-major `(blockIdx*outputDim+o)*144`:
 * `d`(f16) `dMin`(f16) 12 scale bytes (ggml `get_scale_min_k4` packing) 128 code bytes.
 * Each of the 8 sub-blocks (32 elts) contributes `codeSum*scale - inputSum*offset`,
 * with `scale = d*scaleIdx`, `offset = dMin*minIdx`. Math mirrors
 * `JvmQuantizedVectorKernels.matmulQ4_KVec` / `DequantOps.dequantQ4KFromBytes`.
 */
public object ScalarQ4_KMatmulKernel : Q4KMatmulKernel {

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
            "ScalarQ4_KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
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

                // ggml get_scale_min_k4 over the 12 scale bytes.
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
                val inBlockBase = inputOffset + blockIdx * BLOCK_SIZE
                for (groupJ in 0 until 4) {
                    val qsRegion = codesOffset + groupJ * 32
                    // sub-block lo (low nibbles) then hi (high nibbles) of the same 32 bytes.
                    for (half in 0 until 2) {
                        val sb = 2 * groupJ + half
                        val inStart = inBlockBase + sb * SUB_BLOCK
                        var codeSum = 0f
                        var inputSum = 0f
                        for (i in 0 until 32) {
                            val b = weight[qsRegion + i].toInt() and 0xFF
                            val code = if (half == 0) (b and 0x0F) else (b ushr 4)
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
