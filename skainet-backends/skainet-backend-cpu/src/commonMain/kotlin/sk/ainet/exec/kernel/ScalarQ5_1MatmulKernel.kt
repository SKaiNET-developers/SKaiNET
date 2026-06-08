package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Q5_1MatmulKernel

/**
 * Scalar reference [Q5_1MatmulKernel] — per-block dequant + per-element FMA,
 * no SIMD. Always available on every KMP target (commonMain), so Q5_1 packed
 * matmul works on Kotlin/Native, JS and WASM, not only the JVM.
 *
 * Block layout (32-elt, 24 B; block-major `(blockIdx*outputDim+o)*24`):
 * `d`(f16) `m`(f16) `qh[4]` `qs[16]`. Dequant matches
 * `DequantOps.dequantQ5_1FromBytes`: `w = d*(code + (highBit shl 4)) + m`.
 */
public object ScalarQ5_1MatmulKernel : Q5_1MatmulKernel {

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 24

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "ScalarQ5_1MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }
        val blocksPerInputDim = inputDim / BLOCK_SIZE

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val base = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                val d = decodeHalf(((weight[base + 1].toInt() and 0xFF) shl 8) or (weight[base].toInt() and 0xFF))
                val m = decodeHalf(((weight[base + 3].toInt() and 0xFF) shl 8) or (weight[base + 2].toInt() and 0xFF))
                val qh0 = weight[base + 4].toInt() and 0xFF
                val qh1 = weight[base + 5].toInt() and 0xFF
                val qh2 = weight[base + 6].toInt() and 0xFF
                val qh3 = weight[base + 7].toInt() and 0xFF
                val qsBase = base + 8
                val inputBase = inputOffset + blockIdx * BLOCK_SIZE
                for (j in 0 until 16) {
                    val q = weight[qsBase + j].toInt() and 0xFF
                    val lo = q and 0x0F
                    val hi = q ushr 4
                    val bitLo = (qh(qh0, qh1, qh2, qh3, j) ushr (j % 8)) and 0x01
                    val jh = j + 16
                    val bitHi = (qh(qh0, qh1, qh2, qh3, jh) ushr (jh % 8)) and 0x01
                    val wLo = d * (lo + (bitLo shl 4)) + m
                    val wHi = d * (hi + (bitHi shl 4)) + m
                    acc += input[inputBase + j] * wLo + input[inputBase + 16 + j] * wHi
                }
            }
            output[outputOffset + o] = acc
        }
    }

    private inline fun qh(q0: Int, q1: Int, q2: Int, q3: Int, idx: Int): Int =
        when (idx / 8) { 0 -> q0; 1 -> q1; 2 -> q2; else -> q3 }
}
