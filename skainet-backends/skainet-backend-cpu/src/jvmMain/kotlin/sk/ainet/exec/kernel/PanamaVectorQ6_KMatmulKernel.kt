package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Q6KMatmulKernel
import sk.ainet.exec.tensor.ops.JvmQuantizedVectorKernels

/**
 * SIMD-vectorized FP32 × Q6_K matmul on the JDK Vector API. Reuses the existing SIMD
 * Q6_K block dequant ([JvmQuantizedVectorKernels.dequantQ6_KBlock]) into a 256-element
 * scratch buffer, then a Vector-API FMA dot against the matching input window.
 * Numerically equivalent to [ScalarQ6_KMatmulKernel]. Block-major layout
 * `(blockIdx*outputDim+o)*210`.
 */
public object PanamaVectorQ6_KMatmulKernel : Q6KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val BYTES_PER_BLOCK = 210
    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "PanamaVectorQ6_KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }
        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(BLOCK_SIZE)
        val scratch = FloatArray(BLOCK_SIZE)

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val base = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                JvmQuantizedVectorKernels.dequantQ6_KBlock(weight, base, scratch, 0)
                val inputBase = inputOffset + blockIdx * BLOCK_SIZE
                var accVec = FloatVector.zero(floatSpecies)
                var k = 0
                while (k < loopBound) {
                    accVec = FloatVector.fromArray(floatSpecies, input, inputBase + k)
                        .fma(FloatVector.fromArray(floatSpecies, scratch, k), accVec)
                    k += step
                }
                acc += accVec.reduceLanes(VectorOperators.ADD)
                while (k < BLOCK_SIZE) { acc += input[inputBase + k] * scratch[k]; k++ }
            }
            output[outputOffset + o] = acc
        }
    }
}
