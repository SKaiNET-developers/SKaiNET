package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel

/**
 * SIMD-vectorized FP32 × Q4_0 matmul on the JDK Vector API.
 *
 * Pipeline per 32-element block:
 *  1. Decode the 2-byte FP16 scale `d` once.
 *  2. Unpack the 16 code bytes into 32 sign-corrected floats (`nibble - 8`)
 *     in a reusable scratch buffer, using the canonical ggml **split**
 *     layout (low nibbles → elements 0..15, high nibbles → 16..31). The
 *     nibble-pair-per-byte packing makes a fully-fused `ByteVector`
 *     pipeline awkward, so this kernel keeps the scratch-then-FMA shape
 *     (same approach as the legacy `JvmQuantizedVectorKernels` Q4_0 path).
 *  3. SIMD-FMA the scratch against the matching input window into a
 *     lane-wise block accumulator, reduce across lanes, and fold `* d`
 *     exactly once per block.
 *
 * Numerical equivalence with [ScalarQ4_0MatmulKernel] is within FMA +
 * reordered-reduction tolerance — the same bar the Q8_0 / Q4_K Panama
 * kernels use.
 */
public object PanamaVectorQ4_0MatmulKernel : Q4_0MatmulKernel {

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 18

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "PanamaVectorQ4_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) {
            for (o in 0 until outputDim) output[outputOffset + o] = 0f
            return
        }
        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(BLOCK_SIZE)
        val codeBuf = FloatArray(BLOCK_SIZE)

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                // FP16 scale — two LE bytes.
                val dBits = (weight[blockBase].toInt() and 0xFF) or
                    ((weight[blockBase + 1].toInt() and 0xFF) shl 8)
                val d = halfToFloat(dBits)

                // Split-layout unpack: low nibbles → 0..15, high → 16..31.
                val codesBase = blockBase + 2
                for (j in 0 until 16) {
                    val b = weight[codesBase + j].toInt() and 0xFF
                    codeBuf[j] = ((b and 0x0F) - 8).toFloat()
                    codeBuf[16 + j] = ((b ushr 4) - 8).toFloat()
                }

                val inputBase = inputOffset + blockIdx * BLOCK_SIZE
                var blockAccVec = FloatVector.zero(floatSpecies)
                var k = 0
                while (k < loopBound) {
                    val inV = FloatVector.fromArray(floatSpecies, input, inputBase + k)
                    val cV = FloatVector.fromArray(floatSpecies, codeBuf, k)
                    blockAccVec = inV.fma(cV, blockAccVec)
                    k += step
                }
                var blockAcc = blockAccVec.reduceLanes(VectorOperators.ADD)
                // Scalar tail (only if floatSpecies.length() doesn't divide 32 — rare).
                while (k < BLOCK_SIZE) {
                    blockAcc += input[inputBase + k] * codeBuf[k]
                    k++
                }
                acc += blockAcc * d
            }
            output[outputOffset + o] = acc
        }
    }

    /** Same FP16 → FP32 conversion as [ScalarQ4_0MatmulKernel]. */
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
