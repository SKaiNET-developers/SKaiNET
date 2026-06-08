package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Q5_0MatmulKernel

/**
 * SIMD-vectorized FP32 × Q5_0 matmul on the JDK Vector API (scratch-dequant then FMA).
 * Dequant `d*(code + (highBit shl 4) - 16)` (symmetric, no per-block min). Numerically
 * equivalent to [ScalarQ5_0MatmulKernel]. Block-major layout `(blockIdx*outputDim+o)*22`.
 */
public object PanamaVectorQ5_0MatmulKernel : Q5_0MatmulKernel {

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 22
    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "PanamaVectorQ5_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) { for (o in 0 until outputDim) output[outputOffset + o] = 0f; return }
        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val step = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(BLOCK_SIZE)
        val codeBuf = FloatArray(BLOCK_SIZE)

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val base = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                val d = halfToFloat(((weight[base + 1].toInt() and 0xFF) shl 8) or (weight[base].toInt() and 0xFF))
                val qh0 = weight[base + 2].toInt() and 0xFF
                val qh1 = weight[base + 3].toInt() and 0xFF
                val qh2 = weight[base + 4].toInt() and 0xFF
                val qh3 = weight[base + 5].toInt() and 0xFF
                val qsBase = base + 6
                for (j in 0 until 16) {
                    val q = weight[qsBase + j].toInt() and 0xFF
                    val bitLo = ((if (j < 8) qh0 else qh1) ushr (j and 7)) and 1
                    val bitHi = ((if (j < 8) qh2 else qh3) ushr (j and 7)) and 1
                    codeBuf[j] = d * ((q and 0x0F) + (bitLo shl 4) - 16)
                    codeBuf[16 + j] = d * ((q ushr 4) + (bitHi shl 4) - 16)
                }
                val inputBase = inputOffset + blockIdx * BLOCK_SIZE
                var accVec = FloatVector.zero(floatSpecies)
                var k = 0
                while (k < loopBound) {
                    accVec = FloatVector.fromArray(floatSpecies, input, inputBase + k)
                        .fma(FloatVector.fromArray(floatSpecies, codeBuf, k), accVec)
                    k += step
                }
                acc += accVec.reduceLanes(VectorOperators.ADD)
                while (k < BLOCK_SIZE) { acc += input[inputBase + k] * codeBuf[k]; k++ }
            }
            output[outputOffset + o] = acc
        }
    }

    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits and 0x8000) shl 16
        val exp = (hbits and 0x7C00) shr 10
        val mant = hbits and 0x03FF
        return when (exp) {
            0 -> if (mant == 0) Float.fromBits(sign) else {
                var m = mant; var e = -14
                while ((m and 0x400) == 0) { m = m shl 1; e-- }
                Float.fromBits(sign or ((e + 127) shl 23) or ((m and 0x3FF) shl 13))
            }
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
