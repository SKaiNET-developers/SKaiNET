package sk.ainet.exec.kernel

import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel

/**
 * SIMD-vectorized FP32 × Q8_0 matmul on the JDK Vector API.
 *
 * Pipeline per 32-element block:
 *  1. Decode the 2-byte FP16 scale `d` once.
 *  2. Walk the 32 signed int8 codes in `floatSpecies.length()`-sized
 *     chunks. Each chunk: one ByteVector load, one `castShape` to
 *     FloatVector (signed widening — int8 codes become small floats
 *     in [-128, 127]), one `FloatVector.fma(input, codes, blockAcc)`
 *     into a lane-wise block accumulator.
 *  3. Reduce the block accumulator across lanes (`reduceLanes(ADD)`)
 *     and fold `* d` exactly once before adding to the running output
 *     cell. Folding scale per-block (rather than per-element) avoids
 *     32 extra multiplies per block; the broadcast-and-FMA-with-scale
 *     pattern would be wasteful here.
 *
 * Numerical equivalence with [ScalarQ8_0MatmulKernel] is within FMA +
 * reordered-reduction tolerance — the same bar Q4_K Panama uses.
 */
public object PanamaVectorQ8_0MatmulKernel : Q8_0MatmulKernel {

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 34

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    /** Byte species sized so `castShape(floatSpecies, 0)` consumes
     *  `floatSpecies.length()` bytes — same convention as Q4_K. */
    private val byteSpeciesForFloat: VectorSpecies<Byte> = when (floatSpecies.length()) {
        16 -> ByteVector.SPECIES_128
        else -> ByteVector.SPECIES_64 // covers 4-wide (NEON) and 8-wide (AVX2)
    }

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "PanamaVectorQ8_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) {
            for (o in 0 until outputDim) output[outputOffset + o] = 0f
            return
        }
        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val laneCount = floatSpecies.length()

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerInputDim) {
                val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK
                // FP16 scale — two LE bytes.
                val dBits = (weight[blockBase].toInt() and 0xFF) or
                    ((weight[blockBase + 1].toInt() and 0xFF) shl 8)
                val d = halfToFloat(dBits)

                val codesBase = blockBase + 2
                val inputBase = inputOffset + blockIdx * BLOCK_SIZE

                var blockAccVec = FloatVector.zero(floatSpecies)
                var k = 0
                if (laneCount == 4) {
                    // NEON (4-wide float species): each ByteVector load brings
                    // 8 bytes (SPECIES_64 is the smallest byte species). To
                    // consume all 8 — and to avoid reading 4 bytes past the
                    // codes region on the last iteration of a block — convert
                    // both halves via `castShape(species, part)` per load and
                    // step k by 8.
                    while (k < BLOCK_SIZE) {
                        val byteVec = ByteVector.fromArray(byteSpeciesForFloat, weight, codesBase + k)
                        @Suppress("UNCHECKED_CAST")
                        val codesLo = byteVec.castShape(floatSpecies, 0) as FloatVector
                        @Suppress("UNCHECKED_CAST")
                        val codesHi = byteVec.castShape(floatSpecies, 1) as FloatVector
                        val inLo = FloatVector.fromArray(floatSpecies, input, inputBase + k)
                        val inHi = FloatVector.fromArray(floatSpecies, input, inputBase + k + 4)
                        blockAccVec = inLo.fma(codesLo, blockAccVec)
                        blockAccVec = inHi.fma(codesHi, blockAccVec)
                        k += 8
                    }
                } else {
                    // AVX2 (8-wide): the SPECIES_64 load and the
                    // floatSpecies cast width match — one FMA per
                    // iteration, step k by `laneCount`.
                    while (k < BLOCK_SIZE) {
                        val byteVec = ByteVector.fromArray(byteSpeciesForFloat, weight, codesBase + k)
                        @Suppress("UNCHECKED_CAST")
                        val codesVec = byteVec.castShape(floatSpecies, 0) as FloatVector
                        val inputVec = FloatVector.fromArray(floatSpecies, input, inputBase + k)
                        blockAccVec = inputVec.fma(codesVec, blockAccVec)
                        k += laneCount
                    }
                }
                acc += blockAccVec.reduceLanes(VectorOperators.ADD) * d
            }
            output[outputOffset + o] = acc
        }
    }

    /** Same FP16 → FP32 conversion as [ScalarQ8_0MatmulKernel.halfToFloat]. */
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
