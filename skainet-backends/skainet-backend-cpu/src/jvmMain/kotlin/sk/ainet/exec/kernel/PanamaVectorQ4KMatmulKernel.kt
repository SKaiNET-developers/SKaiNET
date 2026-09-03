package sk.ainet.exec.kernel

import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule

import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.exec.tensor.ops.parallelChunks

/**
 * SIMD-vectorized Q4_K matmul on the JDK Vector API.
 *
 * Pipeline per 32-byte qs slab (which carries two adjacent sub-blocks
 * — sub-block `2j` in lo nibbles, sub-block `2j+1` in hi nibbles):
 *   1. `ByteVector.fromArray(byteSpeciesForFloat, weight, qsRegion+idx)` — single load.
 *   2. `loNibVec = byteVec.and(0x0F.toByte())`,
 *      `hiNibVec = byteVec.lanewise(LSHR, 4)` — extract both nibbles.
 *   3. `castShape(floatSpecies, 0)` — widen + I2F.
 *   4. `inputVec.fma(codeFloatVec, codeAcc)` — accumulate `Σ(input·code)`
 *      per sub-block; track `inputAcc = Σ(input)` separately for the
 *      lazy-`dmin` correction.
 *   5. After all super-blocks for a given output cell, sum across
 *      sub-blocks: `acc += scale[s] · codeSum[s] − offset[s] · inputSum[s]`
 *      with `scale[s] = d · scaleIdx[s]` and `offset[s] = dMin · minIdx[s]`.
 *
 * Compared to [sk.ainet.exec.tensor.ops.JvmQuantizedVectorKernels.matmulQ4_KVec]:
 * - Replaces the scalar 32-iteration nibble unpack into a scratch
 *   `FloatArray` with a single `ByteVector` load + `castShape` per
 *   `floatSpecies.length()` elements.
 * - Folds lo + hi nibble passes into a single byte load (existing
 *   helper called the byte-load helper twice — once per nibble).
 *
 * Numerical equivalence with the existing partial-vec kernel is
 * within FMA + reordered-reduction tolerance; verified via parity
 * tests at `1e-5 · inputDim`.
 */
public object PanamaVectorQ4KMatmulKernel : Q4KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val SUB_BLOCK_SIZE = 32
    private const val SUB_BLOCKS_PER_BLOCK = 8
    private const val BYTES_PER_BLOCK = 144

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    /**
     * Byte species sized so `castShape(floatSpecies, 0)` consumes
     * exactly `floatSpecies.length()` bytes — same convention as
     * [sk.ainet.exec.tensor.ops.JvmQuantizedVectorKernels.byteSpeciesForFloat].
     */
    private val byteSpeciesForFloat: VectorSpecies<Byte> = when (floatSpecies.length()) {
        16 -> ByteVector.SPECIES_128
        else -> ByteVector.SPECIES_64 // covers 4-wide (NEON) and 8-wide (AVX2)
    }

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ): Unit = matmul(input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset, CoroutineSchedule.hardware())

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
        schedule: Schedule,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "PanamaVectorQ4KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        val blocksPerInputDim = inputDim / BLOCK_SIZE

        parallelChunks(outputDim, schedule) { startO, endO ->
            // Per-task scratch — must not be shared across worker threads.
            val scaleIdx = IntArray(SUB_BLOCKS_PER_BLOCK)
            val minIdx = IntArray(SUB_BLOCKS_PER_BLOCK)
            for (o in startO until endO) {
                var acc = 0f
                for (blockIdx in 0 until blocksPerInputDim) {
                    val blockBase = weightByteOffset + (blockIdx * outputDim + o) * BYTES_PER_BLOCK

                    // d, dMin (FP16 LE).
                    val dBits = (weight[blockBase + 1].toInt() and 0xFF shl 8) or
                        (weight[blockBase].toInt() and 0xFF)
                    val dMinBits = (weight[blockBase + 3].toInt() and 0xFF shl 8) or
                        (weight[blockBase + 2].toInt() and 0xFF)
                    val d = halfToFloat(dBits)
                    val dMin = halfToFloat(dMinBits)

                    // Sub-scale decode via ggml `get_scale_min_k4`.
                    val scalesOffset = blockBase + 4
                    for (sb in 0 until 4) {
                        scaleIdx[sb] = weight[scalesOffset + sb].toInt() and 0x3F
                        minIdx[sb] = weight[scalesOffset + sb + 4].toInt() and 0x3F
                    }
                    for (sb in 4 until 8) {
                        val low4S = weight[scalesOffset + sb + 4].toInt() and 0x0F
                        val high2S = (weight[scalesOffset + sb - 4].toInt() and 0xFF) ushr 6
                        scaleIdx[sb] = low4S or (high2S shl 4)
                        val low4M = (weight[scalesOffset + sb + 4].toInt() and 0xFF) ushr 4
                        val high2M = (weight[scalesOffset + sb].toInt() and 0xFF) ushr 6
                        minIdx[sb] = low4M or (high2M shl 4)
                    }

                    // 4 strided qs groups; each carries sbLo (lo nibbles) and sbHi (hi nibbles).
                    val codesOffset = blockBase + 16
                    val inputBlockBase = inputOffset + blockIdx * BLOCK_SIZE
                    for (groupJ in 0 until 4) {
                        val qsRegion = codesOffset + groupJ * 32
                        val sbLo = 2 * groupJ
                        val sbHi = sbLo + 1
                        val inputStartLo = inputBlockBase + sbLo * SUB_BLOCK_SIZE
                        val inputStartHi = inputStartLo + SUB_BLOCK_SIZE

                        var codeAccLo = FloatVector.zero(floatSpecies)
                        var inputAccLo = FloatVector.zero(floatSpecies)
                        var codeAccHi = FloatVector.zero(floatSpecies)
                        var inputAccHi = FloatVector.zero(floatSpecies)

                        val floatStep = floatSpecies.length()
                        val byteLoadLen = byteSpeciesForFloat.length()
                        var idx = 0

                        // SIMD body — single byte load feeds both nibble vectors.
                        while (idx + floatStep <= SUB_BLOCK_SIZE &&
                            qsRegion + idx + byteLoadLen <= weight.size
                        ) {
                            val inVecLo = FloatVector.fromArray(floatSpecies, input, inputStartLo + idx)
                            val inVecHi = FloatVector.fromArray(floatSpecies, input, inputStartHi + idx)
                            val byteVec = ByteVector.fromArray(byteSpeciesForFloat, weight, qsRegion + idx)
                            val loBytes = byteVec.and(0x0F.toByte())
                            val hiBytes = byteVec.lanewise(VectorOperators.LSHR, 4.toByte())
                            val codeVecLo = loBytes.castShape(floatSpecies, 0) as FloatVector
                            val codeVecHi = hiBytes.castShape(floatSpecies, 0) as FloatVector
                            codeAccLo = inVecLo.fma(codeVecLo, codeAccLo)
                            inputAccLo = inVecLo.add(inputAccLo)
                            codeAccHi = inVecHi.fma(codeVecHi, codeAccHi)
                            inputAccHi = inVecHi.add(inputAccHi)
                            idx += floatStep
                        }

                        var codeSumLo = codeAccLo.reduceLanes(VectorOperators.ADD)
                        var inputSumLo = inputAccLo.reduceLanes(VectorOperators.ADD)
                        var codeSumHi = codeAccHi.reduceLanes(VectorOperators.ADD)
                        var inputSumHi = inputAccHi.reduceLanes(VectorOperators.ADD)

                        // Scalar tail — only fires if floatSpecies.length() doesn't divide 32 (rare).
                        while (idx < SUB_BLOCK_SIZE) {
                            val byte = weight[qsRegion + idx].toInt() and 0xFF
                            val codeLo = (byte and 0x0F).toFloat()
                            val codeHi = (byte ushr 4).toFloat()
                            val vLo = input[inputStartLo + idx]
                            val vHi = input[inputStartHi + idx]
                            codeSumLo += vLo * codeLo
                            inputSumLo += vLo
                            codeSumHi += vHi * codeHi
                            inputSumHi += vHi
                            idx++
                        }

                        val scaleLo = d * scaleIdx[sbLo]
                        val offsetLo = dMin * minIdx[sbLo]
                        val scaleHi = d * scaleIdx[sbHi]
                        val offsetHi = dMin * minIdx[sbHi]
                        acc += codeSumLo * scaleLo - inputSumLo * offsetLo
                        acc += codeSumHi * scaleHi - inputSumHi * offsetHi
                    }
                }
                output[outputOffset + o] = acc
            }
        }
    }

    /**
     * IEEE 754 binary16 → binary32 conversion. Mirrors the helper used
     * inside `JvmQuantizedVectorKernels` and `Q4_KTensorData` — kept
     * private to this file rather than depending on either, since both
     * are `internal` in their respective modules.
     */
    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits ushr 15) and 0x1
        val exp = (hbits ushr 10) and 0x1F
        val frac = hbits and 0x3FF
        return when {
            exp == 0 -> {
                if (frac == 0) {
                    if (sign == 0) 0.0f else -0.0f
                } else {
                    val f = frac / 1024.0f * (1.0f / 16384.0f)
                    if (sign == 0) f else -f
                }
            }
            exp == 0x1F -> {
                if (frac == 0) {
                    if (sign == 0) Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY
                } else {
                    Float.NaN
                }
            }
            else -> {
                val bits = (sign shl 31) or ((exp - 15 + 127) shl 23) or (frac shl 13)
                Float.fromBits(bits)
            }
        }
    }
}
