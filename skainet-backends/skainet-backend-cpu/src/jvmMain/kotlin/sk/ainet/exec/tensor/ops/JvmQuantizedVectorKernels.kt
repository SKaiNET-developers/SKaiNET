package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder

/**
 * JVM Vector API kernels for quantized tensor operations.
 *
 * These kernels perform fused dequantize-dot-product operations for Q8_0 and Q4_K
 * formats, avoiding full materialization of FP32 weights in memory.
 */
internal object JvmQuantizedVectorKernels {
    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
    private val byteSpecies: VectorSpecies<Byte> = ByteVector.SPECIES_PREFERRED

    /**
     * Compute dot product of input vector with Q8_0 quantized codes for one block.
     *
     * Q8_0 block: 32 int8 codes with shared f16 scale.
     * Result = sum(input[i] * code[i]) * scale
     *
     * @param input Input float array
     * @param inputOffset Starting offset in input array
     * @param codes Quantized int8 codes (32 values)
     * @param codesOffset Starting offset in codes array
     * @param scale Block scale factor
     * @return Dot product result
     */
    fun dotQ8_0Block(
        input: FloatArray,
        inputOffset: Int,
        codes: ByteArray,
        codesOffset: Int,
        scale: Float
    ): Float {
        val blockSize = 32
        val floatStep = floatSpecies.length()
        val byteLoadLen = byteSpeciesForFloat.length()
        var accVec = FloatVector.zero(floatSpecies)
        var accScalar = 0f
        var idx = 0

        while (idx + floatStep <= blockSize && codesOffset + idx + byteLoadLen <= codes.size) {
            val inputVec = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)
            val byteVec = ByteVector.fromArray(byteSpeciesForFloat, codes, codesOffset + idx)
            val codeVec = byteVec.castShape(floatSpecies, 0) as FloatVector
            accVec = inputVec.mul(codeVec).add(accVec)
            idx += floatStep
        }

        // Scalar tail
        while (idx < blockSize) {
            accScalar += input[inputOffset + idx] * codes[codesOffset + idx].toFloat()
            idx++
        }

        return (accVec.reduceLanes(VectorOperators.ADD) + accScalar) * scale
    }

    /**
     * Compute the (codeSum, inputSum) pair for one Q4_K sub-block (32
     * elements) using the *strided* canonical ggml layout: for a 32-byte qs
     * region shared by two sub-blocks, the lo nibbles of bytes 0..31 form
     * sub-block A and the hi nibbles of the same bytes form sub-block B.
     *
     *   codeSum  = sum_i input[i] * code[i]
     *   inputSum = sum_i input[i]
     *
     * The caller turns these into a per-sub-block contribution via
     * `codeSum * scale - inputSum * offset`, where `scale = d * scaleIdx`
     * and `offset = dMin * minIdx`.
     *
     * @param input       activation array
     * @param inputOffset starting offset in input
     * @param qs          packed Q4_K codes (32 bytes consumed at qsOffset)
     * @param qsOffset    starting offset in qs of the 32-byte group
     * @param hiNibble    true to take `byte >>> 4` (sub-block B), false for `byte & 0x0F` (sub-block A)
     * @param codeBuf     scratch FloatArray of length >= SUB_BLOCK_SIZE
     * @param sumsOut     2-element scratch: out[0] = codeSum, out[1] = inputSum
     */
    fun dotQ4_KHalfNibbleSubBlock(
        input: FloatArray,
        inputOffset: Int,
        qs: ByteArray,
        qsOffset: Int,
        hiNibble: Boolean,
        codeBuf: FloatArray,
        sumsOut: FloatArray,
    ) {
        if (hiNibble) {
            for (i in 0 until SUB_BLOCK_SIZE) {
                codeBuf[i] = ((qs[qsOffset + i].toInt() and 0xFF) ushr 4).toFloat()
            }
        } else {
            for (i in 0 until SUB_BLOCK_SIZE) {
                codeBuf[i] = (qs[qsOffset + i].toInt() and 0x0F).toFloat()
            }
        }

        val step = floatSpecies.length()
        var codeAcc = FloatVector.zero(floatSpecies)
        var inputAcc = FloatVector.zero(floatSpecies)
        var idx = 0
        val loopBound = floatSpecies.loopBound(SUB_BLOCK_SIZE)
        while (idx < loopBound) {
            val iv = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)
            val cv = FloatVector.fromArray(floatSpecies, codeBuf, idx)
            codeAcc = iv.fma(cv, codeAcc)
            inputAcc = iv.add(inputAcc)
            idx += step
        }
        var codeSum = codeAcc.reduceLanes(VectorOperators.ADD)
        var inputSum = inputAcc.reduceLanes(VectorOperators.ADD)

        // Scalar tail (only fires if SPECIES_PREFERRED.length() > SUB_BLOCK_SIZE).
        while (idx < SUB_BLOCK_SIZE) {
            val v = input[inputOffset + idx]
            codeSum += v * codeBuf[idx]
            inputSum += v
            idx++
        }

        sumsOut[0] = codeSum
        sumsOut[1] = inputSum
    }

    private const val SUB_BLOCK_SIZE = 32

    /**
     * Vectorized Q8_0 matrix-vector multiplication.
     *
     * @param input Input vector [inputDim]
     * @param packedWeights Packed Q8_0 weight data
     * @param inputDim Input dimension (must be multiple of 32)
     * @param outputDim Output dimension
     * @param output Output array to write results
     * @param outputOffset Starting offset in output array
     */
    fun matmulQ8_0Vec(
        input: FloatArray,
        packedWeights: ByteArray,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0
    ) {
        val blockSize = 32
        val bytesPerBlock = 34  // 2 scale + 32 codes
        val blocksPerInputDim = (inputDim + blockSize - 1) / blockSize

        for (o in 0 until outputDim) {
            var acc = 0f

            for (blockIdx in 0 until blocksPerInputDim) {
                val weightBlockOffset = (blockIdx * outputDim + o) * bytesPerBlock

                // Read f16 scale
                val b0 = packedWeights[weightBlockOffset].toInt() and 0xFF
                val b1 = packedWeights[weightBlockOffset + 1].toInt() and 0xFF
                val scale = halfToFloat((b1 shl 8) or b0)

                // Compute block dot product
                val inputStart = blockIdx * blockSize
                val codesStart = weightBlockOffset + 2

                acc += dotQ8_0Block(input, inputStart, packedWeights, codesStart, scale)
            }

            output[outputOffset + o] = acc
        }
    }

    /**
     * Vectorized Q4_K matrix-vector multiplication.
     *
     * @param input Input vector [inputDim]
     * @param packedWeights Packed Q4_K weight data
     * @param inputDim Input dimension (must be multiple of 256)
     * @param outputDim Output dimension
     * @param output Output array to write results
     * @param outputOffset Starting offset in output array
     */
    fun matmulQ4_KVec(
        input: FloatArray,
        packedWeights: ByteArray,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0
    ) {
        val blockSize = 256
        val subBlockSize = 32
        val bytesPerBlock = 144  // 2 d + 2 dMin + 12 scales + 128 codes
        val blocksPerInputDim = (inputDim + blockSize - 1) / blockSize

        parallelChunks(outputDim) { startO, endO ->
            // Each task owns its own scratch arrays to avoid cross-thread contention.
            val codeBuf = FloatArray(subBlockSize)
            val scaleIdxBuf = IntArray(8)
            val minIdxBuf = IntArray(8)
            val sumsBuf = FloatArray(2)
            for (o in startO until endO) {
                var acc = 0f

                for (blockIdx in 0 until blocksPerInputDim) {
                    val weightBlockOffset = (blockIdx * outputDim + o) * bytesPerBlock

                    // Read f16 d and dMin (super-block scale and min-scale)
                    val dBits = (packedWeights[weightBlockOffset + 1].toInt() and 0xFF shl 8) or
                        (packedWeights[weightBlockOffset].toInt() and 0xFF)
                    val dMinBits = (packedWeights[weightBlockOffset + 3].toInt() and 0xFF shl 8) or
                        (packedWeights[weightBlockOffset + 2].toInt() and 0xFF)
                    val d = halfToFloat(dBits)
                    val dMin = halfToFloat(dMinBits)

                    // Decode 8 sub-block (scaleIdx, minIdx) pairs from the 12 scale
                    // bytes via ggml's `get_scale_min_k4` (sub-blocks 4..7 reuse
                    // top 2 bits of bytes for sub-blocks 0..3 — *not* a flat
                    // 12-bits-per-sub-block packing).
                    val scalesOffset = weightBlockOffset + 4
                    for (sb in 0 until 4) {
                        scaleIdxBuf[sb] = packedWeights[scalesOffset + sb].toInt() and 0x3F
                        minIdxBuf[sb] = packedWeights[scalesOffset + sb + 4].toInt() and 0x3F
                    }
                    for (sb in 4 until 8) {
                        val low4S = packedWeights[scalesOffset + sb + 4].toInt() and 0x0F
                        val high2S = (packedWeights[scalesOffset + sb - 4].toInt() and 0xFF) ushr 6
                        scaleIdxBuf[sb] = low4S or (high2S shl 4)
                        val low4M = (packedWeights[scalesOffset + sb + 4].toInt() and 0xFF) ushr 4
                        val high2M = (packedWeights[scalesOffset + sb].toInt() and 0xFF) ushr 6
                        minIdxBuf[sb] = low4M or (high2M shl 4)
                    }

                    // Walk the 4 strided qs groups (32 bytes each). Group `groupJ`
                    // holds sub-block (2*groupJ) in lo nibbles and sub-block
                    // (2*groupJ + 1) in hi nibbles of the *same* 32 bytes.
                    val codesOffset = weightBlockOffset + 16
                    for (groupJ in 0 until 4) {
                        val qsRegion = codesOffset + groupJ * 32

                        val sbLo = 2 * groupJ
                        val inputStartLo = blockIdx * blockSize + sbLo * subBlockSize
                        if (inputStartLo < inputDim) {
                            dotQ4_KHalfNibbleSubBlock(
                                input, inputStartLo, packedWeights, qsRegion,
                                hiNibble = false, codeBuf, sumsBuf
                            )
                            val scale = d * scaleIdxBuf[sbLo]
                            val offset = dMin * minIdxBuf[sbLo]
                            acc += sumsBuf[0] * scale - sumsBuf[1] * offset
                        }

                        val sbHi = 2 * groupJ + 1
                        val inputStartHi = inputStartLo + subBlockSize
                        if (inputStartHi < inputDim) {
                            dotQ4_KHalfNibbleSubBlock(
                                input, inputStartHi, packedWeights, qsRegion,
                                hiNibble = true, codeBuf, sumsBuf
                            )
                            val scale = d * scaleIdxBuf[sbHi]
                            val offset = dMin * minIdxBuf[sbHi]
                            acc += sumsBuf[0] * scale - sumsBuf[1] * offset
                        }
                    }
                }

                output[outputOffset + o] = acc
            }
        }
    }

    /**
     * Vectorized Q6_K matrix-vector multiplication.
     *
     * Block format (210 bytes per 256-element block):
     *   - bytes [  0..127]: ql — low 4 bits of each 6-bit code (half-interleaved packing)
     *   - bytes [128..191]: qh — high 2 bits of each 6-bit code (half-interleaved packing)
     *   - bytes [192..207]: scales — one signed int8 per 16-element sub-block
     *   - bytes [208..209]: f16 d — block scale
     *
     * Packed-weights layout is input-block-major: the block for output row
     * `o` and input-block index `bI` starts at byte offset
     * `(bI * outputDim + o) * 210`. Callers (GemmaMemSegConverter for the
     * Q6_K relayout; the numeric-parity tests) must honour this.
     *
     * Implementation strategy: dequant each 256-element block into a scratch
     * FloatArray and do a SIMD dot-product with the matching slice of
     * `input`. Same block size as Q4_K, different packing; the reference for
     * the per-element dequant math is
     * [DequantOps.dequantQ6KFromBytes] in skainet-io-gguf.
     *
     * @param input FP32 activation vector [inputDim]
     * @param packedWeights Q6_K packed bytes, laid out `(blockIdx * outputDim + o) * 210`
     * @param inputDim input dimension (must be a multiple of 256)
     * @param outputDim output dimension
     * @param output destination array
     * @param outputOffset starting offset in [output]
     */
    fun matmulQ6_KVec(
        input: FloatArray,
        packedWeights: ByteArray,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0
    ) {
        val blockSize = 256
        val bytesPerBlock = 210
        val blocksPerInputDim = (inputDim + blockSize - 1) / blockSize
        val floatStep = floatSpecies.length()
        val loopBound = floatSpecies.loopBound(blockSize)

        parallelChunks(outputDim) { startO, endO ->
            // Per-task scratch — must not be shared across worker threads.
            val scratch = FloatArray(blockSize)
            for (o in startO until endO) {
                var accVec = FloatVector.zero(floatSpecies)
                var accScalar = 0f
                for (blockIdx in 0 until blocksPerInputDim) {
                    val weightBlockOffset = (blockIdx * outputDim + o) * bytesPerBlock
                    dequantQ6_KBlock(packedWeights, weightBlockOffset, scratch, 0)

                    val inputStart = blockIdx * blockSize
                    if (inputStart >= inputDim) continue

                    val elemsInBlock = minOf(blockSize, inputDim - inputStart)

                    if (elemsInBlock >= floatStep) {
                        val bound = if (elemsInBlock == blockSize) loopBound
                                    else floatSpecies.loopBound(elemsInBlock)
                        var idx = 0
                        while (idx < bound) {
                            val inputVec = FloatVector.fromArray(floatSpecies, input, inputStart + idx)
                            val codeVec = FloatVector.fromArray(floatSpecies, scratch, idx)
                            accVec = inputVec.fma(codeVec, accVec)
                            idx += floatStep
                        }
                        while (idx < elemsInBlock) {
                            accScalar += input[inputStart + idx] * scratch[idx]
                            idx++
                        }
                    } else {
                        for (idx in 0 until elemsInBlock) {
                            accScalar += input[inputStart + idx] * scratch[idx]
                        }
                    }
                }
                output[outputOffset + o] = accVec.reduceLanes(VectorOperators.ADD) + accScalar
            }
        }
    }

    /**
     * Dequantize one 256-element Q6_K block into [scratch] starting at
     * [scratchOffset]. Mirrors
     * `DequantOps.dequantQ6KFromBytes` line-for-line — see that method for
     * the authoritative spec.
     *
     * SIMD-fused via `ByteVector`: per `floatStep`-wide chunk of `l`,
     * loads one slice of `ql[qlBase+l]`, one of `ql[qlBase+l+32]`, and
     * one of `qh[qhBase+l]`, then assembles q1..q4 = `(qlNibble) |
     * ((qhSlice) << 4) − 32` per code lane via byte AND/LSHR/OR ops,
     * widens to FloatVector, multiplies by per-sub-block `d·scale`, and
     * stores to four 32-element output regions in `scratch`. Replaces
     * the prior scalar loop's 32 iterations × 4 codes/iteration of
     * scalar shifts and multiplies with one ByteVector load + 4 FMA
     * stores per chunk. Scalar tail fires only when `floatStep` doesn't
     * divide 16 (rare).
     */
    private fun dequantQ6_KBlock(
        packedWeights: ByteArray,
        blockByteOffset: Int,
        scratch: FloatArray,
        scratchOffset: Int
    ) {
        val qlBase0 = blockByteOffset
        val qhBase0 = blockByteOffset + 128
        val scBase0 = blockByteOffset + 192
        val dOffset = blockByteOffset + 208

        val dBits = (packedWeights[dOffset + 1].toInt() and 0xFF shl 8) or
            (packedWeights[dOffset].toInt() and 0xFF)
        val d = halfToFloat(dBits)

        val floatStep = floatSpecies.length()
        val byteLoadLen = byteSpeciesForFloat.length()

        for (half in 0..1) {
            val qlBase = qlBase0 + half * 64
            val qhBase = qhBase0 + half * 32
            val scBase = scBase0 + half * 8
            val outBase = scratchOffset + half * 128

            for (isIdx in 0..1) {
                val sc1 = d * packedWeights[scBase + isIdx + 0].toInt()
                val sc2 = d * packedWeights[scBase + isIdx + 2].toInt()
                val sc3 = d * packedWeights[scBase + isIdx + 4].toInt()
                val sc4 = d * packedWeights[scBase + isIdx + 6].toInt()
                val sc1Vec = FloatVector.broadcast(floatSpecies, sc1)
                val sc2Vec = FloatVector.broadcast(floatSpecies, sc2)
                val sc3Vec = FloatVector.broadcast(floatSpecies, sc3)
                val sc4Vec = FloatVector.broadcast(floatSpecies, sc4)
                val negThirtyTwo = FloatVector.broadcast(floatSpecies, -32f)

                val lStart = isIdx * 16
                val lEnd = lStart + 16
                var l = lStart
                while (l + floatStep <= lEnd &&
                    qlBase + l + byteLoadLen <= packedWeights.size
                ) {
                    val ql0Vec = ByteVector.fromArray(byteSpeciesForFloat, packedWeights, qlBase + l)
                    val ql32Vec = ByteVector.fromArray(byteSpeciesForFloat, packedWeights, qlBase + l + 32)
                    val qhVec = ByteVector.fromArray(byteSpeciesForFloat, packedWeights, qhBase + l)

                    val ql0Lo = ql0Vec.and(0x0F.toByte())
                    val ql0Hi = ql0Vec.lanewise(VectorOperators.LSHR, 4.toByte())
                    val ql32Lo = ql32Vec.and(0x0F.toByte())
                    val ql32Hi = ql32Vec.lanewise(VectorOperators.LSHR, 4.toByte())

                    val qh1 = qhVec.and(0x03.toByte())
                    val qh2 = qhVec.lanewise(VectorOperators.LSHR, 2.toByte()).and(0x03.toByte())
                    val qh3 = qhVec.lanewise(VectorOperators.LSHR, 4.toByte()).and(0x03.toByte())
                    val qh4 = qhVec.lanewise(VectorOperators.LSHR, 6.toByte())

                    val q1Bytes = ql0Lo.or(qh1.lanewise(VectorOperators.LSHL, 4.toByte()))
                    val q2Bytes = ql32Lo.or(qh2.lanewise(VectorOperators.LSHL, 4.toByte()))
                    val q3Bytes = ql0Hi.or(qh3.lanewise(VectorOperators.LSHL, 4.toByte()))
                    val q4Bytes = ql32Hi.or(qh4.lanewise(VectorOperators.LSHL, 4.toByte()))

                    val q1F = (q1Bytes.castShape(floatSpecies, 0) as FloatVector).add(negThirtyTwo)
                    val q2F = (q2Bytes.castShape(floatSpecies, 0) as FloatVector).add(negThirtyTwo)
                    val q3F = (q3Bytes.castShape(floatSpecies, 0) as FloatVector).add(negThirtyTwo)
                    val q4F = (q4Bytes.castShape(floatSpecies, 0) as FloatVector).add(negThirtyTwo)

                    q1F.mul(sc1Vec).intoArray(scratch, outBase + l + 0)
                    q2F.mul(sc2Vec).intoArray(scratch, outBase + l + 32)
                    q3F.mul(sc3Vec).intoArray(scratch, outBase + l + 64)
                    q4F.mul(sc4Vec).intoArray(scratch, outBase + l + 96)

                    l += floatStep
                }

                // Scalar tail (only fires if floatStep doesn't divide 16).
                while (l < lEnd) {
                    val ql0 = packedWeights[qlBase + l].toInt() and 0xFF
                    val ql32 = packedWeights[qlBase + l + 32].toInt() and 0xFF
                    val qhL = packedWeights[qhBase + l].toInt() and 0xFF
                    val q1 = ((ql0 and 0x0F) or ((qhL and 0x03) shl 4)) - 32
                    val q2 = ((ql32 and 0x0F) or (((qhL ushr 2) and 0x03) shl 4)) - 32
                    val q3 = ((ql0 ushr 4) or (((qhL ushr 4) and 0x03) shl 4)) - 32
                    val q4 = ((ql32 ushr 4) or (((qhL ushr 6) and 0x03) shl 4)) - 32
                    scratch[outBase + l + 0] = sc1 * q1
                    scratch[outBase + l + 32] = sc2 * q2
                    scratch[outBase + l + 64] = sc3 * q3
                    scratch[outBase + l + 96] = sc4 * q4
                    l++
                }
            }
        }
    }

    /**
     * Convert f16 bits to float32.
     */
    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits and 0x8000) shl 16
        val exp = (hbits and 0x7C00) shr 10
        val mant = hbits and 0x03FF

        return when (exp) {
            0 -> {
                if (mant == 0) {
                    Float.fromBits(sign)
                } else {
                    var m = mant
                    var e = -14
                    while ((m and 0x400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x3FF
                    val floatExp = (e + 127) shl 23
                    val floatMant = m shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
            }
            31 -> {
                val floatExp = 0xFF shl 23
                val floatMant = mant shl 13
                Float.fromBits(sign or floatExp or floatMant)
            }
            else -> {
                val floatExp = (exp - 15 + 127) shl 23
                val floatMant = mant shl 13
                Float.fromBits(sign or floatExp or floatMant)
            }
        }
    }

    private fun ByteArray.getOrElse(index: Int, default: () -> Byte): Byte {
        return if (index in indices) this[index] else default()
    }

    // -----------------------------------------------------------------------
    // MemorySegment-based Q4 kernels (F32 activations x Q4 packed weights)
    // -----------------------------------------------------------------------

    private val JAVA_BYTE_LE: ValueLayout.OfByte = ValueLayout.JAVA_BYTE
    private val JAVA_FLOAT_LE: ValueLayout.OfFloat =
        ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN)

    /**
     * Q4_0 dot-product for a single block of 32 elements stored in a MemorySegment.
     *
     * Q4_0 block layout: 2 bytes f16 scale + 16 bytes packed nibbles (32 values).
     * Each byte packs two 4-bit codes — adjacent elements share a byte:
     * `code[2k] = byte[k] & 0x0F`, `code[2k+1] = byte[k] >>> 4`. Subtract
     * 8 for sign correction.
     *
     * Two-stage SIMD: a scalar byte-pair unpack writes 32 sign-corrected
     * floats into [codeBuf] (16 byte loads from the MemorySegment, two
     * nibbles per load — same memory traffic as the prior fully-scalar
     * implementation), then a vector FMA loop dot-products [codeBuf]
     * with the matching input slice. The nibble-pair-per-byte layout
     * makes a fully-fused `ByteVector` pipeline (a la
     * [PanamaVectorQ4KMatmulKernel]) awkward without strided gather or
     * lane-interleave shuffles, so this kernel keeps the scratch +
     * SIMD-dot shape — same approach Q4_K used before its
     * fused-pipeline rewrite (PR #562).
     *
     * @param codeBuf scratch FloatArray of length >= 32, supplied by
     *   the caller so allocation amortizes across blocks.
     */
    fun dotQ4_0BlockMemSeg(
        input: FloatArray,
        inputOffset: Int,
        weightSeg: MemorySegment,
        blockByteOffset: Long,
        codeBuf: FloatArray,
    ): Float {
        val blockSize = 32
        val codesOffset = blockByteOffset + 2

        // Read f16 scale
        val scale = halfToFloat(read2BytesLE(weightSeg, blockByteOffset))

        // Unpack 16 packed bytes → 32 sign-corrected nibbles in the
        // canonical ggml *split* layout: low nibbles decode elements
        // 0..15, high nibbles decode elements 16..31. (Matches
        // DequantOps.dequantQ4_0FromBytes and Q4_0BlockTensorData.)
        for (k in 0 until 16) {
            val b = weightSeg.get(JAVA_BYTE_LE, codesOffset + k.toLong()).toInt() and 0xFF
            codeBuf[k] = (b and 0x0F).toFloat() - 8f
            codeBuf[16 + k] = (b ushr 4).toFloat() - 8f
        }

        // SIMD FMA dot product.
        val step = floatSpecies.length()
        var accVec = FloatVector.zero(floatSpecies)
        var idx = 0
        val loopBound = floatSpecies.loopBound(blockSize)
        while (idx < loopBound) {
            val iv = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)
            val cv = FloatVector.fromArray(floatSpecies, codeBuf, idx)
            accVec = iv.fma(cv, accVec)
            idx += step
        }
        var acc = accVec.reduceLanes(VectorOperators.ADD)
        // Scalar tail (only fires if floatSpecies.length() doesn't divide 32 — rare).
        while (idx < blockSize) {
            acc += input[inputOffset + idx] * codeBuf[idx]
            idx++
        }

        return acc * scale
    }

    /**
     * Backwards-compatible overload that allocates its own scratch
     * buffer. Existing callers that don't pass one still work; the
     * matmul-level [matmulF32Q4_0MemSeg] hoists the allocation out of
     * the per-block loop.
     */
    fun dotQ4_0BlockMemSeg(
        input: FloatArray,
        inputOffset: Int,
        weightSeg: MemorySegment,
        blockByteOffset: Long,
    ): Float = dotQ4_0BlockMemSeg(
        input, inputOffset, weightSeg, blockByteOffset,
        codeBuf = FloatArray(32),
    )

    /**
     * F32 x Q4_0 matrix-vector multiply using MemorySegment for packed Q4 weights.
     *
     * @param input FP32 activation vector [inputDim]
     * @param weightSeg MemorySegment containing packed Q4_0 weight data
     * @param weightByteOffset starting byte offset into weightSeg
     * @param inputDim input dimension (must be multiple of 32)
     * @param outputDim output dimension
     * @param output output float array
     * @param outputOffset starting index in output
     */
    fun matmulF32Q4_0MemSeg(
        input: FloatArray,
        weightSeg: MemorySegment,
        weightByteOffset: Long,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0,
    ) {
        val blockSize = 32
        val bytesPerBlock = 18L // 2 bytes scale + 16 bytes codes
        val blocksPerRow = (inputDim + blockSize - 1) / blockSize
        // Scratch hoisted out of the per-block loop — see dotQ4_0BlockMemSeg kdoc.
        val codeBuf = FloatArray(blockSize)

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerRow) {
                val blockOff = weightByteOffset +
                    (o.toLong() * blocksPerRow + blockIdx) * bytesPerBlock
                val inputStart = blockIdx * blockSize
                acc += dotQ4_0BlockMemSeg(input, inputStart, weightSeg, blockOff, codeBuf)
            }
            output[outputOffset + o] = acc
        }
    }

    /**
     * F32 x Q4_K matrix-vector multiply using MemorySegment for packed Q4_K
     * weights. Same canonical ggml layout as `matmulQ4_KVec` (strided codes,
     * `get_scale_min_k4` scale packing, `code * scale - offset` formula);
     * just reads bytes through `MemorySegment.get`.
     */
    fun matmulF32Q4_KMemSeg(
        input: FloatArray,
        weightSeg: MemorySegment,
        weightByteOffset: Long,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0,
    ) {
        val blockSize = 256
        val subBlockSize = 32
        val bytesPerBlock = 144L
        val blocksPerRow = (inputDim + blockSize - 1) / blockSize
        val scaleIdxBuf = IntArray(8)
        val minIdxBuf = IntArray(8)

        val floatStep = floatSpecies.length()
        val byteLoadLen = byteSpeciesForFloat.length()

        for (o in 0 until outputDim) {
            var acc = 0f

            for (blockIdx in 0 until blocksPerRow) {
                val blockOff = weightByteOffset +
                    (o.toLong() * blocksPerRow + blockIdx) * bytesPerBlock

                // Read f16 d and dMin
                val dBits = (weightSeg.get(JAVA_BYTE_LE, blockOff + 1).toInt() and 0xFF shl 8) or
                    (weightSeg.get(JAVA_BYTE_LE, blockOff).toInt() and 0xFF)
                val dMinBits = (weightSeg.get(JAVA_BYTE_LE, blockOff + 3).toInt() and 0xFF shl 8) or
                    (weightSeg.get(JAVA_BYTE_LE, blockOff + 2).toInt() and 0xFF)
                val d = halfToFloat(dBits)
                val dMin = halfToFloat(dMinBits)

                val scalesOff = blockOff + 4
                val codesOff = blockOff + 16

                // Decode 8 (scaleIdx, minIdx) pairs via ggml's `get_scale_min_k4`.
                for (sb in 0 until 4) {
                    scaleIdxBuf[sb] = weightSeg.get(JAVA_BYTE_LE, scalesOff + sb).toInt() and 0x3F
                    minIdxBuf[sb] = weightSeg.get(JAVA_BYTE_LE, scalesOff + sb + 4).toInt() and 0x3F
                }
                for (sb in 4 until 8) {
                    val low4S = weightSeg.get(JAVA_BYTE_LE, scalesOff + sb + 4).toInt() and 0x0F
                    val high2S = (weightSeg.get(JAVA_BYTE_LE, scalesOff + sb - 4).toInt() and 0xFF) ushr 6
                    scaleIdxBuf[sb] = low4S or (high2S shl 4)
                    val low4M = (weightSeg.get(JAVA_BYTE_LE, scalesOff + sb + 4).toInt() and 0xFF) ushr 4
                    val high2M = (weightSeg.get(JAVA_BYTE_LE, scalesOff + sb).toInt() and 0xFF) ushr 6
                    minIdxBuf[sb] = low4M or (high2M shl 4)
                }

                // 4 strided qs groups; each carries sbLo (lo nibbles) and sbHi (hi nibbles).
                // Single ByteVector load per chunk feeds both nibble accumulators —
                // mirrors the SIMD pipeline in PanamaVectorQ4KMatmulKernel for the
                // ByteArray-backed path; this kernel reads from MemorySegment via
                // ByteVector.fromMemorySegment for mmap'd weight buffers.
                for (groupJ in 0 until 4) {
                    val qsRegion = codesOff + groupJ * 32L
                    val sbLo = 2 * groupJ
                    val sbHi = sbLo + 1
                    val inputStartLo = blockIdx * blockSize + sbLo * subBlockSize
                    val inputStartHi = inputStartLo + subBlockSize

                    var codeAccLo = FloatVector.zero(floatSpecies)
                    var inputAccLo = FloatVector.zero(floatSpecies)
                    var codeAccHi = FloatVector.zero(floatSpecies)
                    var inputAccHi = FloatVector.zero(floatSpecies)
                    var idx = 0

                    while (idx + floatStep <= subBlockSize) {
                        val byteVec = ByteVector.fromMemorySegment(
                            byteSpeciesForFloat, weightSeg, qsRegion + idx, ByteOrder.LITTLE_ENDIAN,
                        )
                        val loBytes = byteVec.and(0x0F.toByte())
                        val hiBytes = byteVec.lanewise(VectorOperators.LSHR, 4.toByte())
                        val codeVecLo = loBytes.castShape(floatSpecies, 0) as FloatVector
                        val codeVecHi = hiBytes.castShape(floatSpecies, 0) as FloatVector
                        val inVecLo = FloatVector.fromArray(floatSpecies, input, inputStartLo + idx)
                        val inVecHi = FloatVector.fromArray(floatSpecies, input, inputStartHi + idx)
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

                    while (idx < subBlockSize) {
                        val b = weightSeg.get(JAVA_BYTE_LE, qsRegion + idx).toInt() and 0xFF
                        val codeLo = (b and 0x0F).toFloat()
                        val codeHi = (b ushr 4).toFloat()
                        val vLo = input[inputStartLo + idx]
                        val vHi = input[inputStartHi + idx]
                        codeSumLo += vLo * codeLo
                        inputSumLo += vLo
                        codeSumHi += vHi * codeHi
                        inputSumHi += vHi
                        idx++
                    }

                    val scaleLo = d * scaleIdxBuf[sbLo]
                    val offsetLo = dMin * minIdxBuf[sbLo]
                    val scaleHi = d * scaleIdxBuf[sbHi]
                    val offsetHi = dMin * minIdxBuf[sbHi]
                    if (inputStartLo < inputDim) {
                        acc += codeSumLo * scaleLo - inputSumLo * offsetLo
                    }
                    if (inputStartHi < inputDim) {
                        acc += codeSumHi * scaleHi - inputSumHi * offsetHi
                    }
                }
            }

            output[outputOffset + o] = acc
        }
    }

    /**
     * Byte species matching the float species lane count — used for loading
     * exactly `floatSpecies.length()` bytes from a MemorySegment so that
     * `castShape(floatSpecies, 0)` produces one full FloatVector.
     */
    private val byteSpeciesForFloat: VectorSpecies<Byte> = when (floatSpecies.length()) {
        // float lane count == byte lane count needed for castShape(floatSpecies, 0)
        // 8 floats * 4 bytes/float = 256-bit float vector; need 8 bytes = 64-bit byte vector
        8 -> ByteVector.SPECIES_64
        // 16 floats = 512-bit; need 16 bytes = 128-bit byte vector
        16 -> ByteVector.SPECIES_128
        // 4 floats = 128-bit; need 4 bytes = 32-bit byte vector (not standard; use SPECIES_64 and part 0)
        else -> ByteVector.SPECIES_64
    }

    /**
     * Read a little-endian unsigned 16-bit value from a MemorySegment.
     */
    private fun read2BytesLE(seg: MemorySegment, off: Long): Int {
        val b0 = seg.get(JAVA_BYTE_LE, off).toInt() and 0xFF
        val b1 = seg.get(JAVA_BYTE_LE, off + 1).toInt() and 0xFF
        return (b1 shl 8) or b0
    }

    /**
     * F32 x Q8_0 matrix-vector multiply using MemorySegment for packed Q8_0 weights.
     *
     * Optimized with:
     * - ByteVector.fromMemorySegment() for bulk byte loads (vs byte-by-byte)
     * - castShape() for byte→float conversion (vs temp FloatArray)
     * - Vector accumulator with single reduceLanes() at the end (vs per-iteration reduce)
     * - Pre-computed row byte offset outside inner loop
     */
    fun matmulF32Q8_0MemSeg(
        input: FloatArray,
        weightSeg: MemorySegment,
        weightByteOffset: Long,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0,
    ) {
        val blockSize = 32
        val bytesPerBlock = 34L // 2 bytes scale + 32 bytes codes
        val blocksPerRow = (inputDim + blockSize - 1) / blockSize
        val floatStep = floatSpecies.length()
        val byteLoadLen = byteSpeciesForFloat.length()
        val segLen = weightSeg.byteSize()

        for (o in 0 until outputDim) {
            var accVec = FloatVector.zero(floatSpecies)
            var accScalar = 0f
            val rowByteOff = weightByteOffset + o.toLong() * blocksPerRow * bytesPerBlock

            for (blockIdx in 0 until blocksPerRow) {
                val blockOff = rowByteOff + blockIdx * bytesPerBlock
                val scale = halfToFloat(read2BytesLE(weightSeg, blockOff))
                val scaleVec = FloatVector.broadcast(floatSpecies, scale)
                val codesOff = blockOff + 2
                val inputStart = blockIdx * blockSize

                // Vectorized: load bytes from MemorySegment, cast to float, multiply-accumulate
                var idx = 0
                while (idx + floatStep <= blockSize) {
                    val byteOff = codesOff + idx.toLong()
                    // Guard: if byte vector load would exceed segment, fall back to scalar
                    if (byteOff + byteLoadLen > segLen) break
                    val inputVec = FloatVector.fromArray(floatSpecies, input, inputStart + idx)
                    val byteVec = ByteVector.fromMemorySegment(
                        byteSpeciesForFloat, weightSeg, byteOff, ByteOrder.LITTLE_ENDIAN
                    )
                    val codeVec = byteVec.castShape(floatSpecies, 0) as FloatVector
                    accVec = inputVec.mul(codeVec).mul(scaleVec).add(accVec)
                    idx += floatStep
                }
                // Scalar tail for remaining elements in block
                while (idx < blockSize) {
                    accScalar += input[inputStart + idx] *
                        weightSeg.get(JAVA_BYTE_LE, codesOff + idx.toLong()).toFloat() * scale
                    idx++
                }
            }

            output[outputOffset + o] = accVec.reduceLanes(VectorOperators.ADD) + accScalar
        }
    }

    /**
     * F32 x Q8_0 matrix-vector multiply reading input directly from a MemorySegment.
     *
     * Same algorithm as [matmulF32Q8_0MemSeg] but uses FloatVector.fromMemorySegment()
     * for the input, avoiding a heap copy.
     */
    fun matmulF32Q8_0MemSegInput(
        inputSeg: MemorySegment,
        inputByteOffset: Long,
        weightSeg: MemorySegment,
        weightByteOffset: Long,
        inputDim: Int,
        outputDim: Int,
        output: FloatArray,
        outputOffset: Int = 0,
    ) {
        val blockSize = 32
        val bytesPerBlock = 34L
        val blocksPerRow = (inputDim + blockSize - 1) / blockSize
        val floatStep = floatSpecies.length()
        val byteLoadLen = byteSpeciesForFloat.length()
        val segLen = weightSeg.byteSize()
        val FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN)

        for (o in 0 until outputDim) {
            var accVec = FloatVector.zero(floatSpecies)
            var accScalar = 0f
            val rowByteOff = weightByteOffset + o.toLong() * blocksPerRow * bytesPerBlock

            for (blockIdx in 0 until blocksPerRow) {
                val blockOff = rowByteOff + blockIdx * bytesPerBlock
                val scale = halfToFloat(read2BytesLE(weightSeg, blockOff))
                val scaleVec = FloatVector.broadcast(floatSpecies, scale)
                val codesOff = blockOff + 2
                val inputStartBytes = inputByteOffset + (blockIdx * blockSize).toLong() * 4

                var idx = 0
                while (idx + floatStep <= blockSize) {
                    val byteOff = codesOff + idx.toLong()
                    if (byteOff + byteLoadLen > segLen) break
                    val inputVec = FloatVector.fromMemorySegment(
                        floatSpecies, inputSeg, inputStartBytes + idx.toLong() * 4, ByteOrder.LITTLE_ENDIAN
                    )
                    val byteVec = ByteVector.fromMemorySegment(
                        byteSpeciesForFloat, weightSeg, byteOff, ByteOrder.LITTLE_ENDIAN
                    )
                    val codeVec = byteVec.castShape(floatSpecies, 0) as FloatVector
                    accVec = inputVec.mul(codeVec).mul(scaleVec).add(accVec)
                    idx += floatStep
                }
                while (idx < blockSize) {
                    val inputVal = inputSeg.get(FLOAT_LE, inputStartBytes + idx.toLong() * 4)
                    accScalar += inputVal *
                        weightSeg.get(JAVA_BYTE_LE, codesOff + idx.toLong()).toFloat() * scale
                    idx++
                }
            }

            output[outputOffset + o] = accVec.reduceLanes(VectorOperators.ADD) + accScalar
        }
    }

}
