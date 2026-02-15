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
        var sum = 0f

        // Try vectorized path if we have enough elements
        val floatStep = floatSpecies.length()
        var idx = 0

        if (floatStep <= blockSize) {
            val loopBound = floatSpecies.loopBound(blockSize)

            while (idx < loopBound) {
                // Load input floats
                val inputVec = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)

                // Load codes and convert to floats (scalar loop for byte-to-float)
                val codeFloats = FloatArray(floatStep)
                for (i in 0 until floatStep) {
                    codeFloats[i] = codes[codesOffset + idx + i].toFloat()
                }
                val codeVec = FloatVector.fromArray(floatSpecies, codeFloats, 0)

                // Multiply and accumulate
                val product = inputVec.mul(codeVec)
                sum += product.reduceLanes(VectorOperators.ADD)

                idx += floatStep
            }
        }

        // Scalar tail
        while (idx < blockSize) {
            sum += input[inputOffset + idx] * codes[codesOffset + idx].toFloat()
            idx++
        }

        return sum * scale
    }

    /**
     * Compute dot product for Q4_K sub-block (32 elements).
     *
     * Q4_K sub-block: 32 4-bit codes with per-sub-block scale and min.
     * Result = sum(input[i] * code[i]) * scale + sum(input[i]) * min
     *
     * @param input Input float array
     * @param inputOffset Starting offset in input array
     * @param qs Packed 4-bit codes (16 bytes for 32 elements)
     * @param qsOffset Starting offset in qs array
     * @param scale Sub-block scale
     * @param min Sub-block minimum value
     * @return Weighted dot product result
     */
    fun dotQ4_KSubBlock(
        input: FloatArray,
        inputOffset: Int,
        qs: ByteArray,
        qsOffset: Int,
        scale: Float,
        min: Float
    ): Float {
        val subBlockSize = 32
        var codeSum = 0f
        var inputSum = 0f

        val floatStep = floatSpecies.length()
        var idx = 0

        if (floatStep <= subBlockSize) {
            val loopBound = floatSpecies.loopBound(subBlockSize)

            while (idx < loopBound) {
                // Load input floats
                val inputVec = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)

                // Accumulate input sum
                inputSum += inputVec.reduceLanes(VectorOperators.ADD)

                // Unpack 4-bit codes (2 codes per byte) and convert to floats
                val codeFloats = FloatArray(floatStep)
                for (i in 0 until floatStep) {
                    val elemIdx = idx + i
                    val byteIdx = qsOffset + elemIdx / 2
                    val codeByte = qs[byteIdx].toInt() and 0xFF
                    val code = if (elemIdx % 2 == 0) codeByte and 0x0F else codeByte ushr 4
                    codeFloats[i] = code.toFloat()
                }
                val codeVec = FloatVector.fromArray(floatSpecies, codeFloats, 0)

                // Multiply input by codes and accumulate
                val product = inputVec.mul(codeVec)
                codeSum += product.reduceLanes(VectorOperators.ADD)

                idx += floatStep
            }
        }

        // Scalar tail
        while (idx < subBlockSize) {
            val inputVal = input[inputOffset + idx]
            inputSum += inputVal

            val byteIdx = qsOffset + idx / 2
            val codeByte = qs[byteIdx].toInt() and 0xFF
            val code = if (idx % 2 == 0) codeByte and 0x0F else codeByte ushr 4
            codeSum += inputVal * code.toFloat()

            idx++
        }

        return codeSum * scale + inputSum * min
    }

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
        val subBlocksPerBlock = 8
        val bytesPerBlock = 144  // 2 d + 2 dMin + 12 scales + 128 codes
        val blocksPerInputDim = (inputDim + blockSize - 1) / blockSize

        for (o in 0 until outputDim) {
            var acc = 0f

            for (blockIdx in 0 until blocksPerInputDim) {
                val weightBlockOffset = (blockIdx * outputDim + o) * bytesPerBlock

                // Read f16 d and dMin
                val dBits = (packedWeights[weightBlockOffset + 1].toInt() and 0xFF shl 8) or
                    (packedWeights[weightBlockOffset].toInt() and 0xFF)
                val dMinBits = (packedWeights[weightBlockOffset + 3].toInt() and 0xFF shl 8) or
                    (packedWeights[weightBlockOffset + 2].toInt() and 0xFF)
                val d = halfToFloat(dBits)
                val dMin = halfToFloat(dMinBits)

                // Process each sub-block
                val scalesOffset = weightBlockOffset + 4
                val codesOffset = weightBlockOffset + 16

                for (subBlockIdx in 0 until subBlocksPerBlock) {
                    // Extract 12-bit packed scale/min indices
                    val bitPos = subBlockIdx * 12
                    val bytePos = bitPos / 8
                    val bitShift = bitPos % 8

                    val packed = (packedWeights[scalesOffset + bytePos].toInt() and 0xFF) or
                        ((packedWeights.getOrElse(scalesOffset + bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
                        ((packedWeights.getOrElse(scalesOffset + bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)

                    val scaleIdx = (packed ushr bitShift) and 0x3F
                    val minIdx = (packed ushr (bitShift + 6)) and 0x3F

                    val scale = d * (scaleIdx / 63.0f)
                    val min = dMin * (minIdx / 63.0f)

                    // Input and codes offsets for this sub-block
                    val inputStart = blockIdx * blockSize + subBlockIdx * subBlockSize
                    val qsStart = codesOffset + subBlockIdx * 16  // 16 bytes = 32 4-bit codes

                    if (inputStart < inputDim) {
                        acc += dotQ4_KSubBlock(input, inputStart, packedWeights, qsStart, scale, min)
                    }
                }
            }

            output[outputOffset + o] = acc
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
     * Each byte packs two 4-bit codes: lo nibble = first, hi nibble = second.
     *
     * Uses the preferred vector species (AVX-256 gives 8-wide, AVX-512 gives 16-wide).
     */
    fun dotQ4_0BlockMemSeg(
        input: FloatArray,
        inputOffset: Int,
        weightSeg: MemorySegment,
        blockByteOffset: Long,
    ): Float {
        val blockSize = 32
        val scaleOffset = blockByteOffset
        val codesOffset = blockByteOffset + 2

        // Read f16 scale
        val b0 = weightSeg.get(JAVA_BYTE_LE, scaleOffset).toInt() and 0xFF
        val b1 = weightSeg.get(JAVA_BYTE_LE, scaleOffset + 1).toInt() and 0xFF
        val scale = halfToFloat((b1 shl 8) or b0)

        val floatStep = floatSpecies.length()
        var sum = 0f
        var idx = 0

        if (floatStep <= blockSize) {
            val loopBound = floatSpecies.loopBound(blockSize)
            while (idx < loopBound) {
                val inputVec = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)
                // Unpack nibbles into floats
                val codeFloats = FloatArray(floatStep)
                for (i in 0 until floatStep) {
                    val elemIdx = idx + i
                    val packedByte = weightSeg.get(JAVA_BYTE_LE, codesOffset + (elemIdx / 2).toLong()).toInt() and 0xFF
                    // Subtract 8 to center codes around zero (Q4_0 convention)
                    codeFloats[i] = (if (elemIdx % 2 == 0) (packedByte and 0x0F) else (packedByte ushr 4)).toFloat() - 8f
                }
                val codeVec = FloatVector.fromArray(floatSpecies, codeFloats, 0)
                sum += inputVec.mul(codeVec).reduceLanes(VectorOperators.ADD)
                idx += floatStep
            }
        }

        // Scalar tail
        while (idx < blockSize) {
            val packedByte = weightSeg.get(JAVA_BYTE_LE, codesOffset + (idx / 2).toLong()).toInt() and 0xFF
            val code = (if (idx % 2 == 0) (packedByte and 0x0F) else (packedByte ushr 4)).toFloat() - 8f
            sum += input[inputOffset + idx] * code
            idx++
        }

        return sum * scale
    }

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

        for (o in 0 until outputDim) {
            var acc = 0f
            for (blockIdx in 0 until blocksPerRow) {
                val blockOff = weightByteOffset +
                    (o.toLong() * blocksPerRow + blockIdx) * bytesPerBlock
                val inputStart = blockIdx * blockSize
                acc += dotQ4_0BlockMemSeg(input, inputStart, weightSeg, blockOff)
            }
            output[outputOffset + o] = acc
        }
    }

    /**
     * F32 x Q4_K matrix-vector multiply using MemorySegment for packed Q4_K weights.
     * Same block structure as matmulQ4_KVec but reads from MemorySegment.
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
        val subBlocksPerBlock = 8
        val bytesPerBlock = 144L
        val blocksPerRow = (inputDim + blockSize - 1) / blockSize

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

                for (sb in 0 until subBlocksPerBlock) {
                    val bitPos = sb * 12
                    val bytePos = bitPos / 8
                    val bitShift = bitPos % 8

                    val b0 = weightSeg.get(JAVA_BYTE_LE, scalesOff + bytePos).toInt() and 0xFF
                    val b1 = if (bytePos + 1 < 12) weightSeg.get(JAVA_BYTE_LE, scalesOff + bytePos + 1).toInt() and 0xFF else 0
                    val b2 = if (bytePos + 2 < 12) weightSeg.get(JAVA_BYTE_LE, scalesOff + bytePos + 2).toInt() and 0xFF else 0
                    val packed = b0 or (b1 shl 8) or (b2 shl 16)

                    val scaleIdx = (packed ushr bitShift) and 0x3F
                    val minIdx = (packed ushr (bitShift + 6)) and 0x3F

                    val scale = d * (scaleIdx / 63.0f)
                    val min = dMin * (minIdx / 63.0f)

                    val inputStart = blockIdx * blockSize + sb * subBlockSize
                    val qsStart = codesOff + sb * 16L

                    if (inputStart < inputDim) {
                        acc += dotQ4_KSubBlockMemSeg(input, inputStart, weightSeg, qsStart, scale, min)
                    }
                }
            }

            output[outputOffset + o] = acc
        }
    }

    /**
     * Q4_K sub-block dot product reading codes from MemorySegment.
     */
    private fun dotQ4_KSubBlockMemSeg(
        input: FloatArray,
        inputOffset: Int,
        weightSeg: MemorySegment,
        qsOffset: Long,
        scale: Float,
        min: Float,
    ): Float {
        val subBlockSize = 32
        var codeSum = 0f
        var inputSum = 0f

        val floatStep = floatSpecies.length()
        var idx = 0

        if (floatStep <= subBlockSize) {
            val loopBound = floatSpecies.loopBound(subBlockSize)
            while (idx < loopBound) {
                val inputVec = FloatVector.fromArray(floatSpecies, input, inputOffset + idx)
                inputSum += inputVec.reduceLanes(VectorOperators.ADD)

                val codeFloats = FloatArray(floatStep)
                for (i in 0 until floatStep) {
                    val elemIdx = idx + i
                    val packedByte = weightSeg.get(JAVA_BYTE_LE, qsOffset + (elemIdx / 2).toLong()).toInt() and 0xFF
                    codeFloats[i] = (if (elemIdx % 2 == 0) packedByte and 0x0F else packedByte ushr 4).toFloat()
                }
                val codeVec = FloatVector.fromArray(floatSpecies, codeFloats, 0)
                codeSum += inputVec.mul(codeVec).reduceLanes(VectorOperators.ADD)
                idx += floatStep
            }
        }

        while (idx < subBlockSize) {
            val inputVal = input[inputOffset + idx]
            inputSum += inputVal
            val packedByte = weightSeg.get(JAVA_BYTE_LE, qsOffset + (idx / 2).toLong()).toInt() and 0xFF
            val code = if (idx % 2 == 0) packedByte and 0x0F else packedByte ushr 4
            codeSum += inputVal * code.toFloat()
            idx++
        }

        return codeSum * scale + inputSum * min
    }

    /**
     * F32 x Q8_0 matrix-vector multiply using MemorySegment for packed Q8_0 weights.
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

        for (o in 0 until outputDim) {
            var acc = 0f

            for (blockIdx in 0 until blocksPerRow) {
                val blockOff = weightByteOffset +
                    (o.toLong() * blocksPerRow + blockIdx) * bytesPerBlock

                val b0 = weightSeg.get(JAVA_BYTE_LE, blockOff).toInt() and 0xFF
                val b1 = weightSeg.get(JAVA_BYTE_LE, blockOff + 1).toInt() and 0xFF
                val scale = halfToFloat((b1 shl 8) or b0)

                val codesOff = blockOff + 2
                val inputStart = blockIdx * blockSize

                // Vectorized dot product over 32 elements
                val floatStep = floatSpecies.length()
                var sum = 0f
                var idx = 0
                if (floatStep <= blockSize) {
                    val loopBound = floatSpecies.loopBound(blockSize)
                    while (idx < loopBound) {
                        val inputVec = FloatVector.fromArray(floatSpecies, input, inputStart + idx)
                        val codeFloats = FloatArray(floatStep)
                        for (i in 0 until floatStep) {
                            codeFloats[i] = weightSeg.get(JAVA_BYTE_LE, codesOff + (idx + i).toLong()).toFloat()
                        }
                        val codeVec = FloatVector.fromArray(floatSpecies, codeFloats, 0)
                        sum += inputVec.mul(codeVec).reduceLanes(VectorOperators.ADD)
                        idx += floatStep
                    }
                }
                while (idx < blockSize) {
                    sum += input[inputStart + idx] * weightSeg.get(JAVA_BYTE_LE, codesOff + idx.toLong()).toFloat()
                    idx++
                }

                acc += sum * scale
            }

            output[outputOffset + o] = acc
        }
    }
}
