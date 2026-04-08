package sk.ainet.lang.tensor.ops.turboquant

import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * End-to-end TurboQuant encode/decode codec.
 *
 * Wires together the full TurboQuant pipeline:
 * 1. Random rotation (spread quantization error)
 * 2. Scalar quantization (map to N-bit codes)
 * 3. Optional QJL residual (preserve inner-product accuracy)
 * 4. Bit-packing (compact storage)
 *
 * Supports two variants:
 * - **PolarOnly**: Steps 1-2-4 (fast, backend-friendly)
 * - **PolarPlusQjl**: Steps 1-2-3-4 (higher accuracy)
 *
 * Usage:
 * ```kotlin
 * val encoded = TurboQuantCodec.encode(vector, config)
 * val decoded = TurboQuantCodec.decode(encoded)
 * ```
 */
public object TurboQuantCodec {

    /**
     * Encode a float vector using TurboQuant.
     *
     * @param input  Raw float vector (e.g., a K or V projection for one head)
     * @param config Encoding configuration
     * @return Encoded block ready for storage
     */
    public fun encode(input: FloatArray, config: TurboQuantConfig): TurboQuantBlock {
        // 1. Random rotation
        val rotated = input.copyOf()
        RandomRotation.rotate(rotated, config.seed)

        // 2. Scalar quantization
        val quantized = ScalarQuantizer.quantize(rotated, config.bits)

        // 3. Bit-packing
        val packedCodes = BitPacker.pack(quantized.codes, config.bits)

        // 4. Optional QJL residual
        val residual = if (config.useQjl) {
            val dequantized = ScalarQuantizer.dequantize(quantized)
            val residualVec = FloatArray(input.size) { rotated[it] - dequantized[it] }
            QjlResidual.encode(residualVec, config.residualBits, config.seed + 1)
        } else null

        return TurboQuantBlock(
            packedCodes = packedCodes,
            scales = quantized.scales,
            seed = config.seed,
            bits = config.bits,
            elementCount = input.size,
            residual = residual
        )
    }

    /**
     * Decode a TurboQuant block back to float values.
     *
     * @param block The encoded block
     * @return Reconstructed float vector
     */
    public fun decode(block: TurboQuantBlock): FloatArray {
        // 1. Unpack codes
        val codes = BitPacker.unpack(block.packedCodes, block.elementCount, block.bits)

        // 2. Dequantize
        val output = FloatArray(block.elementCount)
        ScalarQuantizer.dequantizeInto(codes, block.scales, output)

        // 3. Add QJL residual if present
        if (block.residual != null) {
            QjlResidual.decode(block.residual, output, block.seed + 1)
        }

        // 4. Inverse rotation
        RandomRotation.inverseRotate(output, block.seed)

        return output
    }

    /**
     * Compute the byte size of an encoded block.
     */
    public fun encodedSize(elementCount: Int, config: TurboQuantConfig): Int {
        val codeBytes = BitPacker.packedSize(elementCount, config.bits)
        val scaleBytes = ((elementCount + ScalarQuantizer.GROUP_SIZE - 1) / ScalarQuantizer.GROUP_SIZE) * 4
        val seedBytes = 4
        val residualBytes = if (config.useQjl) {
            BitPacker.packedSize(elementCount, config.residualBits) + 4 // packed + scale
        } else 0
        return codeBytes + scaleBytes + seedBytes + residualBytes
    }
}

/**
 * Configuration for TurboQuant encoding.
 */
public data class TurboQuantConfig(
    /** Bits per quantized code (2, 3, 4, or 8). */
    val bits: Int = 4,
    /** Whether to use QJL residual stage. */
    val useQjl: Boolean = false,
    /** Bits for QJL residual (1-4, only used if [useQjl] is true). */
    val residualBits: Int = 1,
    /** Deterministic seed for random rotation. */
    val seed: Int = 0
) {
    init {
        require(bits in setOf(2, 3, 4, 8)) { "bits must be 2, 3, 4, or 8, got $bits" }
        if (useQjl) {
            require(residualBits in 1..4) { "residualBits must be 1-4, got $residualBits" }
        }
    }

    /** Create a config for PolarOnly variant. */
    public companion object {
        public fun polarOnly(bits: Int = 4, seed: Int = 0): TurboQuantConfig =
            TurboQuantConfig(bits = bits, useQjl = false, seed = seed)

        public fun polarPlusQjl(bits: Int = 4, residualBits: Int = 1, seed: Int = 0): TurboQuantConfig =
            TurboQuantConfig(bits = bits, useQjl = true, residualBits = residualBits, seed = seed)
    }
}

/**
 * A single TurboQuant-encoded block.
 *
 * Contains all data needed to reconstruct the original float vector.
 */
public data class TurboQuantBlock(
    /** Bit-packed quantization codes. */
    val packedCodes: ByteArray,
    /** Per-group scale factors. */
    val scales: FloatArray,
    /** Rotation seed for reproducibility. */
    val seed: Int,
    /** Bits per code. */
    val bits: Int,
    /** Number of logical float elements. */
    val elementCount: Int,
    /** Optional QJL residual (null for PolarOnly). */
    val residual: EncodedResidual? = null
) {
    /** Total bytes used by this block. */
    val sizeInBytes: Int
        get() = packedCodes.size + scales.size * 4 + 4 + (residual?.packedSizeBytes ?: 0)

    val isPolarOnly: Boolean get() = residual == null

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TurboQuantBlock) return false
        return seed == other.seed &&
            bits == other.bits &&
            elementCount == other.elementCount &&
            packedCodes.contentEquals(other.packedCodes) &&
            scales.contentEquals(other.scales) &&
            residual == other.residual
    }

    override fun hashCode(): Int {
        var result = packedCodes.contentHashCode()
        result = 31 * result + scales.contentHashCode()
        result = 31 * result + seed
        result = 31 * result + bits
        result = 31 * result + elementCount
        result = 31 * result + (residual?.hashCode() ?: 0)
        return result
    }
}
