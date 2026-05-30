package sk.ainet.lang.tensor.storage

/**
 * Physical storage encoding — how tensor data is laid out in memory.
 *
 * A [TensorEncoding] describes the byte-level format of a buffer, independent
 * of the logical numeric type ([LogicalDType]). For example, a FLOAT32 tensor
 * may be stored as [Dense] (4 bytes per element) or as [Q4_K] (packed 4-bit
 * blocks with per-block scales).
 *
 * Encodings are sealed so that pattern-matching in loaders and backends is
 * exhaustive and compiler-checked.
 */
public sealed interface TensorEncoding {

    /** Human-readable name for diagnostics and memory reports. */
    public val name: String

    /**
     * Physical bytes required to store [elementCount] logical elements
     * in this encoding, or `null` if the encoding is opaque/variable.
     */
    public fun physicalBytes(elementCount: Long): Long?

    /** Dense element-per-slot layout. One element occupies a fixed number of bytes. */
    public data class Dense(val bytesPerElement: Int) : TensorEncoding {
        override val name: String get() = "Dense(${bytesPerElement}B)"
        override fun physicalBytes(elementCount: Long): Long = elementCount * bytesPerElement
    }

    /** GGML Q4_K block quantization: 256 elements per 144-byte block. */
    public data object Q4_K : TensorEncoding {
        public const val BLOCK_SIZE: Int = 256
        public const val BYTES_PER_BLOCK: Int = 144

        override val name: String get() = "Q4_K"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /** GGML Q6_K block quantization: 256 elements per 210-byte block. */
    public data object Q6_K : TensorEncoding {
        public const val BLOCK_SIZE: Int = 256
        public const val BYTES_PER_BLOCK: Int = 210

        override val name: String get() = "Q6_K"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /** GGML Q4_0 block quantization: 32 elements per 18-byte block. */
    public data object Q4_0 : TensorEncoding {
        public const val BLOCK_SIZE: Int = 32
        public const val BYTES_PER_BLOCK: Int = 18

        override val name: String get() = "Q4_0"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /** GGML Q8_0 block quantization: 32 elements per 34-byte block. */
    public data object Q8_0 : TensorEncoding {
        public const val BLOCK_SIZE: Int = 32
        public const val BYTES_PER_BLOCK: Int = 34

        override val name: String get() = "Q8_0"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /** Ternary encoding: 2 bits per element, packed 4 elements per byte. */
    public data object TernaryPacked : TensorEncoding {
        override val name: String get() = "Ternary"
        override fun physicalBytes(elementCount: Long): Long =
            (elementCount + 3) / 4
    }

    /**
     * TurboQuant PolarOnly encoding: rotation + scalar quantization + bit-packing.
     *
     * Backend-friendly variant that omits the QJL residual stage.
     * Configurable bits per element (2, 3, 4, or 8).
     *
     * Block layout: [rotationSeed(4B)] [scales(numGroups * 2B)] [codes(packed bits)]
     *
     * @param bitsPerElement Number of bits per quantized code (2, 3, 4, or 8)
     * @param blockSize Number of elements per block (must be power of 2, typically 64 or 128)
     */
    public data class TurboQuantPolar(
        val bitsPerElement: Int = 4,
        val blockSize: Int = 128
    ) : TensorEncoding {
        init {
            require(bitsPerElement in setOf(2, 3, 4, 8)) {
                "bitsPerElement must be 2, 3, 4, or 8, got $bitsPerElement"
            }
            require(blockSize > 0 && (blockSize and (blockSize - 1)) == 0) {
                "blockSize must be a positive power of 2, got $blockSize"
            }
        }

        /** Number of quantization groups per block (each group has its own scale). */
        val numGroups: Int get() = blockSize / 32

        override val name: String get() = "TurboQuant-Polar-${bitsPerElement}b"

        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + blockSize - 1) / blockSize
            val seedBytes = 4L                                    // rotation seed per block
            val scaleBytes = numGroups * 2L                       // FP16 scale per group
            val codeBytes = (blockSize.toLong() * bitsPerElement + 7) / 8  // packed codes
            return blocks * (seedBytes + scaleBytes + codeBytes)
        }
    }

    /**
     * TurboQuant PolarPlusQjl encoding: rotation + scalar quantization +
     * QJL residual + bit-packing.
     *
     * Closest to the official TurboQuant paper. The QJL residual stage
     * preserves inner-product accuracy at the cost of additional storage.
     *
     * @param bitsPerElement Bits for the primary quantization (2, 3, 4, or 8)
     * @param residualBits Bits for the QJL residual (typically 1 or 2)
     * @param blockSize Elements per block
     */
    public data class TurboQuantPolarQjl(
        val bitsPerElement: Int = 4,
        val residualBits: Int = 1,
        val blockSize: Int = 128
    ) : TensorEncoding {
        init {
            require(bitsPerElement in setOf(2, 3, 4, 8)) {
                "bitsPerElement must be 2, 3, 4, or 8, got $bitsPerElement"
            }
            require(residualBits in 1..4) {
                "residualBits must be 1-4, got $residualBits"
            }
            require(blockSize > 0 && (blockSize and (blockSize - 1)) == 0) {
                "blockSize must be a positive power of 2, got $blockSize"
            }
        }

        val numGroups: Int get() = blockSize / 32

        override val name: String
            get() = "TurboQuant-PolarQjl-${bitsPerElement}b+${residualBits}r"

        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + blockSize - 1) / blockSize
            val seedBytes = 4L
            val scaleBytes = numGroups * 2L
            val codeBytes = (blockSize.toLong() * bitsPerElement + 7) / 8
            val residualBytes = (blockSize.toLong() * residualBits + 7) / 8
            return blocks * (seedBytes + scaleBytes + codeBytes + residualBytes)
        }
    }

    /**
     * Opaque / unknown encoding. Used as a fallback for formats the runtime
     * cannot yet interpret but still wants to carry through without error.
     */
    public data class Opaque(override val name: String, val rawBytes: Long) : TensorEncoding {
        override fun physicalBytes(elementCount: Long): Long = rawBytes
    }
}
