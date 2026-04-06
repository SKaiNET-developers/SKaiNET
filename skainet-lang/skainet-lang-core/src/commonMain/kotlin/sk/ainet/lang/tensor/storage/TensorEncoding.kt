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
     * Opaque / unknown encoding. Used as a fallback for formats the runtime
     * cannot yet interpret but still wants to carry through without error.
     */
    public data class Opaque(override val name: String, val rawBytes: Long) : TensorEncoding {
        override fun physicalBytes(elementCount: Long): Long = rawBytes
    }
}
