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

    /** GGML Q5_K block quantization: 256 elements per 176-byte block. */
    public data object Q5_K : TensorEncoding {
        public const val BLOCK_SIZE: Int = 256
        public const val BYTES_PER_BLOCK: Int = 176

        override val name: String get() = "Q5_K"
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

    /** GGML Q5_0 block quantization: 32 elements per 22-byte block. */
    public data object Q5_0 : TensorEncoding {
        public const val BLOCK_SIZE: Int = 32
        public const val BYTES_PER_BLOCK: Int = 22

        override val name: String get() = "Q5_0"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /** GGML Q5_1 block quantization: 32 elements per 24-byte block. */
    public data object Q5_1 : TensorEncoding {
        public const val BLOCK_SIZE: Int = 32
        public const val BYTES_PER_BLOCK: Int = 24

        override val name: String get() = "Q5_1"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /**
     * SKaiNET's own ternary packing: 2 bits per element, four **consecutive** elements per byte
     * (element `i` in bits `2*(i % 4)`), code `0,1,2` → `-1,0,+1`, and the scale held out of band
     * by the tensor ([sk.ainet.lang.tensor.data.TernaryTensorData.scale]) rather than in the bytes.
     *
     * This is *not* the GGML on-disk layout: [TQ2_0] carries an FP16 scale per 256-element block
     * and interleaves its elements 32 apart. Use [TQ1_0]/[TQ2_0] for GGUF bytes and this encoding
     * for tensors SKaiNET packs itself.
     */
    public data object TernaryPacked : TensorEncoding {
        override val name: String get() = "Ternary"
        override fun physicalBytes(elementCount: Long): Long =
            (elementCount + 3) / 4
    }

    /**
     * GGML `TQ1_0` — ternary at ~1.69 bits/element: 256 elements per 54-byte block, made of 48
     * base-3 bytes (five ternary digits each, most-significant digit first, scaled by `256/243`),
     * 4 bytes holding the last 16 elements four at a time, and an FP16 block scale.
     *
     * Element order follows `dequantize_row_tq1_0`: the first 32 bytes carry elements
     * `n*32 + m` (`n` = digit, `m` = byte), the next 16 bytes elements `160 + n*16 + m`, and the
     * four `qh` bytes elements `240 + n*4 + j` — decoded by
     * [sk.ainet.lang.memory.TernaryCodec.decodeTq1_0].
     */
    public data object TQ1_0 : TensorEncoding {
        public const val BLOCK_SIZE: Int = 256
        public const val BYTES_PER_BLOCK: Int = 54

        override val name: String get() = "TQ1_0"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /**
     * GGML `TQ2_0` — ternary at ~2.06 bits/element: 256 elements per 66-byte block (64 payload
     * bytes + an FP16 scale). Each byte holds four elements **32 apart**, not four consecutive
     * ones: byte `j + m` of a 32-byte chunk carries element `chunk*128 + l*32 + m` in bit pair `l`
     * (`dequantize_row_tq2_0`) — see [sk.ainet.lang.memory.TernaryCodec.decodeTq2_0].
     */
    public data object TQ2_0 : TensorEncoding {
        public const val BLOCK_SIZE: Int = 256
        public const val BYTES_PER_BLOCK: Int = 66

        override val name: String get() = "TQ2_0"
        override fun physicalBytes(elementCount: Long): Long {
            val blocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE
            return blocks * BYTES_PER_BLOCK
        }
    }

    /**
     * BitNet b1.58 weights: ternary at 2 bits per element, four consecutive elements per byte, with
     * **one FP32 scale for the whole tensor** (b1.58 quantizes per tensor by absmean) stored little
     * endian in the four bytes after the payload.
     *
     * Distinct from [TQ1_0]/[TQ2_0], which are how a b1.58 checkpoint is usually *shipped* in GGUF:
     * this is the in-memory form the ternary kernels consume, whose activations are int8
     * (`W1.58A8` — see [sk.ainet.lang.memory.blockSpec] and its `activation` hint).
     */
    public data object BITNET_B1_58 : TensorEncoding {
        /** Bytes of per-tensor scale appended after the packed codes. */
        public const val SCALE_BYTES: Int = 4

        override val name: String get() = "BitNet-b1.58"
        override fun physicalBytes(elementCount: Long): Long = (elementCount + 3) / 4 + SCALE_BYTES
    }

    /**
     * NeoGPU's multi-plane trit residual format for lm_head-class `[rows, cols]` weights (#1150):
     * **8 sequentially-packed ternary planes + one FP16 scale per row**.
     *
     * Per row, `scale = max|row|` (stored as FP16); the normalized row is decomposed by repeated
     * round-to-trit with ×3 residual scaling, so plane `p` carries weight `1/3^p` and
     *
     *   `w[r, c] ≈ rowScale[r] · Σ_p (code_p(r, c) − 1) / 3^p`
     *
     * — effectively "16 bits as eight ternary digits", with truncation error ≤ `rowScale / (2·3⁷)`.
     * Each plane is a full `[rows, cols]` matrix in the [BITNET_B1_58] payload order (4 codes per
     * byte, low bit-pair first), `rows · cols/4` bytes, planes concatenated, then `rows` FP16
     * little-endian scales. `cols % 4 == 0` required.
     *
     * The point is *speed*, not memory (2 B + ε per weight — FP16-sized): the fused 4-plane LUT
     * kernel reads planes 0–3 in one pass on baseline NEON, and an application can rescore top
     * candidates with planes 4–7 (NeoGPU's two-stage lm_head). [physicalBytes] is `null` — the
     * layout depends on the row count, not just the element count.
     */
    public data object BITNET_PLANES : TensorEncoding {
        /** Number of trit planes. */
        public const val PLANES: Int = 8

        /** Bytes of FP16 per-row scale, per row, after the plane payloads. */
        public const val ROW_SCALE_BYTES: Int = 2

        override val name: String get() = "BitNet-planes"
        override fun physicalBytes(elementCount: Long): Long? = null

        /** Bytes of one plane of a `[rows, cols]` weight (`cols % 4 == 0`). */
        public fun planeStrideBytes(rows: Int, cols: Int): Int = rows * (cols / 4)

        /** Byte offset of the FP16 row-scale table inside the buffer. */
        public fun rowScalesByteOffset(rows: Int, cols: Int): Int = PLANES * planeStrideBytes(rows, cols)

        /** Total buffer size of a `[rows, cols]` weight. */
        public fun bufferBytes(rows: Int, cols: Int): Int =
            rowScalesByteOffset(rows, cols) + rows * ROW_SCALE_BYTES
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
     * Int8 activations with a **per-row (per-token) absmax scale** — the companion format of the
     * ternary weights (`W1.58A8`, #1040).
     *
     * Layout of a `[rows, cols]` activation: `rows * cols` int8 codes in row-major order, followed
     * by `rows` little-endian FP32 scales. A value is `code * scale(row)`; the scale is
     * `absmax(row) / 127`, so a row of zeros has scale zero and decodes to zeros.
     *
     * Deliberately not parameterized by the row length: kernel selection keys on the [Format], and
     * a per-shape encoding would make every hidden size a different key. The row count comes from
     * the view's shape, which is where it belongs. [physicalBytes] is therefore `null` — the byte
     * count needs the row length, and [sk.ainet.lang.memory.I8Absmax.bytesFor] computes it.
     */
    public data object DENSE_I8_ABSMAX : TensorEncoding {
        /** Largest magnitude an int8 code may take; the scale is `absmax / this`. */
        public const val CODE_RANGE: Int = 127

        override val name: String get() = "I8-absmax"
        override fun physicalBytes(elementCount: Long): Long? = null
    }

    /**
     * Opaque / unknown encoding. Used as a fallback for formats the runtime
     * cannot yet interpret but still wants to carry through without error.
     */
    public data class Opaque(override val name: String, val rawBytes: Long) : TensorEncoding {
        override fun physicalBytes(elementCount: Long): Long = rawBytes
    }
}
