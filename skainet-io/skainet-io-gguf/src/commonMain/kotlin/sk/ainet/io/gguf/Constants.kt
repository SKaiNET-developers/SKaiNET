package sk.ainet.io.gguf

/**
 * This is a kotlin gguf reader related logic interpreted from python code "gguf-py/gguf/constants.py"
 * of github repo "https://github.com/ggerganov/llama.cpp"
 */

//TODO convert the rest of file from constants.py

const val GGUF_MAGIC = 0x46554747u
const val GGUF_VERSION = 3
const val GGUF_DEFAULT_ALIGNMENT = 32

/**
 * GGML quantization types.
 *
 * When new quantization types are added to llama.cpp, they may not be immediately
 * supported here. Use [fromValueOrUnknown] to gracefully handle unknown types
 * (they'll be treated as raw bytes).
 */
enum class GGMLQuantizationType(val value: Int) {
    F32(0),
    F16(1),
    Q4_0(2),
    Q4_1(3),
    Q5_0(6),
    Q5_1(7),
    Q8_0(8),
    Q8_1(9),
    Q2_K(10),
    Q3_K(11),
    Q4_K(12),
    Q5_K(13),
    Q6_K(14),
    Q8_K(15),
    IQ2_XXS(16),
    IQ2_XS(17),
    IQ3_XXS(18),
    IQ1_S(19),
    IQ4_NL(20),
    IQ3_S(21),
    IQ2_S(22),
    IQ4_XS(23),
    I8(24),
    I16(25),
    I32(26),
    I64(27),
    F64(28),
    IQ1_M(29),
    BF16(30),
    TQ1_0(34),
    TQ2_0(35),
    // Note: types 31-33 and 36-38 have been removed in llama.cpp
    MXFP4(39),

    /**
     * Placeholder for unknown/unsupported quantization types.
     * The actual type value is stored but not recognized.
     * Data will be treated as raw bytes.
     */
    UNKNOWN(-1);

    /** Whether this is an unknown/unsupported type */
    val isUnknown: Boolean get() = this == UNKNOWN

    companion object {
        /** Returns the enum value or null if not found */
        fun fromValue(value: Int): GGMLQuantizationType? {
            return entries.find { it.value == value && it != UNKNOWN }
        }

        /**
         * Returns the enum value or [UNKNOWN] if not recognized.
         * Use this for graceful degradation when loading models with new quantization types.
         */
        fun fromValueOrUnknown(value: Int): GGMLQuantizationType {
            return fromValue(value) ?: UNKNOWN
        }
    }
}

// Block size constant
const val QK_K = 256

// Quantization type and corresponding sizes
val GGML_QUANT_SIZES: Map<GGMLQuantizationType, Pair<Int, Int>> = mapOf(
    GGMLQuantizationType.F32 to (1 to 4),
    GGMLQuantizationType.F16 to (1 to 2),
    GGMLQuantizationType.Q4_0 to (32 to 2 + 16),
    GGMLQuantizationType.Q4_1 to (32 to 2 + 2 + 16),
    GGMLQuantizationType.Q5_0 to (32 to 2 + 4 + 16),
    GGMLQuantizationType.Q5_1 to (32 to 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0 to (32 to 2 + 32),
    GGMLQuantizationType.Q8_1 to (32 to 4 + 4 + 32),
    GGMLQuantizationType.Q2_K to (256 to 2 + 2 + QK_K / 16 + QK_K / 4),
    GGMLQuantizationType.Q3_K to (256 to 2 + QK_K / 4 + QK_K / 8 + 12),
    GGMLQuantizationType.Q4_K to (256 to 2 + 2 + QK_K / 2 + 12),
    GGMLQuantizationType.Q5_K to (256 to 2 + 2 + QK_K / 2 + QK_K / 8 + 12),
    GGMLQuantizationType.Q6_K to (256 to 2 + QK_K / 2 + QK_K / 4 + QK_K / 16),
    GGMLQuantizationType.Q8_K to (256 to 4 + QK_K + QK_K / 8),
    GGMLQuantizationType.IQ2_XXS to (256 to 2 + QK_K / 4),
    GGMLQuantizationType.IQ2_XS to (256 to 2 + QK_K / 4 + QK_K / 32),
    GGMLQuantizationType.IQ3_XXS to (256 to 2 + QK_K / 4 + QK_K / 8),
    GGMLQuantizationType.IQ1_S to (256 to 2 + QK_K / 8 + QK_K / 16),
    GGMLQuantizationType.IQ4_NL to (32 to 2 + 16),
    GGMLQuantizationType.IQ3_S to (256 to 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4),
    GGMLQuantizationType.IQ2_S to (256 to 2 + QK_K / 4 + QK_K / 16),
    GGMLQuantizationType.IQ4_XS to (256 to 2 + 2 + QK_K / 2 + QK_K / 64),
    GGMLQuantizationType.I8 to (1 to 1),
    GGMLQuantizationType.I16 to (1 to 2),
    GGMLQuantizationType.I32 to (1 to 4),
    GGMLQuantizationType.I64 to (1 to 8),
    GGMLQuantizationType.F64 to (1 to 8),
    GGMLQuantizationType.IQ1_M to (256 to QK_K / 8 + QK_K / 16 + QK_K / 32),
    GGMLQuantizationType.BF16 to (1 to 2),
    GGMLQuantizationType.TQ1_0 to (256 to 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0 to (256 to 2 + 64),
    // MXFP4: Microscaling FP4 format - 32 elements per block, 17 bytes (16 for data + 1 for scale)
    GGMLQuantizationType.MXFP4 to (32 to 17)
)

enum class GGUFValueType(val value: Int) {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    FLOAT32(6),
    BOOL(7),
    STRING(8),
    ARRAY(9),
    UINT64(10),
    INT64(11),
    FLOAT64(12)
}
