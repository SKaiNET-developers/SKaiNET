package sk.ainet.io.safetensors

/**
 * SafeTensors format constants.
 *
 * SafeTensors is a simple, safe format for storing tensors:
 * - 8 bytes: header size (little-endian u64)
 * - N bytes: JSON header with tensor metadata
 * - Remaining: raw tensor data at specified offsets
 *
 * Reference: https://huggingface.co/docs/safetensors
 */

/** Size of the header length field in bytes */
const val HEADER_SIZE_BYTES = 8

/** Maximum allowed header size (100 MB) - safety limit */
const val MAX_HEADER_SIZE = 100 * 1024 * 1024

/** Key for custom metadata in the JSON header */
const val METADATA_KEY = "__metadata__"

/**
 * SafeTensors data types as strings (as they appear in JSON).
 */
object SafeTensorsDataTypes {
    const val BOOL = "BOOL"
    const val U8 = "U8"
    const val I8 = "I8"
    const val U16 = "U16"
    const val I16 = "I16"
    const val U32 = "U32"
    const val I32 = "I32"
    const val U64 = "U64"
    const val I64 = "I64"
    const val F16 = "F16"
    const val BF16 = "BF16"
    const val F32 = "F32"
    const val F64 = "F64"

    /** Size in bytes for each data type */
    val SIZES: Map<String, Int> = mapOf(
        BOOL to 1,
        U8 to 1,
        I8 to 1,
        U16 to 2,
        I16 to 2,
        U32 to 4,
        I32 to 4,
        U64 to 8,
        I64 to 8,
        F16 to 2,
        BF16 to 2,
        F32 to 4,
        F64 to 8
    )

    /** Get size in bytes for a dtype, returns null for unknown types */
    fun sizeOf(dtype: String): Int? = SIZES[dtype.uppercase()]
}
