package sk.ainet.io.model

import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int64
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import sk.ainet.lang.types.UInt16
import sk.ainet.lang.types.UInt32
import sk.ainet.lang.types.UInt64
import sk.ainet.lang.types.UInt8

/**
 * Bidirectional mapping utility between SKaiNET DType and unified DataType.
 *
 * This utility provides:
 * - Conversion from DataType to SKaiNET DType (for tensor creation)
 * - Conversion from SKaiNET DType to DataType (for display)
 * - Size calculations and compatibility checks
 */
public object DTypeMapping {

    /**
     * Converts a DataType to the corresponding SKaiNET DType.
     *
     * @param dataType The DataType to convert
     * @return The corresponding SKaiNET DType, or null if no mapping exists
     */
    public fun toSkainetDType(dataType: DataType): DType? = when (dataType) {
        DataType.FLOAT64 -> FP64
        DataType.FLOAT32 -> FP32
        DataType.FLOAT16 -> FP16
        DataType.BFLOAT16 -> BF16
        DataType.INT64 -> Int64
        DataType.INT32 -> Int32
        DataType.INT16 -> Int16
        DataType.INT8 -> Int8
        DataType.UINT64 -> UInt64
        DataType.UINT32 -> UInt32
        DataType.UINT16 -> UInt16
        DataType.UINT8 -> UInt8
        // Quantized types
        DataType.QUANT4 -> Int4  // Best approximation
        DataType.QUANT8 -> Int8  // Best approximation
        // Types without direct SKaiNET mapping
        DataType.BOOL -> null  // Could map to Int8 but semantically different
        DataType.STRING -> null
        DataType.UNKNOWN -> null
    }

    /**
     * Converts a SKaiNET DType to the corresponding DataType.
     *
     * @param dtype The SKaiNET DType to convert
     * @return The corresponding DataType
     */
    public fun fromSkainetDType(dtype: DType): DataType = when (dtype) {
        FP64 -> DataType.FLOAT64
        FP32 -> DataType.FLOAT32
        FP16 -> DataType.FLOAT16
        BF16 -> DataType.BFLOAT16
        Int64 -> DataType.INT64
        Int32 -> DataType.INT32
        Int16 -> DataType.INT16
        Int8 -> DataType.INT8
        Int4 -> DataType.INT8   // Closest approximation (no 4-bit in DataType)
        Ternary -> DataType.INT8  // Closest approximation
        UInt64 -> DataType.UINT64
        UInt32 -> DataType.UINT32
        UInt16 -> DataType.UINT16
        UInt8 -> DataType.UINT8
    }

    /**
     * Checks if a DataType can be natively represented in SKaiNET.
     *
     * @param dataType The DataType to check
     * @return true if SKaiNET has a corresponding DType
     */
    public fun isNativelySupported(dataType: DataType): Boolean =
        toSkainetDType(dataType) != null

    /**
     * Gets the size in bytes per element for a DataType.
     *
     * @param dataType The DataType to get size for
     * @return Size in bytes, or null for variable-size types
     */
    public fun bytesPerElement(dataType: DataType): Int? = dataType.sizeInBytes

    /**
     * Gets the size in bits for a DataType.
     *
     * @param dataType The DataType to get bit size for
     * @return Size in bits, or null for variable-size types
     */
    public fun bitsPerElement(dataType: DataType): Int? = dataType.sizeInBits

    /**
     * Gets all DataTypes that are natively supported by SKaiNET.
     */
    public fun nativelySupportedTypes(): List<DataType> =
        DataType.entries.filter { isNativelySupported(it) }

    /**
     * Gets all SKaiNET DTypes.
     */
    public fun allSkainetTypes(): List<DType> = listOf(
        FP64, FP32, FP16, BF16,
        Int64, Int32, Int16, Int8, Int4, Ternary,
        UInt64, UInt32, UInt16, UInt8
    )

    /**
     * Finds the best matching SKaiNET DType for a given DataType,
     * potentially with lossy conversion.
     *
     * @param dataType The DataType to find a match for
     * @return A pair of (DType, isLossy) where isLossy indicates if conversion loses information
     */
    public fun findBestMatch(dataType: DataType): Pair<DType, Boolean>? = when (dataType) {
        DataType.FLOAT64 -> FP64 to false
        DataType.FLOAT32 -> FP32 to false
        DataType.FLOAT16 -> FP16 to false
        DataType.BFLOAT16 -> BF16 to false
        DataType.INT64 -> Int64 to false
        DataType.INT32 -> Int32 to false
        DataType.INT16 -> Int16 to false
        DataType.INT8 -> Int8 to false
        DataType.UINT64 -> UInt64 to false
        DataType.UINT32 -> UInt32 to false
        DataType.UINT16 -> UInt16 to false
        DataType.UINT8 -> UInt8 to false
        DataType.QUANT4 -> Int4 to true  // Lossy: needs dequantization
        DataType.QUANT8 -> Int8 to true  // Lossy: needs dequantization
        DataType.BOOL -> Int8 to true  // Lossy: bool -> int8
        DataType.STRING -> null
        DataType.UNKNOWN -> null
    }

    /**
     * Gets the display name for a DataType including SKaiNET support status.
     */
    public fun getDisplayInfo(dataType: DataType): String {
        val native = if (isNativelySupported(dataType)) " (native)" else ""
        return "${dataType.displayName}$native"
    }
}
