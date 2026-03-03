package sk.ainet.lang.types

// Base marker interface for all dtypes
public sealed interface DType {
    public val sizeInBits: Int
    public val name: String

    /**
     * Checks if this data type is compatible with another data type for operations.
     *
     * Compatibility means that the two types can be used together in mathematical
     * operations, potentially with automatic promotion. This method should return
     * true if and only if there exists a valid promotion path between the types.
     *
     * @param other The other data type to check compatibility with
     * @return true if the types are compatible for operations, false otherwise
     */
    public fun isCompatible(other: DType): Boolean

    /**
     * Determines the result type when promoting this type with another type.
     *
     * Promotion rules define how mixed-type operations should be handled.
     * The result should be a type that can represent values from both input types
     * without loss of precision where possible.
     *
     * This method should only be called after verifying compatibility with isCompatible().
     *
     * @param other The other data type to promote with
     * @return The promoted data type that can represent both input types
     * @throws IllegalArgumentException if the types are not compatible
     */
    public fun promoteTo(other: DType): DType

    public companion object {
        /**
         * Registry of all available data types.
         */
        private val typeRegistry: Map<String, DType> = mapOf(
            "Ternary" to Ternary,
            "Int4" to Int4,
            "Int8" to Int8,
            "Int16" to Int16,
            "Int32" to Int32,
            "Int64" to Int64,
            "UInt8" to UInt8,
            "UInt16" to UInt16,
            "UInt32" to UInt32,
            "UInt64" to UInt64,
            "Float16" to FP16,
            "BFloat16" to BF16,
            "Float32" to FP32,
            "Float64" to FP64
        )

        /**
         * Gets all registered data types.
         *
         * @return Map of type names to DType instances
         */
        public fun getAllTypes(): Map<String, DType> = typeRegistry

        /**
         * Finds a data type by name.
         *
         * @param name The name of the data type to find
         * @return The DType instance or null if not found
         */
        public fun findByName(name: String): DType? = typeRegistry[name]

        // --- Java-friendly static accessors (A.1 / A.6) ---

        /** @return The FP32 (32-bit float) data type. */
        @kotlin.jvm.JvmStatic public fun fp32(): DType = FP32
        /** @return The FP16 (16-bit float) data type. */
        @kotlin.jvm.JvmStatic public fun fp16(): DType = FP16
        /** @return The FP64 (64-bit float) data type. */
        @kotlin.jvm.JvmStatic public fun fp64(): DType = FP64
        /** @return The BF16 (BFloat16) data type. */
        @kotlin.jvm.JvmStatic public fun bf16(): DType = BF16
        /** @return The Int4 (4-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun int4(): DType = Int4
        /** @return The Int8 (8-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun int8(): DType = Int8
        /** @return The Int16 (16-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun int16(): DType = Int16
        /** @return The Int32 (32-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun int32(): DType = Int32
        /** @return The Int64 (64-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun int64(): DType = Int64
        /** @return The UInt8 (unsigned 8-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun uint8(): DType = UInt8
        /** @return The UInt16 (unsigned 16-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun uint16(): DType = UInt16
        /** @return The UInt32 (unsigned 32-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun uint32(): DType = UInt32
        /** @return The UInt64 (unsigned 64-bit integer) data type. */
        @kotlin.jvm.JvmStatic public fun uint64(): DType = UInt64
        /** @return The Ternary data type. */
        @kotlin.jvm.JvmStatic public fun ternary(): DType = Ternary

    }

}

