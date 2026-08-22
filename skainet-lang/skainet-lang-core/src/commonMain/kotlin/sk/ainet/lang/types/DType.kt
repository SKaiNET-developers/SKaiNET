package sk.ainet.lang.types

import kotlin.jvm.JvmStatic
import kotlin.reflect.KClass

/**
 * The logical element type of a tensor: what a value *means* when you read it.
 *
 * `DType` is sealed with exactly fourteen `object` members, so it is switchable like an enum
 * (`when (dtype) { FP32 -> … }`) while each member still serves as the generic witness of
 * `Tensor<T : DType, V>` through [witness]. SKEEP-003 (decision #13): this single type replaces
 * the storage-side `LogicalDType` enum; `Format = (DType, TensorEncoding)` has exactly one dtype
 * type.
 */
public sealed interface DType {
    public val sizeInBits: Int
    public val name: String

    /**
     * The `KClass` witness of this dtype — the value `Tensor.dtype` / `TensorData<T, V>` carry as
     * their type argument. Each member returns its own class (`FP32.witness == FP32::class`), so
     * [DType.fromWitness] maps a `Tensor.dtype` back to the `DType` object. (Not to be confused with
     * [kotlinClass], which is the KClass of the *value* representation, e.g. `Float::class`.)
     */
    public val witness: KClass<out DType>

    /** Whether the type carries a sign (false only for the unsigned integer types). */
    public val isSigned: Boolean get() = true

    /** Storage width of one element rounded up to whole bytes. */
    public val sizeInBytes: Int get() = (sizeInBits + 7) / 8

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
        // All companion collections are lazy on purpose: `DType` now has default members, so on
        // the JVM initializing any `object` (e.g. FP32) also initializes this interface's statics;
        // an eager map would capture the half-initialized object (class-init cycle →
        // ExceptionInInitializerError). Lazy delegates defer the member references until first use.
        /**
         * Registry of all available data types.
         */
        private val typeRegistry: Map<String, DType> by lazy { mapOf(
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
        ) }

        /**
         * Gets all registered data types.
         *
         * @return Map of type names to DType instances
         */
        public fun getAllTypes(): Map<String, DType> = typeRegistry

        /**
         * All fourteen dtypes in a stable order (the storage-layer order: ternary, signed ints,
         * unsigned ints, floats) — the enum-like `entries` of this sealed type.
         */
        public val entries: List<DType> by lazy {
            listOf(
                Ternary, Int4, Int8, Int16, Int32, Int64,
                UInt8, UInt16, UInt32, UInt64,
                FP16, BF16, FP32, FP64,
            )
        }

        // Identity-keyed (KClass equality), never name-based: stable on JS/Wasm where class
        // names may be minified.
        private val byWitness: Map<KClass<out DType>, DType> by lazy { entries.associateBy { it.witness } }

        /**
         * The [DType] whose [witness] is [kclass], or `null` if [kclass] is not one of the
         * fourteen dtype classes (e.g. `DType::class` itself).
         */
        @JvmStatic
        public fun fromWitnessOrNull(kclass: KClass<out DType>): DType? = byWitness[kclass]

        /**
         * The [DType] whose [witness] is [kclass] — the inverse of [witness], e.g.
         * `DType.fromWitness(tensor.dtype)`.
         *
         * @throws IllegalArgumentException if [kclass] is not a dtype class
         */
        @JvmStatic
        public fun fromWitness(kclass: KClass<out DType>): DType =
            byWitness[kclass] ?: throw IllegalArgumentException("Not a DType witness: $kclass")

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

