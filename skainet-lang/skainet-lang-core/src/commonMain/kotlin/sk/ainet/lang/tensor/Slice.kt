package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Represents a slice descriptor for tensor slicing operations.
 * 
 * Slice is a sealed class hierarchy that describes different ways to select
 * elements along a single tensor dimension. It provides the building blocks
 * for complex tensor slicing operations while maintaining type safety and
 * enabling compile-time optimization opportunities.
 * 
 * ## Slice Types
 * 
 * The sealed class hierarchy includes:
 * - **Range**: Contiguous range of indices [start, end)
 * - **At**: Single specific index selection  
 * - **All**: Full dimension selection (equivalent to [:])
 * - **Step**: Strided access with start, end, and step size
 * 
 * ## Design Principles
 * 
 * - **Type Safety**: Compile-time guarantees about slice validity
 * - **Composability**: Slices can be combined and normalized
 * - **Performance**: Enable optimization based on slice patterns
 * - **Expressiveness**: Support common slicing use cases
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type
 */
public sealed class Slice<T : DType, V> {
    
    /**
     * Represents a contiguous range slice [start, end).
     * 
     * Range slices select a continuous sequence of elements along a dimension,
     * similar to Python's `tensor[start:end]` notation. This is the most common
     * and performance-friendly slice type.
     * 
     * ## Characteristics
     * - Always contiguous in memory
     * - Enables vectorized operations
     * - Efficient for batch processing
     * - Natural stride pattern (stride = 1)
     * 
     * @param start the starting index (inclusive)
     * @param end the ending index (exclusive)
     */
    public data class Range<T : DType, V>(val start: Int, val end: Int) : Slice<T, V>() {
        init {
            require(start >= 0) { "Range start must be non-negative: $start" }
            require(end >= start) { "Range end must be >= start: start=$start, end=$end" }
        }
    }
    
    /**
     * Represents selection of a single specific index.
     * 
     * At slices reduce dimensionality by selecting a single element along
     * a dimension, similar to Python's `tensor[index]` notation. This results
     * in a view with one fewer dimension than the parent.
     * 
     * ## Characteristics
     * - Reduces tensor rank by 1
     * - Point access (no range)
     * - Efficient for layer/channel extraction
     * - Zero stride along selected dimension
     * 
     * @param index the specific index to select
     */
    public data class At<T : DType, V>(val index: Int) : Slice<T, V>() {
        init {
            require(index >= 0) { "At index must be non-negative: $index" }
        }
    }
    
    /**
     * Represents selection of the entire dimension.
     * 
     * All slices select every element along a dimension, equivalent to
     * Python's `tensor[:]` notation. This is useful for explicit specification
     * of unchanged dimensions in multi-dimensional slicing operations.
     * 
     * ## Characteristics
     * - Preserves dimension size
     * - No coordinate transformation needed
     * - Identity mapping for the dimension
     * - Natural stride pattern
     * 
     */
    public class All<T : DType, V> : Slice<T, V>() {
        override fun equals(other: Any?): Boolean = other is All<*, *>
        override fun hashCode(): Int = All::class.hashCode()
        override fun toString(): String = "All"
    }
    
    /**
     * Represents strided access with custom step size.
     * 
     * Step slices select elements at regular intervals along a dimension,
     * similar to Python's `tensor[start:end:step]` notation. This enables
     * efficient subsampling and stride-based operations.
     * 
     * ## Characteristics
     * - Configurable sampling interval
     * - May create non-contiguous patterns
     * - Useful for downsampling operations
     * - Custom stride calculation required
     * 
     * @param start the starting index (inclusive)
     * @param end the ending index (exclusive)  
     * @param step the step size (must be positive)
     */
    public data class Step<T : DType, V>(val start: Int, val end: Int, val step: Int) : Slice<T, V>() {
        init {
            require(start >= 0) { "Step start must be non-negative: $start" }
            require(end >= start) { "Step end must be >= start: start=$start, end=$end" }
            require(step > 0) { "Step size must be positive: $step" }
        }
    }
    
    /**
     * Validates that this slice is compatible with the given dimension size.
     * 
     * This method checks that all indices referenced by the slice are within
     * the valid bounds for a dimension of the specified size. Different slice
     * types have different validation requirements.
     * 
     * @param dimensionSize the size of the dimension this slice will be applied to
     * @return true if the slice is valid for the given dimension size
     */
    public fun isValid(dimensionSize: Int): Boolean {
        require(dimensionSize >= 0 || Dim.isDynamic(dimensionSize)) {
            "Dimension size must be non-negative or dynamic: $dimensionSize"
        }

        // Dynamic (unknown-extent) axis: `all()` is the symbolic full axis and is always valid; a partial
        // Range/At/Step can't be bounds-checked against an unknown size, so only its own self-consistent,
        // non-negative (never from-end) bounds are validated.
        if (Dim.isDynamic(dimensionSize)) {
            return when (this) {
                is Range -> start >= 0 && end >= start
                is At -> index >= 0
                is All -> true
                is Step -> start >= 0 && end >= start && step > 0
            }
        }

        return when (this) {
            is Range -> start < dimensionSize && end <= dimensionSize
            is At -> index < dimensionSize
            is All -> true // Always valid
            is Step -> start < dimensionSize && end <= dimensionSize
        }
    }
    
    /**
     * Determines if this slice represents contiguous memory access.
     * 
     * Contiguous slices can be optimized with vectorized operations and
     * bulk memory transfers. Non-contiguous slices require element-by-element
     * processing but enable more flexible access patterns.
     * 
     * @return true if the slice represents contiguous access
     */
    public fun isContiguous(): Boolean = when (this) {
        is Range -> true
        is At -> true  
        is All -> true
        is Step -> step == 1
    }
    
    /**
     * Checks if this slice uses non-trivial stride patterns.
     * 
     * Non-trivial strides require additional computation during coordinate
     * mapping and may impact cache performance. This information helps
     * optimize access pattern decisions.
     * 
     * @return true if the slice requires non-unit stride calculations
     */
    public fun hasNonTrivialStride(): Boolean = when (this) {
        is Range -> false
        is At -> false
        is All -> false  
        is Step -> step != 1
    }
    
    /**
     * Determines if this slice selects zero elements.
     * 
     * Empty slices can occur with invalid ranges or edge cases and should
     * be handled specially to avoid unnecessary computation or memory allocation.
     * 
     * @return true if the slice selects no elements
     */
    public fun isEmpty(): Boolean = when (this) {
        is Range -> start >= end
        is At -> false
        is All -> false
        is Step -> start >= end
    }
    
    /**
     * Calculates the number of elements this slice will select.
     * 
     * This method computes the resulting dimension size after applying
     * the slice, which is essential for calculating view shapes.
     * 
     * @param dimensionSize the original size of the dimension
     * @return the number of elements selected by this slice
     */
    public fun getResultSize(dimensionSize: Int): Int {
        require(isValid(dimensionSize)) { "Invalid slice for dimension size $dimensionSize: $this" }
        
        return when (this) {
            is Range -> maxOf(0, end - start)
            is At -> 0 // At slices reduce dimensionality
            is All -> dimensionSize
            is Step -> maxOf(0, (end - start + step - 1) / step)
        }
    }
    
    /**
     * Normalizes this slice against a specific dimension size.
     * 
     * Normalization resolves negative indices, validates bounds, and
     * converts slice specifications into canonical form for efficient
     * processing.
     * 
     * @param dimensionSize the size of the dimension to normalize against
     * @return a normalized slice equivalent to this slice
     */
    public fun normalize(dimensionSize: Int): Slice<T, V> {
        require(dimensionSize >= 0 || Dim.isDynamic(dimensionSize)) {
            "Dimension size must be non-negative or dynamic: $dimensionSize"
        }

        // A dynamic axis has no known size to resolve from-end (negative) indices against, so those are
        // rejected; the full-axis `all()` stays symbolic and explicit non-negative bounds pass through.
        if (Dim.isDynamic(dimensionSize)) {
            return when (this) {
                is All -> this
                is Range -> { require(start >= 0 && end >= 0) { "dynamic-axis Range needs non-negative bounds: $this" }; this }
                is At -> { require(index >= 0) { "dynamic-axis At needs a non-negative index: $this" }; this }
                is Step -> { require(start >= 0 && end >= 0) { "dynamic-axis Step needs non-negative bounds: $this" }; this }
            }
        }

        return when (this) {
            is Range -> {
                val normStart = if (start < 0) maxOf(0, dimensionSize + start) else minOf(start, dimensionSize)
                val normEnd = if (end < 0) maxOf(0, dimensionSize + end) else minOf(end, dimensionSize)
                Range(normStart, normEnd)
            }
            is At -> {
                val normIndex = if (index < 0) dimensionSize + index else index
                require(normIndex in 0 until dimensionSize) { "Index out of bounds: $normIndex for dimension size $dimensionSize" }
                At(normIndex)
            }
            is All -> this
            is Step -> {
                val normStart = if (start < 0) maxOf(0, dimensionSize + start) else minOf(start, dimensionSize)
                val normEnd = if (end < 0) maxOf(0, dimensionSize + end) else minOf(end, dimensionSize)
                Step(normStart, normEnd, step)
            }
        }
    }
}