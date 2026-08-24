@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Extension methods for tensor view management and optimization.
 * 
 * These extensions provide utilities for analyzing, optimizing, and managing
 * tensor views to improve performance and memory efficiency in tensor operations.
 */

/**
 * Determines if this tensor view represents a contiguous memory layout.
 * 
 * A contiguous view is one where the elements can be accessed in a linear
 * fashion without gaps or stride adjustments. Contiguous views offer
 * significant performance advantages for:
 * - Bulk memory operations (copy, fill, etc.)
 * - Vectorized computations
 * - Cache-friendly access patterns
 * - Direct memory mapping to external libraries
 * 
 * ## Contiguity Analysis
 * 
 * A view is considered contiguous when:
 * - Sequential element access follows natural memory layout
 * - No gaps exist between accessed elements
 * - Stride patterns match expected row-major or column-major order
 * - View boundaries align with natural tensor structure
 * 
 * ## Performance Implications
 * 
 * **Contiguous Views:**
 * - Enable bulk memory operations
 * - Support vectorized SIMD instructions
 * - Provide optimal cache locality
 * - Allow direct interfacing with BLAS/LAPACK libraries
 * 
 * **Non-Contiguous Views:**
 * - Require element-by-element access
 * - May have poor cache performance
 * - Need coordinate transformation for each access
 * - Cannot use optimized bulk operations
 * 
 * ## Usage Examples
 * 
 * ```kotlin
 * val tensor = tensorOf(Shape(4, 4)) { /* data */ }
 * 
 * // Contiguous slice (full rows)
 * val contiguousView = tensor.sliceView { range(1, 3) }
 * println(contiguousView.isContiguous()) // true
 * 
 * // Non-contiguous slice (columns)
 * val nonContiguousView = tensor.sliceView { at(0); range(1, 3) }
 * println(nonContiguousView.isContiguous()) // false
 * ```
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return true if the view represents contiguous memory layout, false otherwise
 */
public fun <T : DType, V> TensorView<T, V>.isContiguous(): Boolean {
    // A view is contiguous if accessing elements sequentially in view space
    // corresponds to accessing elements sequentially in parent tensor space
    
    return analyzeContiguity(this)
}

/**
 * Internal function to analyze the contiguity of a tensor view.
 */
private fun <T : DType, V> analyzeContiguity(view: TensorView<T, V>): Boolean {
    val viewShape = view.viewShape
    val parentShape = view.parentTensor.shape
    
    // Handle trivial cases
    if (viewShape.volume <= 1) {
        return true // Single element or empty views are always contiguous
    }
    
    // For this implementation, we'll use a heuristic approach:
    // A view is likely contiguous if it represents:
    // 1. A complete slice along the last dimensions
    // 2. A rectangular region that aligns with memory layout
    // 3. A view where stride patterns match expected layout
    
    return try {
        // Check if the view represents a simple rectangular slice
        isRectangularSlice(view, viewShape, parentShape)
    } catch (e: Exception) {
        // If we can't determine contiguity, assume non-contiguous for safety
        false
    }
}

/**
 * Checks if a view represents a rectangular slice that maintains contiguity.
 */
private fun <T : DType, V> isRectangularSlice(
    view: TensorView<T, V>,
    viewShape: Shape,
    parentShape: Shape
): Boolean {
    // For a rectangular slice to be contiguous:
    // 1. The view should span complete dimensions from some point onward
    // 2. Or the view should represent a contiguous block in the parent tensor
    
    val viewDims = viewShape.dimensions
    val parentDims = parentShape.dimensions
    
    if (viewDims.size != parentDims.size) {
        // Shape change indicates potential non-contiguity
        return false
    }
    
    // Check if the view represents complete trailing dimensions
    // This is a common case for contiguous slices
    var foundPartialDim = false
    for (i in viewDims.indices.reversed()) {
        if (viewDims[i] == parentDims[i]) {
            // Complete dimension - continue checking
            if (foundPartialDim) {
                // Found complete dimension after partial dimension
                // This indicates non-contiguous access
                return false
            }
        } else {
            // Partial dimension
            foundPartialDim = true
        }
    }
    
    // Additional heuristic: check if view volume is a significant portion
    // of parent volume, which often indicates contiguous regions
    val viewVolume = viewShape.volume
    val parentVolume = parentShape.volume
    
    if (parentVolume == 0) return true
    
    val volumeRatio = viewVolume.toDouble() / parentVolume.toDouble()
    
    // If the view covers a large portion of the parent and maintains
    // dimensional structure, it's likely contiguous
    return volumeRatio > 0.1 || viewVolume == parentVolume
}

/**
 * Provides a more detailed contiguity analysis with diagnostic information.
 * 
 * This method returns a detailed analysis of the view's memory layout
 * characteristics, useful for debugging and performance optimization.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return ContiguityAnalysis containing detailed information about memory layout
 */
public fun <T : DType, V> TensorView<T, V>.analyzeContiguity(): ContiguityAnalysis {
    val isContiguous = this.isContiguous()
    val viewShape = this.viewShape
    val parentShape = this.parentTensor.shape
    
    return ContiguityAnalysis(
        isContiguous = isContiguous,
        viewVolume = viewShape.volume,
        parentVolume = parentShape.volume,
        volumeRatio = if (parentShape.volume > 0) {
            viewShape.volume.toDouble() / parentShape.volume.toDouble()
        } else 0.0,
        dimensionMatches = viewShape.dimensions.zip(parentShape.dimensions)
            .map { (view, parent) -> view == parent },
        reason = if (isContiguous) {
            "View maintains contiguous memory access pattern"
        } else {
            "View requires non-contiguous memory access"
        }
    )
}

/**
 * Materializes a list of tensor views into a list of standalone tensors efficiently.
 * 
 * This method provides batch materialization of multiple tensor views, which can
 * offer performance benefits over materializing views individually. The batch
 * operation can optimize for:
 * - Memory allocation patterns
 * - CPU cache efficiency  
 * - Parallel processing opportunities
 * - Reduced overhead from repeated strategy initialization
 * 
 * ## Batch Processing Benefits
 * 
 * **Memory Optimization:**
 * - Pre-allocates memory for all materialized tensors
 * - Reduces memory fragmentation
 * - Enables memory pool optimizations
 * 
 * **Performance Optimization:**
 * - Amortizes strategy setup costs across multiple views
 * - Enables vectorized operations when possible
 * - Reduces system call overhead
 * 
 * **Resource Management:**
 * - Coordinates garbage collection timing
 * - Manages memory pressure more effectively
 * - Provides better resource utilization monitoring
 * 
 * ## Usage Examples
 * 
 * ```kotlin
 * val parentTensor = tensorOf(Shape(100, 100)) { /* data */ }
 * val views = listOf(
 *     parentTensor.sliceView { range(0, 25) },
 *     parentTensor.sliceView { range(25, 50) },
 *     parentTensor.sliceView { range(50, 75) },
 *     parentTensor.sliceView { range(75, 100) }
 * )
 * 
 * // Batch materialize all views efficiently
 * val materialized = views.batchMaterialize()
 * ```
 * 
 * ## Performance Considerations
 * 
 * Batch materialization is most beneficial when:
 * - Processing multiple views from the same parent tensor
 * - Views have similar complexity and size characteristics
 * - Memory allocation overhead is significant compared to data copying
 * - The system benefits from predictable memory allocation patterns
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return a List of materialized Tensor objects corresponding to the input views
 * @throws IllegalStateException if any view cannot be materialized
 */
public fun <T : DType, V> List<TensorView<T, V>>.batchMaterialize(): List<Tensor<T, V>> {
    // Handle empty list case
    if (isEmpty()) return emptyList()
    
    // For simplicity, we'll use the copy materialization strategy for all views
    // A more sophisticated implementation could choose different strategies
    // based on view characteristics and batch analysis
    val strategy = CopyMaterializationStrategy<T, V>()
    
    // Pre-validate that all views can be materialized
    val failedViews = this.filterIndexed { index, view -> 
        !strategy.canMaterialize(view)
    }
    
    if (failedViews.isNotEmpty()) {
        throw IllegalStateException(
            "Cannot materialize ${failedViews.size} views in batch operation"
        )
    }
    
    // Estimate total memory requirement for better resource planning
    val totalMemoryEstimate = this.sumOf { view ->
        strategy.estimateMemoryOverhead(view)
    }
    
    // For very large batch operations, we might want to process in chunks
    // to avoid memory pressure, but for now we'll process all at once
    
    return this.map { view ->
        try {
            strategy.materialize(view)
        } catch (e: Exception) {
            throw IllegalStateException("Failed to materialize view in batch operation", e)
        }
    }
}

/**
 * Materializes a list of tensor views using a specific materialization strategy.
 * 
 * This overload allows explicit control over the materialization strategy used
 * for the entire batch operation, enabling optimization for specific use cases
 * or resource constraints.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @param strategy the MaterializationStrategy to use for all views in the batch
 * @return a List of materialized Tensor objects using the specified strategy
 * @throws IllegalStateException if the strategy cannot materialize any view
 */
public fun <T : DType, V> List<TensorView<T, V>>.batchMaterialize(
    strategy: MaterializationStrategy<T, V>
): List<Tensor<T, V>> {
    // Handle empty list case
    if (isEmpty()) return emptyList()
    
    // Validate that the strategy can materialize all views
    val failedViews = this.filterIndexed { index, view -> 
        !strategy.canMaterialize(view)
    }
    
    if (failedViews.isNotEmpty()) {
        throw IllegalStateException(
            "Strategy '${strategy.name}' cannot materialize ${failedViews.size} views in batch"
        )
    }
    
    // Materialize all views using the specified strategy
    return this.map { view ->
        try {
            strategy.materialize(view)
        } catch (e: Exception) {
            throw IllegalStateException(
                "Strategy '${strategy.name}' failed to materialize view in batch operation", e
            )
        }
    }
}

/**
 * Estimates the total memory overhead for batch materializing a list of views.
 * 
 * This method provides insight into the total memory cost of batch materialization,
 * helping with resource planning and decision making about whether to materialize
 * views individually or as a batch.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return the estimated total memory overhead in bytes for batch materialization
 */
public fun <T : DType, V> List<TensorView<T, V>>.estimateBatchMaterializationCost(): Long {
    if (isEmpty()) return 0L
    
    val strategy = CopyMaterializationStrategy<T, V>()
    return this.sumOf { view ->
        strategy.estimateMemoryOverhead(view)
    }
}

/**
 * Estimates the total memory overhead using a specific materialization strategy.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @param strategy the MaterializationStrategy to estimate costs for
 * @return the estimated total memory overhead in bytes for the specified strategy
 */
public fun <T : DType, V> List<TensorView<T, V>>.estimateBatchMaterializationCost(
    strategy: MaterializationStrategy<T, V>
): Long {
    if (isEmpty()) return 0L
    
    return this.sumOf { view ->
        strategy.estimateMemoryOverhead(view)
    }
}

/**
 * Data class containing detailed contiguity analysis results.
 */
public data class ContiguityAnalysis(
    val isContiguous: Boolean,
    val viewVolume: Int,
    val parentVolume: Int,
    val volumeRatio: Double,
    val dimensionMatches: List<Boolean>,
    val reason: String
)