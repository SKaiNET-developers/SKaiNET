@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Strategy interface for materializing tensor views into concrete tensors.
 * 
 * Materialization is the process of converting a tensor view (which references
 * data from a parent tensor through coordinate transformations) into a standalone
 * tensor with its own data storage. Different strategies provide different
 * trade-offs between memory usage, computation time, and access patterns.
 * 
 * ## Strategy Patterns
 * 
 * - **Immediate Materialization**: Creates a copy of the view data immediately
 * - **Lazy Materialization**: Defers copying until the data is actually accessed
 * - **Reference Materialization**: Maintains references while optimizing access
 * - **Batch Materialization**: Optimizes for materializing multiple views together
 * 
 * ## Usage Context
 * 
 * Materialization strategies are particularly important for:
 * - Performance optimization when views are accessed frequently
 * - Memory management when parent tensors can be garbage collected
 * - Computational efficiency when the view will be used in multiple operations
 * - Interoperability when passing tensors between different computational backends
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 */
public interface MaterializationStrategy<T : DType, V> {
    
    /**
     * Materializes the given tensor view into a concrete tensor.
     * 
     * This method performs the core materialization operation, transforming
     * a view (which references parent tensor data through coordinate mapping)
     * into a standalone tensor with its own data storage.
     * 
     * The implementation should:
     * - Preserve the view's shape and data type
     * - Copy or reference the appropriate data elements
     * - Maintain data integrity and type safety
     * - Apply any strategy-specific optimizations
     * 
     * @param view the TensorView to materialize
     * @return a new Tensor containing the materialized data
     * @throws IllegalStateException if the view cannot be materialized
     */
    public fun materialize(view: TensorView<T, V>): Tensor<T, V>
    
    /**
     * Checks if this strategy can materialize the given view.
     * 
     * Some materialization strategies may have constraints or requirements
     * that make them unsuitable for certain types of views. This method
     * allows the strategy to indicate whether it can handle a particular view.
     * 
     * Common constraints might include:
     * - View complexity (too many nested transformations)
     * - Memory availability (not enough space for immediate copying)
     * - Data type compatibility (specialized strategies for specific types)
     * - Backend requirements (GPU vs CPU accessibility)
     * 
     * @param view the TensorView to check
     * @return true if this strategy can materialize the view, false otherwise
     */
    public fun canMaterialize(view: TensorView<T, V>): Boolean
    
    /**
     * Estimates the memory overhead of materializing the given view.
     * 
     * This method provides insight into the memory cost of materialization,
     * helping with resource planning and strategy selection. The estimate
     * includes both the data storage and any additional metadata overhead.
     * 
     * The returned value represents:
     * - Positive numbers: additional memory required (in bytes)
     * - Zero: no additional memory required (e.g., reference-only strategies)
     * - Negative numbers: potential memory savings (e.g., when parent can be GC'd)
     * 
     * @param view the TensorView to analyze
     * @return estimated memory overhead in bytes
     */
    public fun estimateMemoryOverhead(view: TensorView<T, V>): Long
    
    /**
     * Gets a human-readable name for this materialization strategy.
     * 
     * This property provides a descriptive name that can be used for
     * logging, debugging, and strategy selection interfaces.
     * 
     * @return the strategy name (e.g., "CopyMaterialization", "LazyMaterialization")
     */
    public val name: String
}