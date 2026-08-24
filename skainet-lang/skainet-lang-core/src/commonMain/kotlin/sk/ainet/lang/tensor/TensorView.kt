@file:Suppress("DEPRECATION") // this file *is* the deprecated mechanism

package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * A tensor view interface that represents a zero-copy slice or subset of a parent tensor.
 * 
 * TensorView extends the base Tensor interface while adding view-specific functionality
 * for efficient tensor slicing operations. Views provide memory-efficient access to
 * portions of larger tensors without copying the underlying data.
 * 
 * ## Key Characteristics
 * 
 * - **Zero-Copy**: Views reference the parent tensor's data without duplication
 * - **Shape Transformation**: Views can have different shapes than their parent
 * - **Index Mapping**: Coordinate transformation between view space and parent space
 * - **Memory Efficiency**: Multiple views can reference the same underlying data
 * 
 * ## Usage Patterns
 * 
 * Views are particularly useful for:
 * - Batch processing (extracting mini-batches from larger datasets)
 * - Channel extraction (accessing specific feature maps in NCHW tensors)
 * - Spatial regions (selecting rectangular areas in image tensors)
 * - Sequence windows (sliding window operations on temporal data)
 * 
 * @param T the data type constraint extending DType, defining the numerical precision
 * @param V the actual value type that will be stored and accessed
 */
@Deprecated(
    message = "One view mechanism (SKEEP-003 §4.6, #1034): a view is a `sk.ainet.lang.memory.TensorView` — " +
        "Shape + Format + Layout + Storage over the parent's bytes — built with `tensor.view()` and reshaped " +
        "with `narrow`/`step`/`transpose`/`unsqueeze`/`squeeze`/`slice(slices)`. No ReplaceWith: the replacement " +
        "is not type-parameterized, so an automatic fix would produce code that does not compile. This interface " +
        "and its implementations keep working until the next major.",
)
public interface TensorView<T : DType, V> : Tensor<T, V> {
    
    /**
     * The parent tensor that this view references.
     * 
     * This property provides access to the original tensor from which this view
     * was created. The parent tensor contains the actual data storage, while
     * the view provides a transformed interface to access portions of that data.
     * 
     * Views maintain a reference to their parent to enable:
     * - Data access through coordinate transformation
     * - Memory lifecycle management
     * - Composition of multiple view layers
     * 
     * @return the parent Tensor<T, V> that contains the underlying data
     */
    public val parentTensor: Tensor<T, V>
    
    /**
     * The shape of this view, which may differ from the parent tensor's shape.
     * 
     * The view shape represents the dimensions and size as seen from the view's
     * perspective, after applying slicing transformations. This shape determines:
     * - The valid coordinate ranges for element access
     * - The iteration bounds for view operations
     * - The compatibility for tensor operations involving this view
     * 
     * Note: This property shadows the inherited shape property from Tensor,
     * providing view-specific shape information rather than delegating to
     * the underlying data component.
     * 
     * @return the Shape describing this view's dimensional structure
     */
    public val viewShape: Shape
    
    /**
     * The index mapper responsible for coordinate transformation between view and parent.
     * 
     * The index mapper handles the complex task of translating coordinates from
     * the view's coordinate system to the parent tensor's coordinate system.
     * This enables efficient element access while maintaining the illusion of
     * a contiguous, reshaped tensor.
     * 
     * The mapper encapsulates:
     * - Coordinate system transformation algorithms
     * - Stride calculations for memory access patterns
     * - Optimization hints for contiguous access detection
     * - Bounds checking and validation logic
     * 
     * @return the IndexMapper that handles coordinate transformations
     */
    public val indexMapping: IndexMapper
    
    /**
     * Override the shape property to return the view-specific shape.
     * 
     * This override ensures that operations on the view use the view's shape
     * rather than the parent tensor's shape, maintaining consistency with
     * the view's logical structure.
     */
    override val shape: Shape
        get() = viewShape
}