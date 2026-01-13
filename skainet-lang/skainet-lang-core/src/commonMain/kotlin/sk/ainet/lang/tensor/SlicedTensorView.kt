package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * A concrete implementation of TensorView that provides zero-copy sliced access to parent tensors.
 * 
 * SlicedTensorView creates a view of a parent tensor by applying a list of slice operations,
 * one for each dimension. The view appears as a regular tensor but delegates all data access
 * through coordinate transformation to the parent tensor's storage.
 * 
 * ## Key Features
 * 
 * - **Zero-Copy Operations**: No data duplication, only metadata storage
 * - **Lazy Evaluation**: Shape and index mapping computed on construction
 * - **Efficient Access**: Optimized coordinate transformation for common patterns
 * - **Type Safety**: Maintains type constraints from parent tensor
 * 
 * ## Memory Layout Considerations
 * 
 * The view maintains minimal memory overhead:
 * - Reference to parent tensor
 * - Computed view shape
 * - Index mapper instance
 * - Slice list (for potential reuse/optimization)
 * 
 * ## Performance Characteristics
 * 
 * Access performance depends on slice patterns:
 * - Contiguous slices: Near-native performance
 * - Strided slices: Moderate overhead from coordinate calculation  
 * - Complex patterns: Higher overhead but still zero-copy
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type
 * @param parentTensor the parent tensor to create a view of
 * @param slices list of slice operations, one per dimension
 */
public class SlicedTensorView<T : DType, V>(
    override val parentTensor: Tensor<T, V>,
    private val slices: List<Slice<T, V>>
) : TensorView<T, V> {
    override val gradState: GradState<T, V> = parentTensor.gradState
    
    init {
        require(slices.size == parentTensor.rank) {
            "Number of slices (${slices.size}) must match parent tensor rank (${parentTensor.rank})"
        }
        
        // Validate all slices against parent dimensions
        slices.forEachIndexed { dim, slice ->
            val dimensionSize = parentTensor.shape[dim]
            require(slice.isValid(dimensionSize)) {
                "Invalid slice for dimension $dim (size $dimensionSize): $slice"
            }
        }
    }
    
    /**
     * The computed shape of this view after applying all slice operations.
     * 
     * This shape is calculated by applying each slice to its corresponding
     * parent dimension and collecting the results. At slices reduce dimensionality,
     * while other slice types preserve or modify dimension sizes.
     */
    override val viewShape: Shape by lazy {
        computeSlicedShape()
    }
    
    /**
     * The index mapper for coordinate transformation between view and parent space.
     * 
     * The mapper is created based on the slice patterns and parent tensor
     * characteristics. Different mapper implementations may be selected for
     * optimization based on access patterns.
     */
    override val indexMapping: IndexMapper by lazy {
        createIndexMapper()
    }
    
    /**
     * Delegate data access to a wrapper that handles coordinate transformation.
     * 
     * This creates a TensorData wrapper that intercepts get/set operations
     * and applies coordinate mapping before delegating to parent tensor data.
     */
    override val data: TensorData<T, V> by lazy {
        SlicedTensorData(parentTensor.data, viewShape, indexMapping)
    }
    
    /**
     * Inherit operations component from parent tensor.
     * 
     * The operations component can be reused since it operates on the same
     * value type. Operations will see the view's shape and delegate data
     * access through the view's data wrapper.
     */
    override val ops: TensorOps = parentTensor.ops
    
    /**
     * Inherit data type from parent tensor.
     * 
     * The data type remains unchanged as views don't transform value types,
     * only coordinate access patterns.
     */
    override val dtype: KClass<T> = parentTensor.dtype
    
    /**
     * The slice operations applied to create this view.
     * 
     * This property exposes the slice configuration for composition operations,
     * debugging, and optimization analysis. Each slice corresponds to a
     * dimension in the parent tensor.
     */
    public val sliceOperations: List<Slice<T, V>> get() = slices
    
    /**
     * Computes the resulting shape after applying all slice operations.
     * 
     * This method processes each slice against its corresponding parent dimension,
     * calculating the resulting dimension size and handling dimensionality
     * reduction for At slices.
     * 
     * ## Algorithm
     * 
     * 1. Iterate through each slice-dimension pair
     * 2. Calculate resulting dimension size using slice.getResultSize()
     * 3. Skip dimensions that are eliminated by At slices
     * 4. Build final shape from remaining dimensions
     * 
     * @return the computed Shape for this view
     */
    private fun computeSlicedShape(): Shape {
        val resultDimensions = mutableListOf<Int>()
        
        slices.forEachIndexed { dim, slice ->
            val parentDimSize = parentTensor.shape[dim]
            val resultSize = slice.getResultSize(parentDimSize)
            
            // At slices reduce dimensionality (resultSize = 0)
            // Other slices preserve or modify dimension size
            if (resultSize > 0 || slice !is Slice.At) {
                if (slice is Slice.At) {
                    // At slices eliminate the dimension entirely
                } else {
                    resultDimensions.add(resultSize)
                }
            }
        }
        
        return Shape(resultDimensions.toIntArray())
    }
    
    /**
     * Creates an appropriate IndexMapper based on slice patterns and parent characteristics.
     * 
     * This method analyzes the slice patterns and parent tensor characteristics
     * to select the most efficient mapper implementation. Different mappers
     * can be optimized for specific memory layouts or access patterns.
     * 
     * ## Mapper Selection Strategy
     * 
     * - **NCHWIndexMapper**: For 4D tensors with NCHW-optimized access
     * - **SliceIndexMapper**: General-purpose mapper for arbitrary patterns
     * 
     * Future optimizations could include:
     * - Contiguous-only mappers for simple range slices
     * - Specialized mappers for common ML access patterns
     * - Hardware-specific optimizations
     * 
     * @return an IndexMapper optimized for this view's access patterns
     */
    private fun createIndexMapper(): IndexMapper {
        // For now, use a general SliceIndexMapper
        // Future optimization: select specialized mappers based on patterns
        return SliceIndexMapper(parentTensor.shape, slices, viewShape)
    }
    
    /**
     * TensorData wrapper that applies coordinate transformation for sliced access.
     * 
     * This internal class intercepts TensorData operations and applies the
     * index mapping before delegating to the parent tensor's data component.
     * It maintains the TensorData interface contract while providing transparent
     * coordinate transformation.
     */
    private inner class SlicedTensorData(
        private val parentData: TensorData<T, V>,
        override val shape: Shape,
        private val mapper: IndexMapper
    ) : TensorData<T, V> {
        
        /**
         * Get element with coordinate transformation.
         * 
         * This method:
         * 1. Validates view coordinates are within bounds
         * 2. Maps view coordinates to parent coordinates
         * 3. Delegates to parent data for actual element access
         * 
         * @param indices coordinates in view space
         * @return the element value at the transformed parent coordinates
         */
        override fun get(vararg indices: Int): V {
            require(indices.size == shape.rank) {
                "Expected ${shape.rank} indices but got ${indices.size}"
            }
            
            // Validate bounds in view space
            indices.forEachIndexed { dim, index ->
                require(index >= 0 && index < shape[dim]) {
                    "Index $index out of bounds for dimension $dim (size ${shape[dim]})"
                }
            }
            
            // Map to parent coordinates and delegate
            val parentIndices = mapper.mapToParent(indices)
            return parentData.get(*parentIndices)
        }
        
        /**
         * Set element with coordinate transformation.
         * 
         * This method follows the same pattern as get() but delegates
         * to the parent data's set operation after coordinate transformation.
         * 
         * @param indices coordinates in view space
         * @param value the value to set at the transformed parent coordinates
         */
        override fun set(vararg indices: Int, value: V) {
            require(indices.size == shape.rank) {
                "Expected ${shape.rank} indices but got ${indices.size}"
            }
            
            // Validate bounds in view space
            indices.forEachIndexed { dim, index ->
                require(index >= 0 && index < shape[dim]) {
                    "Index $index out of bounds for dimension $dim (size ${shape[dim]})"
                }
            }
            
            // Map to parent coordinates and delegate
            val parentIndices = mapper.mapToParent(indices)
            when (parentIndices.size) {
                1 -> parentData[parentIndices[0]] = value
                2 -> parentData[parentIndices[0], parentIndices[1]] = value
                3 -> parentData[parentIndices[0], parentIndices[1], parentIndices[2]] = value
                4 -> parentData[parentIndices[0], parentIndices[1], parentIndices[2], parentIndices[3]] = value
                else -> {
                    // For higher dimensions, use reflection or manual call
                    // This is a temporary solution - ideally we'd support arbitrary dimensions
                    val call = StringBuilder()
                    call.append("parentData[")
                    parentIndices.forEachIndexed { i, idx ->
                        if (i > 0) call.append(", ")
                        call.append(idx)
                    }
                    call.append("] = value")
                    throw UnsupportedOperationException("Tensor dimensions > 4 not yet supported in set operation. Need: $call")
                }
            }
        }
    }
}
