package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.storage.ActiveMemoryTracker
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import kotlin.reflect.KClass
import sk.ainet.lang.tensor.GradState

/**
 * A materialization strategy that immediately copies all data from a tensor view
 * into a new, standalone tensor with contiguous memory layout.
 *
 * This strategy provides immediate materialization by iterating through all
 * elements in the view and copying them to a new tensor with the view's shape.
 * The resulting tensor is completely independent of the parent tensor and can
 * be used even after the parent tensor is garbage collected.
 *
 * ## Characteristics
 *
 * - **Immediate Execution**: Materialization happens synchronously when called
 * - **Memory Independent**: Result tensor has no dependencies on parent tensor
 * - **Contiguous Layout**: Output data is stored in standard row-major order
 * - **Type Preservation**: Maintains the same data type and value type as the view
 *
 * ## Trade-offs
 *
 * **Benefits:**
 * - Predictable memory usage and performance
 * - No ongoing computational overhead for element access
 * - Enables garbage collection of parent tensors
 * - Compatible with all downstream operations
 *
 * **Costs:**
 * - Immediate memory allocation for full tensor size
 * - Computational cost of copying all elements
 * - Temporary memory pressure during materialization
 *
 * ## Usage Scenarios
 *
 * This strategy is optimal when:
 * - The materialized tensor will be accessed frequently
 * - Memory usage is predictable and acceptable
 * - The parent tensor can be released after materialization
 * - Compatibility with external libraries is required
 *
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 */
public class CopyMaterializationStrategy<T : DType, V> : MaterializationStrategy<T, V> {

    override val name: String = "CopyMaterialization"

    override fun materialize(view: TensorView<T, V>): Tensor<T, V> {
        val viewShape = view.viewShape
        val viewVolume = viewShape.volume

        // Create a new data array to hold the materialized elements
        val materializedData = createDataArray(view, viewVolume)

        // Copy all elements from the view to the new array
        copyViewElements(view, materializedData, viewShape)

        // Record the copy for memory tracking
        ActiveMemoryTracker.recordCopy("CopyMaterializationStrategy", viewVolume.toLong() * 4)

        // Create and return the materialized tensor
        return createMaterializedTensor(view, materializedData, viewShape)
    }

    override fun canMaterialize(view: TensorView<T, V>): Boolean {
        // CopyMaterializationStrategy can handle any view as long as:
        // 1. The view has a valid shape
        // 2. Memory is available for allocation
        return try {
            view.viewShape.volume >= 0
        } catch (e: Exception) {
            false
        }
    }

    override fun estimateMemoryOverhead(view: TensorView<T, V>): Long {
        // Estimate memory required for a copy of the view data
        val viewVolume = view.viewShape.volume
        //val bytesPerElement = estimateBytesPerElement(view.dtype)
        return viewVolume.toLong() * 4 // bytesPerElement
    }

    /**
     * Creates a data array suitable for storing the materialized view elements.
     *
     * This method needs to create an appropriate array type based on the
     * tensor's value type. Since we don't have direct access to the tensor
     * factory here, we'll need to work with the existing data structure.
     */
    @Suppress("UNCHECKED_CAST")
    private fun createDataArray(view: TensorView<T, V>, volume: Int): Array<V?> {
        return arrayOfNulls<Any>(volume) as Array<V?>
    }

    /**
     * Copies all elements from the tensor view to the materialized data array.
     *
     * This method iterates through the view's coordinate space and copies
     * each element to the corresponding position in the output array using
     * row-major order.
     */
    private fun copyViewElements(view: TensorView<T, V>, data: Array<V?>, shape: Shape) {
        val dimensions = shape.dimensions
        val indices = IntArray(dimensions.size)

        fun copyRecursive(dimension: Int, flatIndex: Int): Int {
            var currentIndex = flatIndex

            if (dimension == dimensions.size) {
                // Base case: copy the element at this coordinate
                val element = view.data.get(*indices)
                data[currentIndex] = element
                return currentIndex + 1
            }

            // Recursive case: iterate through this dimension
            for (i in 0 until dimensions[dimension]) {
                indices[dimension] = i
                currentIndex = copyRecursive(dimension + 1, currentIndex)
            }

            return currentIndex
        }

        copyRecursive(0, 0)
    }

    /**
     * Creates a materialized tensor from the copied data.
     *
     * This method constructs a new Tensor instance using the copied data
     * and the view's shape and data type information.
     */
    private fun createMaterializedTensor(
        view: TensorView<T, V>,
        data: Array<V?>,
        shape: Shape
    ): Tensor<T, V> {
        // Create a simple tensor implementation that wraps our materialized data
        return MaterializedTensor(
            data = MaterializedTensorData<T, V>(shape, data),
            ops = view.ops,
            dtype = view.dtype,
            gradState = view.gradState
        )
    }

    /**
     * Estimates the number of bytes per element for the given data type.
     */
    private fun estimateBytesPerElement(dtype: KClass<DType>): Int {
        return when (dtype) {
            FP32 -> 4
            FP16 -> 2
            Int32 -> 4
            Int8 -> 1
            Int4 -> 1 // Packed, but estimate 1 byte for simplicity
            Ternary -> 1 // Packed, but estimate 1 byte for simplicity
            else -> 4 // Default to 4 bytes
        }
    }

    /**
     * Simple tensor data implementation for materialized tensors.
     */
    private class MaterializedTensorData<T : DType, V>(
        override val shape: Shape,
        private val data: Array<V?>
    ) : TensorData<T, V> {

        override fun get(vararg indices: Int): V {
            val flatIndex = calculateFlatIndex(indices)
            return data[flatIndex] ?: throw IllegalStateException("Null data at index $flatIndex")
        }

        override fun set(vararg indices: Int, value: V) {
            val flatIndex = calculateFlatIndex(indices)
            data[flatIndex] = value
        }

        private fun calculateFlatIndex(indices: IntArray): Int {
            require(indices.size == shape.dimensions.size) {
                "Expected ${shape.dimensions.size} indices, got ${indices.size}"
            }

            var flatIndex = 0
            var stride = 1

            // Calculate flat index using row-major order
            for (i in shape.dimensions.size - 1 downTo 0) {
                require(indices[i] >= 0 && indices[i] < shape.dimensions[i]) {
                    "Index ${indices[i]} out of bounds for dimension $i with size ${shape.dimensions[i]}"
                }
                flatIndex += indices[i] * stride
                stride *= shape.dimensions[i]
            }

            return flatIndex
        }
    }

    /**
     * Simple tensor implementation for materialized tensors.
     */
    private class MaterializedTensor<T : DType, V>(
        override val data: TensorData<T, V>,
        override val ops: TensorOps,
        override val dtype: KClass<T>,
        override val gradState: GradState<T, V>
    ) : Tensor<T, V>
}
