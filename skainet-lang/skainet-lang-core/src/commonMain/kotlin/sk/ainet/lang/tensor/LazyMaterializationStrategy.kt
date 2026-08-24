@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * A materialization strategy that defers copying of tensor view data until
 * individual elements are actually accessed.
 *
 * This strategy creates a materialized tensor that maintains a reference to
 * the original view but caches accessed elements in a sparse data structure.
 * Elements are copied from the view only when first accessed, providing
 * memory efficiency for cases where only a subset of the tensor data is used.
 *
 * ## Characteristics
 *
 * - **Deferred Execution**: Elements are copied only when accessed
 * - **Sparse Caching**: Only accessed elements consume additional memory
 * - **Lazy Evaluation**: Computation overhead is distributed over access time
 * - **Memory Efficient**: Optimal for partial access patterns
 *
 * ## Trade-offs
 *
 * **Benefits:**
 * - Lower initial memory allocation
 * - Efficient for sparse access patterns
 * - Allows partial materialization of large views
 * - Maintains parent reference efficiency for unaccessed elements
 *
 * **Costs:**
 * - Per-access overhead for coordinate transformation
 * - Additional memory overhead for caching metadata
 * - Potential synchronization complexity in multi-threaded access
 * - Parent tensor cannot be garbage collected until full materialization
 *
 * ## Usage Scenarios
 *
 * This strategy is optimal when:
 * - Only a subset of tensor elements will be accessed
 * - Memory is constrained and immediate full copying is not feasible
 * - Access patterns are sparse or unknown at materialization time
 * - The parent tensor should remain available for other operations
 *
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 */
public class LazyMaterializationStrategy<T : DType, V> : MaterializationStrategy<T, V> {

    override val name: String = "LazyMaterialization"

    override fun materialize(view: TensorView<T, V>): Tensor<T, V> {
        // Create a lazy materialized tensor that defers copying until access
        return LazyMaterializedTensor(
            originalView = view,
            dtype = view.dtype,
            ops = view.ops
        )
    }

    override fun canMaterialize(view: TensorView<T, V>): Boolean {
        // LazyMaterializationStrategy can handle any view as long as:
        // 1. The view has a valid shape
        // 2. The view remains accessible (parent tensor not GC'd)
        return try {
            view.viewShape.volume >= 0 && view.parentTensor != null
        } catch (e: Exception) {
            false
        }
    }

    override fun estimateMemoryOverhead(view: TensorView<T, V>): Long {
        // For lazy materialization, initial overhead is minimal:
        // - Cache data structure overhead
        // - Reference to original view
        // - Metadata for tracking materialized elements
        val baseCacheOverhead = 256L // Estimated overhead for cache structure
        val referenceOverhead = 8L   // Reference to view

        return baseCacheOverhead + referenceOverhead
    }

    /**
     * A tensor implementation that lazily materializes elements from a view.
     */
    private class LazyMaterializedTensor<T : DType, V>(
        private val originalView: TensorView<T, V>,
        override val dtype: KClass<T>,
        override val ops: TensorOps
    ) : Tensor<T, V> {
        override val gradState: GradState<T, V> = originalView.gradState

        override val data: TensorData<T, V> = LazyMaterializedTensorData(originalView)

        /**
         * Lazy tensor data implementation with sparse element caching.
         */
        private class LazyMaterializedTensorData<T : DType, V>(
            // named `sourceView`, not `view`: TensorData.view is the memory-model view (SKEEP-003)
            private val sourceView: TensorView<T, V>
        ) : TensorData<T, V> {

            override val shape: Shape = sourceView.viewShape

            // Cache for materialized elements
            // Using a map to store only accessed elements
            private val elementCache = mutableMapOf<String, V>()

            override fun get(vararg indices: Int): V {
                val cacheKey = indicesToCacheKey(indices)

                // Check if element is already cached
                return elementCache[cacheKey] ?: run {
                    // Element not cached, fetch from view and cache it
                    val element = sourceView.data.get(*indices)
                    elementCache[cacheKey] = element
                    element
                }
            }

            override fun set(vararg indices: Int, value: V) {
                // For lazy materialization, we need to decide whether to:
                // 1. Update the cache only
                // 2. Update both cache and original view
                // 3. Mark as modified and handle during full materialization

                // For now, we'll update the cache and mark the element as modified
                val cacheKey = indicesToCacheKey(indices)
                elementCache[cacheKey] = value

                // Note: This creates a semantic difference from the original view
                // In a production implementation, you might want to track modifications
                // and apply them during a "commit" operation or throw an exception
                // to indicate that lazy materialized tensors are read-only
            }

            /**
             * Converts multidimensional indices to a cache key string.
             */
            private fun indicesToCacheKey(indices: IntArray): String {
                return indices.joinToString(",")
            }

            /**
             * Forces full materialization of all elements.
             * This can be called to convert the lazy tensor to a fully materialized one.
             */
            fun forceMaterialize(): Map<String, V> {
                val dimensions = shape.dimensions
                val allIndices = mutableListOf<IntArray>()

                // Generate all possible index combinations
                fun generateIndices(currentIndices: IntArray, dimension: Int) {
                    if (dimension == dimensions.size) {
                        allIndices.add(currentIndices.copyOf())
                        return
                    }

                    for (i in 0 until dimensions[dimension]) {
                        currentIndices[dimension] = i
                        generateIndices(currentIndices, dimension + 1)
                    }
                }

                generateIndices(IntArray(dimensions.size), 0)

                // Force access to all elements to populate cache
                allIndices.forEach { indices ->
                    get(*indices) // This will cache the element if not already cached
                }

                return elementCache.toMap()
            }
        }
    }

    /**
     * Provides access to force materialization of a lazy tensor.
     * This is useful for debugging or when full materialization is eventually needed.
     */
    public fun forceMaterialize(tensor: Tensor<T, V>): Tensor<T, V> {
        return if (tensor is LazyMaterializedTensor) {
            // Force materialization and return a copy-materialized version
            val copyStrategy = CopyMaterializationStrategy<T, V>()
            copyStrategy.materialize(tensor as TensorView<T, V>)
        } else {
            // Already materialized or not a lazy tensor
            tensor
        }
    }
}
