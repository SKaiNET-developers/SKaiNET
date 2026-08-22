package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * The core tensor abstraction in the SKaiNET framework, representing a fundamental
 * architectural decision to compose tensors from data and operations.
 *
 * ## Fundamental Architectural Decision: Tensor Composition
 *
 * This interface embodies a key architectural principle in the SKaiNET framework:
 * **tensors are composed of two distinct, complementary components:**
 *
 * 1. **Data Component (`TensorData`)** - Responsible for:
 *    - Multi-dimensional data storage and indexing
 *    - Memory layout and access patterns
 *    - Shape and dimensional metadata
 *    - Type-safe element access
 *
 * 2. **Operations Component (`TensorOps`)** - Responsible for:
 *    - Mathematical operations and transformations
 *    - Computational algorithms
 *    - Operation chaining and composition
 *    - Performance-optimized implementations
 *
 * ## Benefits of This Compositional Architecture
 *
 * **Separation of Concerns**: Data management is cleanly separated from computational
 * logic, making the codebase more maintainable and easier to understand.
 *
 * **Flexibility**: Different data storage strategies (dense, sparse, distributed) can
 * be combined with different operation implementations (CPU, GPU, specialized hardware)
 * without tight coupling.
 *
 * **Performance Optimization**: Data layout can be optimized independently from
 * computational algorithms, enabling targeted performance improvements.
 *
 * **Extensibility**: New data formats or operation types can be added without
 * modifying existing code, following the open-closed principle.
 *
 * **Testability**: Data and operations can be tested independently, improving
 * test coverage and reducing complexity.
 *
 * ## Usage Pattern
 *
 * The tensor acts as a unified interface that delegates data access to the `data`
 * component and computational operations to the `ops` component, providing a
 * seamless experience while maintaining the benefits of compositional design.
 *
 * @param T the data type constraint extending DType, defining the numerical precision
 * @param V the actual value type that will be stored and accessed
 */
public interface Tensor<T : DType, V> {
    /**
     * The data component responsible for storage, indexing, and memory management.
     *
     * This component handles all aspects related to data storage and access,
     * including multidimensional indexing, shape management, and memory layout
     * optimization. It provides the foundation for the tensor's data model.
     */
    public val data: TensorData<T, V>

    /**
     * The operations component responsible for computational algorithms and transformations.
     *
     * This component provides all mathematical operations that can be performed
     * on the tensor, from basic arithmetic to complex neural network operations.
     * It leverages the data component for element access while encapsulating
     * all computational logic.
     */
    public val ops: TensorOps

    /**
     * The data type descriptor defining the numerical precision and value representation.
     *
     * This property provides metadata about the tensor's data type, enabling
     * type-safe operations and appropriate memory allocation strategies.
     */
    public val dtype: KClass<T>

    /**
     * Stable identity of this tensor within its model (`model.layers[3].attn.q_proj.weight`), or
     * `null` for an anonymous tensor. Assigned from the module tree by `assignTensorIds`; carried by
     * implementations of [TensorIdBearer]. Default `null` so existing implementations are unaffected.
     */
    public val id: TensorId? get() = null

    /**
     * Gradient state tied to this tensor instance.
     */
    public val gradState: GradState<T, V>

    /**
     * Flag indicating if this tensor participates in gradient computation.
     */
    public val requiresGrad: Boolean
        get() = gradState.requiresGrad

    /**
     * Accumulated gradient, if any.
     */
    public val grad: Tensor<T, V>?
        get() = gradState.grad

    /**
     * Accumulate a gradient tensor onto this tensor.
     */
    public fun accumulateGrad(g: Tensor<T, V>) {
        gradState.accumulate(g)
    }

    /**
     * Reset accumulated gradient.
     */
    public fun zeroGrad() {
        gradState.zero()
    }

    /**
     * The shape descriptor inherited from the data component.
     *
     * This property delegates to the data component's shape, maintaining consistency
     * between the tensor interface and its underlying data representation.
     */
    public val shape: Shape
        get() = data.shape

    /**
     * The total number of elements in the tensor.
     *
     * This computed property provides quick access to the tensor's total size,
     * calculated from the shape's volume. Useful for memory allocation and
     * iteration operations.
     */
    public val volume: Int
        get() = shape.volume

    /**
     * The number of dimensions in the tensor.
     *
     * This computed property provides quick access to the tensor's dimensionality,
     * derived from the shape's rank. Essential for dimension-aware operations
     * and broadcasting compatibility checks.
     */
    public val rank: Int
        get() = shape.rank

}
