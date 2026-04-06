package sk.ainet.context

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.storage.MemoryPlanner
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

public interface ExecutionContext {
    public val ops: TensorOps

    // Optional forward hooks for recording or diagnostics (null → disabled)
    public val hooks: sk.ainet.lang.nn.hooks.ForwardHooks? get() = null

    // Execution phase and convenience training flag
    public val phase: Phase
    public val inTraining: Boolean get() = phase == Phase.TRAIN

    public val tensorDataFactory: TensorDataFactory

    // Execution observers for tracing/benchmarking
    public val observers: ExecutionObserverRegistry

    public fun registerObserver(observer: ExecutionObserver) {
        observers.register(observer)
    }

    public fun unregisterObserver(observer: ExecutionObserver) {
        observers.unregister(observer)
    }

    public fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): Tensor<T, V> {
        val data = tensorDataFactory.full<T, V>(shape, dtype, value)
        return fromData(data, dtype)
    }


    public fun <T : DType, V> zeros(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val data = tensorDataFactory.zeros<T, V>(shape, dtype)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> ones(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val data = tensorDataFactory.ones<T, V>(shape, dtype)
        return fromData(data, dtype)
    }


    public fun <T : DType, V> fromData(data: TensorData<T, V>, dtype: KClass<T>): Tensor<T, V> = OpsBoundTensor.fromData(
        data, dtype,
        ops
    )

    public fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromFloatArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromIntArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> fromByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromByteArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    /**
     * Wraps a FloatArray without copying (borrow semantics).
     * The caller must ensure the array is not mutated while the tensor is in use.
     */
    public fun <T : DType, V> wrapFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): Tensor<T, V> {
        val tensorData = tensorDataFactory.wrapFloatArray<T, V>(shape, dtype, data)
        return fromData(tensorData, dtype)
    }

    /**
     * Wraps an IntArray without copying (borrow semantics).
     * The caller must ensure the array is not mutated while the tensor is in use.
     */
    public fun <T : DType, V> wrapIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): Tensor<T, V> {
        val tensorData = tensorDataFactory.wrapIntArray<T, V>(shape, dtype, data)
        return fromData(tensorData, dtype)
    }

    /**
     * Wraps a ByteArray without copying (borrow semantics).
     * The caller must ensure the array is not mutated while the tensor is in use.
     */
    public fun <T : DType, V> wrapByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): Tensor<T, V> {
        val tensorData = tensorDataFactory.wrapByteArray<T, V>(shape, dtype, data)
        return fromData(tensorData, dtype)
    }

    // runtime information
    public val memoryInfo: MemoryInfo
    public val executionStats: ExecutionStats

    /** Memory planner for resolving placement intents. Default: CPU-only. */
    public val memoryPlanner: MemoryPlanner get() = MemoryPlanner()

    /** Memory tracker for observability and copy tracing. Default: no-op (not tracking). */
    public val memoryTracker: MemoryTracker? get() = null
}
