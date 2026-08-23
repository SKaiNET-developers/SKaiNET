package sk.ainet.context

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.scratch.NoopScratchPool
import sk.ainet.lang.tensor.scratch.ScratchPool
import sk.ainet.lang.tensor.storage.MemoryPlanner
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

public interface ExecutionContext {
    /**
     * Where this context's trace events go (SKEEP-003 §4.9): phases, kernel runs, adapter
     * insertions, allocations. Default [sk.ainet.lang.memory.trace.NoopTraceSink] — nothing is
     * recorded until a context opts in with a recording or exporting sink.
     */
    @sk.ainet.lang.memory.ExperimentalMemoryApi
    public val traceSink: sk.ainet.lang.memory.trace.TraceSink get() = sk.ainet.lang.memory.trace.NoopTraceSink

    /**
     * The scope new activations and adapter outputs are allocated in (SKEEP-003 §4.5). Default
     * [sk.ainet.lang.memory.Scope.Ambient] — GC-managed, today's behaviour; a generation loop opts
     * in by providing a `ForwardScope` and calling `reset()` per step.
     */
    @sk.ainet.lang.memory.ExperimentalMemoryApi
    public val memoryScope: sk.ainet.lang.memory.Scope get() = sk.ainet.lang.memory.Scope.Ambient

    public val ops: TensorOps

    // Optional forward hooks for recording or diagnostics (null → disabled)
    public val hooks: sk.ainet.lang.nn.hooks.ForwardHooks? get() = null

    // Execution phase and convenience training flag
    public val phase: Phase
    public val inTraining: Boolean get() = phase == Phase.TRAIN

    /**
     * Whether this context is currently *recording* a trace/graph (vs. plain eager execution).
     * Defaults to `false` (eager); the graph/tape context overrides it. Modules with an eager
     * fast-path that bypasses `ops.*` (e.g. RoPE's raw-array interleaved rotation) can check this
     * to emit a graph-traceable `ctx.ops.*` path instead when recording, so they export to
     * StableHLO while keeping the fast path for eager inference.
     */
    public val isRecording: Boolean get() = false

    public val tensorDataFactory: TensorDataFactory

    /**
     * Workspace allocator for short-lived intermediate buffers (attention
     * scratch, RoPE tables, KV-cache slice copies, padding scratch, etc.).
     *
     * Default is [NoopScratchPool] — every acquire allocates a fresh array,
     * matching pre-pool behavior. Implementations that want pooling override
     * this property (or wrap an existing context).
     *
     * Callers MUST acquire inside an active [ScratchPool.scope] block;
     * acquires outside a scope succeed but the buffer is not returned to the
     * pool when dropped.
     */
    public val scratch: ScratchPool get() = NoopScratchPool

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

    /**
     * Lazy-initialized zero tensor — see [TensorDataFactory.placeholder].
     * The underlying primitive array allocates on first read; if the parameter
     * is replaced before any read (the common case for DSL modules whose weights
     * are loaded from disk), the allocation is skipped entirely.
     */
    public fun <T : DType, V> placeholder(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val data = tensorDataFactory.placeholder<T, V>(shape, dtype)
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
