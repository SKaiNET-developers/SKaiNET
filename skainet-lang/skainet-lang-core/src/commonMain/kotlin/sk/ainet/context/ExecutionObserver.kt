package sk.ainet.context

import sk.ainet.lang.tensor.Tensor

/**
 * Observer that can tap into execution events produced by an [ExecutionContext].
 *
 * Implementations can be used to collect metrics, trace operations or build
 * higher-level benchmark tooling.
 */
public interface ExecutionObserver {
    /**
     * Called before the backend executes an operation.
     */
    public fun onOpStart(
        context: ExecutionContext,
        opName: String,
        inputs: List<Tensor<*, *>>
    ) {}

    /**
     * Called after a successful operation. `result` can be a single tensor, a
     * collection of tensors, or any backend specific payload.
     */
    public fun onOpEnd(
        context: ExecutionContext,
        opName: String,
        result: Any?
    ) {}

    /**
     * Called if the backend throws while evaluating an operation.
     */
    public fun onOpError(
        context: ExecutionContext,
        opName: String,
        error: Throwable
    ) {}

    /**
     * Called whenever a tensor is materialised inside the context (e.g. as the
     * result of an operation or via factory helpers).
     */
    public fun onTensorMaterialized(
        context: ExecutionContext,
        tensor: Tensor<*, *>
    ) {}
}

public interface ResettableExecutionObserver : ExecutionObserver {
    public fun reset()
}

/**
 * Registry maintaining execution observers and providing helper methods to
 * broadcast events.
 */
public class ExecutionObserverRegistry {
    private val observers: MutableList<ExecutionObserver> = mutableListOf()

    public fun register(observer: ExecutionObserver) {
        if (observers.contains(observer)) return
        observers.add(observer)
    }

    public fun unregister(observer: ExecutionObserver) {
        observers.remove(observer)
    }

    public fun clear() {
        observers.clear()
    }

    public fun <R> notifyOp(
        context: ExecutionContext,
        opName: String,
        inputs: List<Tensor<*, *>>,
        block: () -> R
    ): R {
        val snapshot = observers.toList()
        if (snapshot.isEmpty()) {
            return block()
        }
        snapshot.forEach { it.onOpStart(context, opName, inputs) }
        return try {
            val result = block()
            snapshot.forEach { it.onOpEnd(context, opName, result) }
            when (result) {
                is Tensor<*, *> -> notifyTensorMaterialized(context, result, snapshot)
                is Collection<*> -> result.filterIsInstance<Tensor<*, *>>()
                    .forEach { notifyTensorMaterialized(context, it, snapshot) }
            }
            result
        } catch (t: Throwable) {
            snapshot.forEach { it.onOpError(context, opName, t) }
            throw t
        }
    }

    public fun notifyTensorMaterialized(
        context: ExecutionContext,
        tensor: Tensor<*, *>
    ) {
        notifyTensorMaterialized(context, tensor, observers.toList())
    }

    private fun notifyTensorMaterialized(
        context: ExecutionContext,
        tensor: Tensor<*, *>,
        snapshot: List<ExecutionObserver>
    ) {
        if (snapshot.isEmpty()) return
        snapshot.forEach { it.onTensorMaterialized(context, tensor) }
    }
}
