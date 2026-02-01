package sk.ainet.context

import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps

public class DefaultDataExecutionContext(
    override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory(),
    override val phase: Phase = Phase.EVAL,
    private val _hooks: sk.ainet.lang.nn.hooks.ForwardHooks? = null
) : ExecutionContext {

    // class instance of voidOps created only once
    private companion object Companion {
        val voidOps = VoidTensorOps()
    }

    override val ops: TensorOps
        get() = voidOps
    private val observerRegistry = ExecutionObserverRegistry()
    override val observers: ExecutionObserverRegistry
        get() = observerRegistry

    override val hooks: sk.ainet.lang.nn.hooks.ForwardHooks?
        get() = _hooks


    override val memoryInfo: MemoryInfo
        get() = emptyMemoryInfo()

    private fun emptyMemoryInfo(): MemoryInfo = MemoryInfo.getEmptyInfo()

    override val executionStats: ExecutionStats
        get() = ExecutionStats()
}
