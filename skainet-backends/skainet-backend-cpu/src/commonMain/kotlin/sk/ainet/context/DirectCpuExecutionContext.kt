package sk.ainet.context

import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.exec.tensor.ops.platformDefaultCpuOpsFactory

public class DirectCpuExecutionContext @kotlin.jvm.JvmOverloads constructor(
    override val executionStats: ExecutionStats = ExecutionStats(),
    override val phase: Phase = Phase.EVAL,
    private val _hooks: sk.ainet.lang.nn.hooks.ForwardHooks? = null,
    override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory(),
) : ExecutionContext {

    public companion object {
        /**
         * Creates a new DirectCpuExecutionContext with sensible defaults.
         * This is the primary entry point for Java developers.
         *
         * @param phase The execution phase (EVAL or TRAIN). Defaults to EVAL.
         * @return A new DirectCpuExecutionContext instance.
         */
        @kotlin.jvm.JvmStatic
        @kotlin.jvm.JvmOverloads
        public fun create(
            phase: Phase = Phase.EVAL,
        ): DirectCpuExecutionContext = DirectCpuExecutionContext(phase = phase)
    }
    private val observerRegistry = ExecutionObserverRegistry()
    private val _memoryInfo = MemoryInfo(
        totalMemory = 0,
        usedMemory = 0,
        freeMemory = 0,
        usagePercentage = 0.0
    )
    private val opsFactory = platformDefaultCpuOpsFactory()
    override val memoryInfo: MemoryInfo
        get() = _memoryInfo
    override val observers: ExecutionObserverRegistry
        get() = observerRegistry

    override val ops: TensorOps
        get() = opsFactory(tensorDataFactory)

    override val hooks: sk.ainet.lang.nn.hooks.ForwardHooks?
        get() = _hooks
}
