package sk.ainet.context

import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.tensor.ops.platformDefaultCpuOpsFactory
import sk.ainet.exec.tensor.ops.platformDefaultSchedule

public class DirectCpuExecutionContext @kotlin.jvm.JvmOverloads constructor(
    override val executionStats: ExecutionStats = ExecutionStats(),
    override val phase: Phase = Phase.EVAL,
    private val _hooks: sk.ainet.lang.nn.hooks.ForwardHooks? = null,
    override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory(),
) : ExecutionContext {

    /**
     * SKEEP-005: a context whose ops run under [schedule]. The four-parameter constructor keeps
     * its exact JVM signature (binary compatibility); this one adds the schedule.
     */
    public constructor(
        executionStats: ExecutionStats = ExecutionStats(),
        phase: Phase = Phase.EVAL,
        hooks: sk.ainet.lang.nn.hooks.ForwardHooks? = null,
        tensorDataFactory: TensorDataFactory = DenseTensorDataFactory(),
        schedule: Schedule,
    ) : this(executionStats, phase, hooks, tensorDataFactory) {
        this.scheduleOrNull = schedule
    }

    private var scheduleOrNull: Schedule? = null

    /** How this context's ops map independent work onto cores; the platform default when not given. */
    override val schedule: Schedule
        get() = scheduleOrNull ?: platformDefaultSchedule().also { scheduleOrNull = it }

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
    // Cached: the getter used to build a fresh ops instance per access, which
    // re-ran the per-instance lazy kernel resolution and allocated on every
    // `ctx.ops` touch in the eager hot loop (#949).
    private val cachedOps: TensorOps by lazy { opsFactory(tensorDataFactory, schedule) }
    override val memoryInfo: MemoryInfo
        get() = _memoryInfo
    override val observers: ExecutionObserverRegistry
        get() = observerRegistry

    override val ops: TensorOps
        get() = cachedOps

    override val hooks: sk.ainet.lang.nn.hooks.ForwardHooks?
        get() = _hooks

    /** A sibling context whose cached ops allocate through [factory] (#1146). */
    override fun withTensorDataFactory(factory: TensorDataFactory): ExecutionContext =
        DirectCpuExecutionContext(executionStats, phase, _hooks, factory, schedule = schedule)

    /** A sibling context whose cached ops run under [schedule] (SKEEP-005). */
    override fun withSchedule(schedule: Schedule): ExecutionContext =
        if (schedule === this.schedule) this else DirectCpuExecutionContext(executionStats, phase, _hooks, tensorDataFactory, schedule = schedule)
}
