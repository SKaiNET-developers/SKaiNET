package sk.ainet.context

import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * [base] with a [schedule] (SKEEP-005): the deployment-level knob that says how the independent
 * work inside an op is mapped onto cores. A schedule is never a DSL word — a model is defined once
 * and runs sequentially or in parallel depending only on the context it is given.
 *
 * Mirrors [ScopedExecutionContext]: the base is rebuilt through [ExecutionContext.withSchedule] so
 * the *ops instance* carries the schedule (ops are constructed per context), and tensors created
 * through this context are bound to those ops. The same ops-binding rule applies: `a + b`
 * dispatches through the ops that created `a`, so create inputs through the scheduled context.
 *
 * A base that cannot rebuild itself keeps its own ops and reports the unhonoured request as a
 * [sk.ainet.lang.memory.trace.TraceEvent.ScheduleDowngraded] — visible, never silent.
 */
public class ScheduledExecutionContext(
    private val base: ExecutionContext,
    override val schedule: Schedule,
) : ExecutionContext by base {

    private val scheduledBase: ExecutionContext = base.withSchedule(schedule)

    override val ops: TensorOps get() = scheduledBase.ops

    override val tensorDataFactory: TensorDataFactory get() = scheduledBase.tensorDataFactory

    override fun <T : DType, V> fromData(data: TensorData<T, V>, dtype: KClass<T>): Tensor<T, V> =
        OpsBoundTensor.fromData(data, dtype, ops)

    override fun withSchedule(schedule: Schedule): ExecutionContext =
        if (schedule === this.schedule) this else ScheduledExecutionContext(base, schedule)

    /** Keep the schedule when a further decorator (e.g. `forwardScope`) swaps the data factory. */
    override fun withTensorDataFactory(factory: TensorDataFactory): ExecutionContext =
        ScheduledExecutionContext(base.withTensorDataFactory(factory), schedule)
}

/**
 * Run [block] with [schedule] active on this context:
 *
 * ```kotlin
 * ctx.withSchedule(CoroutineSchedule.hardware()) { scheduled ->
 *     model.forward(input, scheduled)      // attention heads etc. run in parallel, results unchanged
 * }
 * ```
 */
public inline fun <R> ExecutionContext.withSchedule(
    schedule: Schedule,
    block: (ctx: ExecutionContext) -> R,
): R = block(ScheduledExecutionContext(this, schedule))
