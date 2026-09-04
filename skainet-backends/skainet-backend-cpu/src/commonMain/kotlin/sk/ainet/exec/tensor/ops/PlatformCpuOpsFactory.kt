package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

/** Ops constructor for this platform; the second argument is the [sk.ainet.context.schedule.Schedule] the ops run under (SKEEP-005). */
internal expect fun platformDefaultCpuOpsFactory(): (TensorDataFactory, sk.ainet.context.schedule.Schedule) -> TensorOps

/**
 * The schedule a `DirectCpuExecutionContext` runs under when none is given: core-count coroutines
 * on the JVM (today's matmul parallelism, now visible), [sk.ainet.context.schedule.Schedule.Sequential]
 * everywhere else.
 */
internal expect fun platformDefaultSchedule(): sk.ainet.context.schedule.Schedule
