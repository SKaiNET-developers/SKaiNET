package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider

internal actual fun platformDefaultCpuOpsFactory(): (sk.ainet.lang.tensor.data.TensorDataFactory, sk.ainet.context.schedule.Schedule) -> sk.ainet.lang.tensor.ops.TensorOps {
    KernelRegistry.register(ScalarKernelProvider)
    return { factory, schedule -> DefaultCpuOps(factory, schedule) }
}


internal actual fun platformDefaultSchedule(): sk.ainet.context.schedule.Schedule =
    sk.ainet.context.schedule.Schedule.Sequential
