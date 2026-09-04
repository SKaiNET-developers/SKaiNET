package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

/**
 * Android's native targets get the same portable factory the Linux one uses, and for the same
 * reason: `DirectCpuExecutionContext` is pure Kotlin here, with no cinterop of its own.
 *
 * It is a copy rather than a `dependsOn(linuxMain)` on purpose. Android native is Linux-based,
 * but it is bionic, not glibc — `linuxMain` is free to grow code that assumes the latter, and
 * inheriting that silently is worse than the six duplicated lines below. The accelerated
 * kernels for these targets live in `skainet-backend-native-cpu`, not here.
 */
internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory, sk.ainet.context.schedule.Schedule) -> TensorOps {
    // Non-JVM has no ServiceLoader; register the scalar packed-quant kernels
    // (Q4_K/Q6_K/Q5_1/Q5_0/Q8_0/Q4_0) so DefaultCpuOpsBase can dispatch them.
    KernelRegistry.register(ScalarKernelProvider)
    return { factory, schedule -> DefaultCpuOps(factory, schedule) }
}


internal actual fun platformDefaultSchedule(): sk.ainet.context.schedule.Schedule =
    sk.ainet.context.schedule.Schedule.Sequential
