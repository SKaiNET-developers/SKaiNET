package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider

internal actual fun platformDefaultCpuOpsFactory(): (sk.ainet.lang.tensor.data.TensorDataFactory) -> sk.ainet.lang.tensor.ops.TensorOps {
    KernelRegistry.register(ScalarKernelProvider)
    return { factory -> DefaultCpuOps(factory) }
}
