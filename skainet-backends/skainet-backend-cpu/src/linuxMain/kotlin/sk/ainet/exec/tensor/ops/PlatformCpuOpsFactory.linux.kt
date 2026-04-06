package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps =
    { factory -> DefaultCpuOps(factory) }
