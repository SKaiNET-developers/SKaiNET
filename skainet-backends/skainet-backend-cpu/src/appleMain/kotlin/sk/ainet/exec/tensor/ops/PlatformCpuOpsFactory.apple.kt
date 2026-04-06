package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    println("[SKaiNET] Using Accelerate-backed CPU operations (ARM NEON + AMX)")
    return { factory -> AccelerateCpuOps(factory) }
}
