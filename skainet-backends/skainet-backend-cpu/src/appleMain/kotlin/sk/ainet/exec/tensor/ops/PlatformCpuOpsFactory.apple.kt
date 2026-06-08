package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    println("[SKaiNET] Using Accelerate-backed CPU operations (ARM NEON + AMX)")
    // Accelerate overrides dense FP32 matmul; packed-quant weights still flow through
    // DefaultCpuOpsBase, so register the scalar packed kernels (no ServiceLoader on Native).
    KernelRegistry.register(ScalarKernelProvider)
    return { factory -> AccelerateCpuOps(factory) }
}
