package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

/**
 * Portable fallback — identical to [PlatformCpuOpsFactory.linux.kt]'s, no
 * accelerated (NEON cinterop) path yet for this target. See
 * skainet-backend-native-cpu's own doc comments: its NEON kernels are built
 * for Linux aarch64 (glibc, the SL2610 board target), not Android's bionic
 * 32-bit ABI — wiring a real accelerated path here is separate, larger work.
 */
internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    // Non-JVM has no ServiceLoader; register the scalar packed-quant kernels
    // (Q4_K/Q6_K/Q5_1/Q5_0/Q8_0/Q4_0) so DefaultCpuOpsBase can dispatch them.
    KernelRegistry.register(ScalarKernelProvider)
    return { factory -> DefaultCpuOps(factory) }
}
