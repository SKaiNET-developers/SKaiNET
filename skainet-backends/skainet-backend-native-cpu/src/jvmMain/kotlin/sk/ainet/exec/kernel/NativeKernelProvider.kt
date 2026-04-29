package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel

/**
 * Native (FFM) [KernelProvider]. Sits at priority `100`, above
 * [PanamaVectorKernelProvider] (`50`) and the scalar reference (`0`).
 *
 * Availability is gated on [NativeQ4KMatmulKernel.isAvailable] — the
 * bundled `libskainet_kernels` shared library has to load AND the
 * `skainet_q4k_matmul` symbol has to resolve via FFM. When either
 * fails (missing arch, sandbox, JDK without FFM, kill-switch),
 * `KernelRegistry.bestAvailable()` cleanly cascades to
 * [PanamaVectorKernelProvider] at priority 50.
 *
 * PR 2 of the staged rollout: real Q4_K matmul wired into the SPI.
 * `matmulFp32` follows in a later PR alongside a native FP32 kernel.
 */
public object NativeKernelProvider : KernelProvider {
    override val name: String = "native-ffm"
    override val priority: Int = 100

    override fun isAvailable(): Boolean = NativeQ4KMatmulKernel.isAvailable()

    override fun matmulFp32(): Fp32MatmulKernel? = null

    override fun matmulQ4K(): Q4KMatmulKernel? =
        if (NativeQ4KMatmulKernel.isAvailable()) NativeQ4KMatmulKernel else null
}
