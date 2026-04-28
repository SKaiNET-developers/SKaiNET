package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider

/**
 * Scalar (non-SIMD) [KernelProvider] — always available, lowest
 * priority. Acts as the correctness reference and the guaranteed
 * fallback when no accelerated provider is registered.
 *
 * Callers can pin this provider directly when they want deterministic
 * scalar arithmetic without registry interaction (useful in tests):
 *
 * ```kotlin
 * val kernel = ScalarKernelProvider.matmulFp32()!!
 * kernel.matmul(a, 0, k, b, 0, n, out, 0, n, m, n, k)
 * ```
 */
public object ScalarKernelProvider : KernelProvider {
    override val name: String = "scalar"
    override val priority: Int = 0
    override fun isAvailable(): Boolean = true
    override fun matmulFp32(): Fp32MatmulKernel = ScalarMatmulKernel
}
