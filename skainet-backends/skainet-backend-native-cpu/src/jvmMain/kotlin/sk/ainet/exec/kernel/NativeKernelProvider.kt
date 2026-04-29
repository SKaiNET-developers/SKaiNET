package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.MemSegKernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4KMemSegMatmulKernel

/**
 * Native (FFM) [KernelProvider] / [MemSegKernelProvider]. Sits at
 * priority `100`, above [PanamaVectorKernelProvider] (`50`) and the
 * scalar reference (`0`).
 *
 * Availability is gated on [NativeQ4KMatmulKernel.isAvailable] — the
 * bundled `libskainet_kernels` shared library has to load AND
 * `skainet_q4k_matmul` has to resolve via FFM. When either fails
 * (missing arch, sandbox, JDK without FFM, kill-switch),
 * `KernelRegistry.bestAvailable()` cleanly cascades to
 * [PanamaVectorKernelProvider] at priority 50.
 *
 * The MemSeg surface ([matmulQ4KMemSeg]) is the JVM-only zero-copy
 * path for mmap'd Q4_K weights — sized for inference loops that
 * project against pre-loaded `MemorySegment`-backed tensors. Heap
 * callers stick with [matmulQ4K]; both wrap the same C symbol so
 * outputs are bit-for-bit identical.
 *
 * Staged rollout cursor (see `native-ffm-plan` asciidoc):
 *  - PR 2: real Q4_K matmul wired into the heap SPI.
 *  - PR 3: MemSeg-input zero-copy sibling.
 *  - PR 5 (this commit): native FP32 matmul wired into [matmulFp32].
 *  - Later: native `matmulQ6K`, `matmulQ8_0` (need new SPI accessors).
 */
public object NativeKernelProvider : KernelProvider, MemSegKernelProvider {
    override val name: String = "native-ffm"
    override val priority: Int = 100

    override fun isAvailable(): Boolean = NativeQ4KMatmulKernel.isAvailable()

    override fun matmulFp32(): Fp32MatmulKernel? =
        if (NativeFp32MatmulKernel.isAvailable()) NativeFp32MatmulKernel else null

    override fun matmulQ4K(): Q4KMatmulKernel? =
        if (NativeQ4KMatmulKernel.isAvailable()) NativeQ4KMatmulKernel else null

    override fun matmulQ4KMemSeg(): Q4KMemSegMatmulKernel? =
        if (NativeQ4KMemSegMatmulKernel.isAvailable()) NativeQ4KMemSegMatmulKernel else null
}
