package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel

/**
 * Native (FFM) [KernelProvider]. Sits at priority `100`, above
 * [PanamaVectorKernelProvider] (`50`) and the scalar reference (`0`).
 *
 * PR 1 of the staged native-FFM rollout (see the `native-ffm-plan`
 * asciidoc) only ships the module scaffolding: the Gradle ↔ CMake
 * pipeline that produces a host-arch shared library, its bundling into
 * JAR resources, and an end-to-end FFM smoke downcall test. No real
 * matmul kernel is wired into the public SPI yet.
 *
 * Until [NativeQ4KMatmulKernel] (or its `MemSegment`-input sibling)
 * lands in PR 2, this provider deliberately reports `isAvailable() =
 * false` and returns `null` from every kernel accessor. That keeps
 * `KernelRegistry.bestAvailable()` cleanly cascading down to the
 * Panama priority-50 provider on every shape we measure today, so
 * adding the new module to the classpath produces no behavior change.
 */
public object NativeKernelProvider : KernelProvider {
    override val name: String = "native-ffm"
    override val priority: Int = 100

    override fun isAvailable(): Boolean = false

    override fun matmulFp32(): Fp32MatmulKernel? = null

    override fun matmulQ4K(): Q4KMatmulKernel? = null
}
