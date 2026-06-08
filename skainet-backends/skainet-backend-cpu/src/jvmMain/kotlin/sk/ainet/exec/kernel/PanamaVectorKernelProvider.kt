package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Bf16MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5_1MatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel
import sk.ainet.exec.tensor.ops.JvmCpuBackendConfig

/**
 * JVM Vector API (`jdk.incubator.vector`) [KernelProvider]. Available
 * when the runtime is JDK 21+, the incubator module is loaded
 * (`--add-modules jdk.incubator.vector`), and the
 * `skainet.cpu.vector.enabled` kill switch hasn't been flipped to
 * `false`.
 *
 * Priority is `50` — above [ScalarKernelProvider] (`0`) and below a
 * future hand-tuned native provider (`100`). Concrete kernels are
 * exposed via the per-kernel accessors; today only [matmulFp32] is
 * specialized — other accessors fall back to `null` so callers can
 * cascade to a lower-priority provider.
 *
 * Registration is **manual** (per the kernel-SPI contract today): the
 * runtime that wants this provider must call
 * `KernelRegistry.register(PanamaVectorKernelProvider)` at startup.
 * Auto-registration via `ServiceLoader` will be layered on once a
 * second concrete JVM provider exists.
 */
public object PanamaVectorKernelProvider : KernelProvider {
    override val name: String = "panama-vector"
    override val priority: Int = 50

    private val cachedAvailable: Boolean by lazy {
        isJdk21Plus() && isVectorApiClassLoaded()
    }

    override fun isAvailable(): Boolean =
        cachedAvailable && JvmCpuBackendConfig.vectorEnabled

    override fun matmulFp32(): Fp32MatmulKernel? =
        if (isAvailable()) PanamaVectorMatmulKernel else null

    override fun matmulQ4K(): Q4KMatmulKernel? =
        if (isAvailable()) PanamaVectorQ4KMatmulKernel else null

    override fun matmulBf16(): Bf16MatmulKernel? =
        if (isAvailable()) PanamaVectorBf16MatmulKernel else null

    override fun matmulQ8_0(): Q8_0MatmulKernel? =
        if (isAvailable()) PanamaVectorQ8_0MatmulKernel else null

    override fun matmulQ4_0(): Q4_0MatmulKernel? =
        if (isAvailable()) PanamaVectorQ4_0MatmulKernel else null

    override fun matmulQ5_1(): Q5_1MatmulKernel? =
        if (isAvailable()) PanamaVectorQ5_1MatmulKernel else null

    override fun matmulQ5_0(): Q5_0MatmulKernel? =
        if (isAvailable()) PanamaVectorQ5_0MatmulKernel else null

    private fun isVectorApiClassLoaded(): Boolean = runCatching {
        Class.forName("jdk.incubator.vector.FloatVector")
        Class.forName("jdk.incubator.vector.VectorSpecies")
        true
    }.getOrElse { false }

    private fun isJdk21Plus(): Boolean {
        val runtimeFeature = runCatching {
            val runtimeClass = Class.forName("java.lang.Runtime")
            val versionMethod = runtimeClass.getMethod("version")
            val versionObj = versionMethod.invoke(Runtime.getRuntime())
            val featureMethod = versionObj.javaClass.getMethod("feature")
            featureMethod.invoke(versionObj) as Int
        }.getOrNull()
        if (runtimeFeature != null) return runtimeFeature >= 21

        val spec = System.getProperty("java.specification.version") ?: return false
        return spec.toIntOrNull()?.let { it >= 21 } ?: run {
            val major = spec.split('.', '-').firstOrNull()?.toIntOrNull() ?: return@run false
            major >= 21
        }
    }
}
