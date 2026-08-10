package sk.ainet.exec.tensor.ops

import java.util.ServiceLoader
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

internal actual fun platformDefaultCpuOpsFactory(): (TensorDataFactory) -> TensorOps {
    // ART supports java.util.ServiceLoader, so Android discovers kernel
    // providers the same way the JVM does (#920): modules like
    // skainet-backend-jni-cpu ship a META-INF/services entry and are
    // picked up here at priority 100. Discovery failures are non-fatal —
    // a provider that can't load just doesn't register.
    runCatching {
        ServiceLoader.load(KernelProvider::class.java).forEach { provider ->
            runCatching { KernelRegistry.register(provider) }
        }
    }
    // Scalar reference last: priority 0, always available — the floor the
    // registry cascades to when no accelerated provider carries a kernel.
    KernelRegistry.register(ScalarKernelProvider)
    return { factory -> DefaultCpuOps(factory) }
}
