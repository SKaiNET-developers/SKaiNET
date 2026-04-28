package sk.ainet.backend.api.kernel

import java.util.ServiceLoader

/**
 * JVM auto-discovery for [KernelProvider] implementations via the
 * standard [ServiceLoader] mechanism.
 *
 * Each backend module that ships a provider declares it in
 * `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider` (one
 * fully-qualified class name per line). Implementations need a
 * **public no-arg constructor**, so Kotlin `object` providers must be
 * exposed via a thin loader class:
 *
 * ```kotlin
 * public class MyProviderFactory : KernelProvider by MyProvider
 * ```
 *
 * Auto-discovery is JVM-only on purpose: `ServiceLoader` doesn't exist
 * on Kotlin/Native, JS, or Wasm targets. Those platforms continue to
 * use [KernelRegistry.register] directly.
 *
 * Typical startup wiring on JVM:
 *
 * ```kotlin
 * // Register every provider visible on the classpath.
 * KernelServiceLoader.installAll()
 *
 * val kernel = KernelRegistry.bestAvailable()?.matmulFp32()
 *     ?: error("no FP32 matmul kernel available")
 * ```
 *
 * Idempotent: re-installing the same providers is a no-op (the
 * registry deduplicates by instance identity).
 */
public object KernelServiceLoader {

    /**
     * Returns every [KernelProvider] discovered on the current
     * thread's context class loader. Order is unspecified —
     * [KernelRegistry] re-sorts by priority on insertion.
     */
    public fun discover(): List<KernelProvider> {
        val loader = ServiceLoader.load(KernelProvider::class.java)
        return loader.toList()
    }

    /**
     * Discovers providers via [discover] and registers each into
     * [KernelRegistry]. Returns the names of providers that were
     * successfully registered, in registration order.
     */
    public fun installAll(): List<String> {
        val providers = discover()
        for (p in providers) KernelRegistry.register(p)
        return providers.map { it.name }
    }
}
