package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.MemSegKernelProvider

/**
 * `ServiceLoader`-friendly wrapper around [NativeKernelProvider]. The
 * platform `ServiceLoader` machinery requires a public no-arg
 * constructor, which a Kotlin `object` does not expose; this factory
 * delegates every [KernelProvider] / [MemSegKernelProvider] member
 * back to the singleton.
 *
 * Implementing both interfaces here matters for the MemSeg lookup
 * pattern at the call site:
 *
 * ```kotlin
 * val provider = KernelRegistry.bestAvailable()       // KernelProvider
 * val memSeg = (provider as? MemSegKernelProvider)    // smart-cast
 *     ?.matmulQ4KMemSeg()
 * ```
 *
 * Without the second `by`, the factory instance the registry hands out
 * wouldn't satisfy the smart-cast even though the underlying singleton
 * implements both interfaces.
 *
 * Listed in
 * `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider` so
 * `KernelServiceLoader.installAll()` discovers the provider on JVM
 * startup.
 */
public class NativeKernelProviderFactory :
    KernelProvider by NativeKernelProvider,
    MemSegKernelProvider by NativeKernelProvider
