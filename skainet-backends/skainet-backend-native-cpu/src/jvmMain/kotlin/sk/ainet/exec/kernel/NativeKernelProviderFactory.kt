package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelProvider

/**
 * `ServiceLoader`-friendly wrapper around [NativeKernelProvider]. The
 * platform `ServiceLoader` machinery requires a public no-arg
 * constructor, which a Kotlin `object` does not expose; this factory
 * delegates every [KernelProvider] member back to the singleton.
 *
 * Listed in
 * `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider` so
 * `KernelServiceLoader.installAll()` discovers the provider on JVM
 * startup.
 */
public class NativeKernelProviderFactory : KernelProvider by NativeKernelProvider
